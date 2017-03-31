#include <cstdlib>
#include <string>

#include "dbglog/dbglog.hpp"

#include "math/transform.hpp"

#include "geometry/mesh.hpp"
#include "geometry/meshop.hpp"

#include "imgproc/scanconversion.hpp"
#include "imgproc/jpeg.hpp"

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/openmp.hpp"
#include "utility/progress.hpp"
#include "roarchive/roarchive.hpp"

#include "service/cmdline.hpp"
#include "service/verbosity.hpp"

#include "vts-libs/vts.hpp"
#include "vts-libs/vts/encoder.hpp"
#include "vts-libs/vts/meshop.hpp"
#include "vts-libs/vts/csconvertor.hpp"
#include "vts-libs/registry/po.hpp"
#include "vts-libs/vts/ntgenerator.hpp"
#include "vts-libs/vts/opencv/atlas.hpp"



#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <opencv2/highgui/highgui.hpp>

#include "lodtreefile.hpp"

#include "./tilemapping.hpp"
#include "./importutil.hpp"

namespace vs = vtslibs::storage;
namespace vr = vtslibs::registry;
namespace vts = vtslibs::vts;
namespace tools = vtslibs::vts::tools;
namespace po = boost::program_options;
namespace ublas = boost::numeric::ublas;

namespace {

typedef vts::opencv::HybridAtlas HybridAtlas;

math::Point3 point3(const aiVector3D &vec)
{
    return {vec.x, vec.y, vec.z};
}

// io support

const aiScene* readScene(Assimp::Importer &imp
                         , const roarchive::RoArchive &archive
                         , const fs::path &path
                         , unsigned int flags)
{
    // if (archive.directio()) {
    //     return imp.ReadFile(archive.path(path).string(), flags);
    // }

    const auto buf(archive.istream(path)->read());

    const auto *scene
        (imp.ReadFileFromMemory(buf.data(), buf.size(), flags));
                                // , path.extension().c_str()));
    if (!scene) {
        LOGTHROW(err3, std::runtime_error)
            << "Error loading scene " << path
            << "( " << imp.GetErrorString() << " ).";
    }

    return scene;
}

cv::Mat readTexture(const roarchive::RoArchive &archive, const fs::path &path
                    , bool useEmpty = false)
{
    cv::Mat texture;
    // if (archive.directio()) {
    //     texture = cv::imread(archive.path(path).string(), CV_LOAD_IMAGE_COLOR);
    // } else {
        const auto buf(archive.istream(path)->read());
        texture = cv::imdecode(buf, CV_LOAD_IMAGE_COLOR);
    // }

    if (texture.data) { return texture; }

    if (!useEmpty) {
        LOGTHROW(err3, std::runtime_error)
            << "Error loading texture from " << path << ".";
    }

    LOG(warn3)
        << "Error loading image " << path << "; using empty texture.";
    texture.create(64, 64, CV_8UC3);
    texture = cv::Scalar(255, 255, 255);

    return texture;
}

//// utility main //////////////////////////////////////////////////////////////

struct InputTile : public tools::InputTile
{
    const lt::LodTreeNode *node;
    const geo::SrsDefinition *srs;

    math::Extents2 extents;
    double sdsArea, texArea;

    mutable int loadCnt; // stats (how many times loaded to cache)

    virtual math::Points2 projectCorners(
            const vr::ReferenceFrame::Division::Node &node) const;

    InputTile(int id, int depth, const lt::LodTreeNode *node,
              const geo::SrsDefinition &srs)
        : tools::InputTile(id, depth), node(node), srs(&srs)
    {}

    typedef std::vector<InputTile> list;
};

struct Config
{
    std::string tileSetId;
    std::string referenceFrame;
    vs::CreditIds credits;
    int textureQuality;
    int maxLevel;
    unsigned int ntLodPixelSize; // FIXME: int?
    double dtmExtractionRadius;
    double offsetX, offsetY, offsetZ;

    Config()
        : textureQuality(85), maxLevel(-1)
        , ntLodPixelSize(1.0), dtmExtractionRadius(40.0)
        , offsetX(0.), offsetY(0.), offsetZ(0.)
    {}
};

class LodTree2Vts : public service::Cmdline
{
public:
    LodTree2Vts()
        : service::Cmdline("lodtree2vts", BUILD_TARGET_VERSION)
    {
    }

private:
    virtual void configuration(po::options_description &cmdline
                               , po::options_description &config
                               , po::positional_options_description &pd)
        UTILITY_OVERRIDE;

    virtual void configure(const po::variables_map &vars)
        UTILITY_OVERRIDE;

    virtual bool help(std::ostream &out, const std::string &what) const
        UTILITY_OVERRIDE;

    virtual int run() UTILITY_OVERRIDE;

    fs::path input_;
    fs::path output_;

    vts::CreateMode createMode_;

    Config config_;
};

void LodTree2Vts::configuration(po::options_description &cmdline
                               , po::options_description &config
                               , po::positional_options_description &pd)
{
    vr::registryConfiguration(cmdline, vr::defaultPath());
    vr::creditsConfiguration(cmdline);

    cmdline.add_options()
        ("input", po::value(&input_)->required()
         , "Path to LODTreeExport.xml input file.")

        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")

        ("overwrite", "Existing tile set gets overwritten if set.")

        ("tilesetId", po::value(&config_.tileSetId)->required()
         , "Output tileset ID.")

        ("referenceFrame", po::value(&config_.referenceFrame)->required()
         , "Output reference frame.")

        ("textureQuality", po::value(&config_.textureQuality)
         ->default_value(config_.textureQuality)->required()
         , "Texture quality for JPEG texture encoding (0-100).")

        ("maxLevel", po::value(&config_.maxLevel)
         ->default_value(config_.maxLevel)
         , "If not -1, ignore LODTree levels > maxLevel.")

        ("navtileLodPixelSize"
         , po::value(&config_.ntLodPixelSize)
         ->default_value(config_.ntLodPixelSize)->required()
         , "Navigation data are generated at first LOD (starting from root) "
         "where rounded value of pixel size (in navigation grid) is less or "
         "equal to this value.")

        ("dtmExtraction.radius"
         , po::value(&config_.dtmExtractionRadius)
         ->default_value(config_.dtmExtractionRadius)->required()
         , "Radius (in meters) of DTM extraction element (in meters).")

        ("offsetX", po::value(&config_.offsetX)->default_value(config_.offsetX),
         "Force X shift of the model.")
        ("offsetY", po::value(&config_.offsetY)->default_value(config_.offsetY),
         "Force Y shift of the model.")
        ("offsetZ", po::value(&config_.offsetZ)->default_value(config_.offsetZ),
         "Force Z shift of the model.")

        ;

    pd.add("input", 1);
    pd.add("output", 1);

    (void) config;
}

void LodTree2Vts::configure(const po::variables_map &vars)
{
    vr::registryConfigure(vars);
    config_.credits = vr::creditsConfigure(vars);

    createMode_ = (vars.count("overwrite")
                   ? vts::CreateMode::overwrite
                   : vts::CreateMode::failIfExists);
}

bool LodTree2Vts::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(lodtree2vts
usage
    lodtree2vts INPUT OUTPUT [OPTIONS]

)RAW";
    }
    return false;
}


//// import + cache ////////////////////////////////////////////////////////////

/** Represents a model (meshes + textures) loaded in memory.
 */
struct Model
{
    Model(int id) : id(id) {}

    int id;
    vts::Mesh mesh;
    HybridAtlas atlas;
    std::mutex loadMutex;

    void load(const roarchive::RoArchive &archive
              , const fs::path &path, const math::Point3 &origin);

    typedef std::shared_ptr<Model> pointer;
};

std::string textureFile(const aiScene *scene, const aiMesh *mesh, int channel)
{
    aiString texFile;
    aiMaterial *mat = scene->mMaterials[mesh->mMaterialIndex];
    mat->Get(AI_MATKEY_TEXTURE_DIFFUSE(channel), texFile);
    return {texFile.C_Str()};
}

void Model::load(const roarchive::RoArchive &archive
                 , const fs::path &path, const math::Point3 &origin)
{
    LOG(info2) << "Loading model " << id << " (" << path << ").";

    Assimp::Importer imp;
    const auto &scene(readScene(imp, archive, path, aiProcess_Triangulate));

    // TODO: error checking

    for (unsigned m = 0; m < scene->mNumMeshes; m++)
    {
        vts::SubMesh submesh;

        aiMesh *aimesh = scene->mMeshes[m];
        for (unsigned i = 0; i < aimesh->mNumVertices; i++)
        {
            math::Point3 pt(origin + point3(aimesh->mVertices[i]));
            submesh.vertices.push_back(pt);

            const aiVector3D &tc(aimesh->mTextureCoords[0][i]);
            submesh.tc.push_back({tc.x, tc.y});
        }

        for (unsigned i = 0; i < aimesh->mNumFaces; i++)
        {
            assert(aimesh->mFaces[i].mNumIndices == 3);
            unsigned int* idx(aimesh->mFaces[i].mIndices);
            submesh.faces.emplace_back(idx[0], idx[1], idx[2]);
            submesh.facesTc.emplace_back(idx[0], idx[1], idx[2]);
        }

        // remove duplicate vertices introduced by AssImp
        tools::optimizeMesh(submesh);

        mesh.add(submesh);

        fs::path texPath(path.parent_path() / textureFile(scene, aimesh, 0));
        LOG(info2) << "Loading " << texPath;
        atlas.add(readTexture(archive, texPath, true));
    }
}

class ModelCache
{
public:
    ModelCache(const roarchive::RoArchive &archive
               , const InputTile::list &input, unsigned cacheLimit)
        : archive_(archive), input_(input), cacheLimit_(cacheLimit)
        , hitCnt_(), missCnt_()
    {}

    Model::pointer get(int id);

    ~ModelCache();

private:
    const roarchive::RoArchive &archive_;
    const InputTile::list &input_;
    unsigned cacheLimit_;
    long hitCnt_, missCnt_;

    std::list<Model::pointer> cache_;
    std::mutex mutex_;
};

Model::pointer ModelCache::get(int id)
{
    std::unique_lock<std::mutex> cacheLock(mutex_);

    // return model immediately if present in the cache
    for (auto it = cache_.begin(); it != cache_.end(); it++) {
        Model::pointer ptr(*it);
        if (ptr->id == id)
        {
            LOG(info1) << "Cache hit: model " << id;
            ++hitCnt_;

            // move the the front of the list
            cache_.erase(it);
            cache_.push_front(ptr);

            // make sure model is not being loaded and return
            std::lock_guard<std::mutex> loadLock(ptr->loadMutex);
            return ptr;
        }
    }

    LOG(info2) << "Cache miss: model " << id;
    ++missCnt_;

    // free LRU items from the cache
    while (cache_.size() >= cacheLimit_)
    {
        LOG(info1) << "Releasing model " << cache_.back()->id;
        cache_.pop_back();
    }

    // create new cache entry
    Model::pointer ptr(std::make_shared<Model>(id));
    cache_.push_front(ptr);

    // unlock cache and load data
    std::lock_guard<std::mutex> loadLock(ptr->loadMutex);
    cacheLock.unlock();

    const InputTile &intile(input_[id]);
    ptr->load(archive_, intile.node->modelPath, intile.node->origin);
    intile.loadCnt++;
    return ptr;
}

ModelCache::~ModelCache()
{
    // print stats
    double sum(0.0);
    for (const auto &tile : input_) {
        sum += tile.loadCnt;
    }
    LOG(info2) << "Cache miss/hit: " << missCnt_ << "/" << hitCnt_;
    LOG(info2) << "Tile average load count: " << sum / input_.size();
}


//// encoder ///////////////////////////////////////////////////////////////////

class Encoder : public vts::Encoder
{
public:
    Encoder(const roarchive::RoArchive &archive, const fs::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const InputTile::list &inputTiles
            , const geo::SrsDefinition &inputSrs
            , const tools::NavTileParams &nt
            , const std::string &ntsds
            , const Config &config)
        : vts::Encoder(path, properties, mode)
        , archive_(archive)
        , inputTiles_(inputTiles)
        , inputSrs_(inputSrs)
        , nt_(nt)
        , config_(config)
        , tileMap_(inputTiles, referenceFrame(), 1.0/*config.maxClipMargin()*/)
        , modelCache_(archive_, inputTiles, 128)
        , ntg_(&referenceFrame())
    {
        ntg_.addAccumulator
            (ntsds, nt.lodRange, nt.sourceLodPixelSize);

        setConstraints(Constraints().setValidTree(tileMap_.validTree()));
        setEstimatedTileCount(tileMap_.size());
    }

private:
    void generateHeightMap(const vts::TileId &tileId
                           , const vts::SubMesh &submesh
                           , const math::Extents2 &extents);

    virtual TileResult
    generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
             , const TileResult&) UTILITY_OVERRIDE;

    virtual void finish(vts::TileSet&);

    const roarchive::RoArchive &archive_;
    const InputTile::list &inputTiles_;
    const geo::SrsDefinition &inputSrs_;
    const tools::NavTileParams &nt_;
    const Config config_;

    tools::TileMapping tileMap_;
    ModelCache modelCache_;
    vts::NtGenerator ntg_;
};

Encoder::TileResult
Encoder::generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
                  , const TileResult&)
{
    // query which source models transform to the destination tileId
    const auto &srcIds(tileMap_.source(tileId));
    if (srcIds.empty()) {
        return TileResult::Result::noDataYet;
    }

    LOG(info1) << "Source models (" << srcIds.size() << "): "
               << utility::join(srcIds, ", ") << ".";

    // get the models from the cache
    std::vector<Model::pointer> srcModels;
    srcModels.reserve(srcIds.size());
    for (int id : srcIds) {
        srcModels.push_back(modelCache_.get(id));
    }

    // CS convertors
    // src -> dst SDS
    const vts::CsConvertor src2DstSds
        (inputSrs_, nodeInfo.srs());

    // dst SDS -> dst physical
    const vts::CsConvertor sds2DstPhy
        (nodeInfo.srs(), referenceFrame().model.physicalSrs);

    /*auto clipExtents(vts::inflateTileExtents
                     (nodeInfo.extents(), config_.clipMargin
                      , borderCondition, config_.borderClipMargin));*/
    auto clipExtents(nodeInfo.extents());

    // output
    Encoder::TileResult result;
    auto &tile(result.tile());
    vts::Mesh &mesh
        (*(tile.mesh = std::make_shared<vts::Mesh>(false)));
    auto patlas([&]() -> HybridAtlas::pointer
    {
        auto atlas(std::make_shared<HybridAtlas>());
        tile.atlas = atlas;
        return atlas;
    }());
    auto &atlas(*patlas);

    // clip and add all source meshes (+atlases) to the output
    for (const auto &model : srcModels) {
        int smIndex(0);
        for (const auto &submesh : model->mesh)
        {
            // copy mesh and convert it to destination SDS...
            vts::SubMesh copy(submesh);
            tools::warpInPlace(src2DstSds, copy);

            // ...where we clip it
            vts::SubMesh clipped(vts::clip(copy, clipExtents));

            if (!clipped.empty()) {
                // update mesh coverage mask
                updateCoverage(mesh, clipped, nodeInfo.extents());

                // generate external texture coordinates (if division node
                // allows)
                generateEtc(clipped, nodeInfo.extents()
                            , nodeInfo.node().externalTexture);

                // add to output
                mesh.add(clipped);

                // copy texture
                atlas.add(model->atlas.get(smIndex));
            }
            smIndex++;
        }
    }

    if (mesh.empty()) {
        // no mesh
        return TileResult::Result::noDataYet;
    }

    // add tile to navtile generator
    ntg_.addTile(tileId, nodeInfo, mesh);

    // merge submeshes if allowed
    std::tie(tile.mesh, tile.atlas)
        = vts::mergeSubmeshes
        (tileId, tile.mesh, patlas, config_.textureQuality);

    // NB: do not use `mesh` from here (broken reference!)

    // convert to destination physical SRS
    tools::warpInPlace(sds2DstPhy, *tile.mesh);

    if (atlas.empty()) {
        // no atlas -> disable
        tile.atlas.reset();
    }

    // TODO: add external texture coordinates

    // set credits
    tile.credits = config_.credits;

    // done:
    return result;
}

void Encoder::finish(vts::TileSet &ts)
{
    // generate navtiles and surrogates
    ntg_.generate(ts, config_.dtmExtractionRadius);
}

//// main //////////////////////////////////////////////////////////////////////

void collectInputTiles(const lt::LodTreeNode &node, unsigned depth,
                       unsigned maxDepth, InputTile::list &list,
                       const geo::SrsDefinition &srs)
{
    if (!node.modelPath.empty()) {
        list.emplace_back(list.size(), depth, &node, srs);
    }
    if (depth < maxDepth) {
        for (const auto &ch : node.children) {
            collectInputTiles(ch, depth+1, maxDepth, list, srs);
        }
    }
}

int imageArea(const roarchive::RoArchive &archive, const fs::path &path)
{
    // try to get the size without loading the whole image
    std::string ext(path.extension().string()), jpg(".jpg"), jpeg(".jpeg");
    if (boost::iequals(ext, jpg) || boost::iequals(ext, jpeg))
    {
        try {
            return area(imgproc::jpegSize(*archive.istream(path), path));
        }
        catch (...) {}
    }

    // fallback: do load the image
    auto img(readTexture(archive, path));

    return img.rows * img.cols;
}

/** Convert tile corners into node.srs, check if they are in node.extents.
 */
math::Points2
InputTile::projectCorners(const vr::ReferenceFrame::Division::Node &node) const
{
    math::Points2 corners = {
        ul(extents), ur(extents), lr(extents), ll(extents)
    };

    const vts::CsConvertor conv(*srs, node.srs);

    math::Points2 dst;
    dst.reserve(4);
    try {
        for (const auto &c : corners) {
            dst.push_back(conv(c));
            LOG(info1) << std::fixed << "corner: " << c << " -> " << dst.back();
            if (!inside(node.extents, dst.back())) {
                // projected dst tile cannot fit inside this node's
                // extents -> ignore
                return {};
            }
        }
    }
    catch (std::exception) {
        // whole tile cannot be projected -> ignore
        return {};
    }
    // OK, we could convert whole tile into this reference system
    return dst;
}

/** Calculate and set tile.extents, tile.sdsArea, tile.texArea
 */
void calcModelExtents(const roarchive::RoArchive &archive
                      , InputTile &tile, const vts::CsConvertor &csconv)
{
    fs::path path(tile.node->modelPath);

    Assimp::Importer imp;
    const auto *scene(readScene(imp, archive, path, aiProcess_Triangulate));

    tile.extents = math::Extents2(math::InvalidExtents{});
    tile.sdsArea = 0.0;
    tile.texArea = 0.0;

    for (unsigned m = 0; m < scene->mNumMeshes; m++)
    {
        aiMesh *mesh = scene->mMeshes[m];

        math::Points3d physPts(mesh->mNumVertices);
        for (unsigned i = 0; i < mesh->mNumVertices; i++)
        {
            math::Point3 pt(tile.node->origin + point3(mesh->mVertices[i]));
            math::update(tile.extents, pt);
            physPts[i] = csconv(pt);
        }

        if (!mesh->GetNumUVChannels()) {
            LOGTHROW(err3, std::runtime_error)
                << path << ": mesh is not textured.";
        }

        std::string texFile(textureFile(scene, mesh, 0));
        if (texFile.empty()) {
            LOGTHROW(err3, std::runtime_error)
                << path << ": mesh does not reference a texture file.";
        }

        fs::path texPath(path.parent_path() / texFile);
        int imgArea(imageArea(archive, texPath));

        for (unsigned f = 0; f < mesh->mNumFaces; f++)
        {
            aiFace &face = mesh->mFaces[f];
            assert(face.mNumIndices == 3);

            math::Point3 a(physPts[face.mIndices[0]]);
            math::Point3 b(physPts[face.mIndices[1]]);
            math::Point3 c(physPts[face.mIndices[2]]);

            tile.sdsArea += 0.5*norm_2(math::crossProduct(b - a, c - a));

            math::Point3 ta(point3(mesh->mTextureCoords[0][face.mIndices[0]]));
            math::Point3 tb(point3(mesh->mTextureCoords[0][face.mIndices[1]]));
            math::Point3 tc(point3(mesh->mTextureCoords[0][face.mIndices[2]]));

            tile.texArea += 0.5*norm_2(math::crossProduct(tb - ta, tc - ta))
                               *imgArea;
        }
    }
}


int LodTree2Vts::run()
{
    roarchive::RoArchive archive(input_, lt::mainXmlFileName);

    math::Point3 offset(config_.offsetX, config_.offsetY, config_.offsetZ);
    if (norm_2(offset) > 0.) {
        LOG(info4) << "Using offset " << offset;
    }

    // parse the XMLs
    lt::LodTreeExport lte(archive, offset);

    lte.origin += offset;

    // TODO: error checking (empty?)
    auto inputSrs(geo::SrsDefinition::fromString(lte.srs));

    // find a suitable reference frame division node
    auto sdsNode =
        tools::findSpatialDivisionNode(
                vr::system.referenceFrames(config_.referenceFrame),
                inputSrs, lte.origin);

    vts::CsConvertor csconv(inputSrs, sdsNode.srs);

    // create a list of InputTiles
    InputTile::list inputTiles;
    for (const auto& block : lte.blocks) {
        collectInputTiles(block, 0, config_.maxLevel, inputTiles, inputSrs);
    }

    // determine extents of the input tiles
    UTILITY_OMP(parallel for)
    for (unsigned i = 0; i < inputTiles.size(); i++)
    {
        auto &tile(inputTiles[i]);
        LOG(info2) << "Getting extents of " << tile.node->modelPath;
        calcModelExtents(archive, tile, csconv);

        LOG(info1)
            << "\ntile.extents = " << std::fixed << tile.extents
            << "\ntile.sdsArea = " << tile.sdsArea
            << "\ntile.texArea = " << tile.texArea << "\n";
    }

    // assign LODs to tiles based on texture resolution
    tools::NavTileParams ntParams =
            tools::assignTileLods(inputTiles, sdsNode, config_.ntLodPixelSize);

    vts::TileSetProperties properties;
    properties.referenceFrame = config_.referenceFrame;
    properties.id = config_.tileSetId;
    properties.credits.insert(config_.credits.begin(), config_.credits.end());

    // run the encoder
    LOG(info4) << "Building tile mapping.";
    Encoder enc(archive, output_, properties, createMode_, inputTiles
                , inputSrs, ntParams, sdsNode.srs, config_);

    LOG(info4) << "Encoding VTS tiles.";
    enc.run();

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[])
{
    return LodTree2Vts()(argc, argv);
}
