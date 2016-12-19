#include <cstdlib>
#include <string>
#include <unordered_map>

#include <boost/algorithm/string/split.hpp>

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

#include "service/cmdline.hpp"

#include "vts-libs/vts.hpp"
#include "vts-libs/vts/encoder.hpp"
#include "vts-libs/vts/meshop.hpp"
#include "vts-libs/vts/csconvertor.hpp"
#include "vts-libs/registry/po.hpp"
#include "vts-libs/vts/heightmap.hpp"

#include "tinyxml2/tinyxml2.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <opencv2/highgui/highgui.hpp>

#include "tilemapping.hpp"

#include "data/empty.jpg.hpp"

namespace vs = vadstena::storage;
namespace vr = vadstena::registry;
namespace vts = vadstena::vts;
namespace tools = vadstena::vts::tools;
namespace po = boost::program_options;
namespace ba = boost::algorithm;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;
namespace xml = tinyxml2;

namespace {

math::Point3 point3(const aiVector3D &vec)
{
    return {vec.x, vec.y, vec.z};
}


//// LodTreeExport.xml parse ///////////////////////////////////////////////////

struct LodTreeNode
{
    double radius, minRange;
    math::Point3 origin;
    fs::path modelPath;
    std::vector<LodTreeNode> children;

    LodTreeNode(xml::XMLElement *elem, const fs::path &dir,
                const math::Point3 &rootOrigin);
};

struct LodTreeExport
{
    std::string srs;
    math::Point3 origin;
    std::vector<LodTreeNode> blocks;

    LodTreeExport(const fs::path &xmlPath, const math::Point3 &offset);
};


xml::XMLElement* getElement(xml::XMLNode *node, const char* elemName)
{
    xml::XMLElement* elem = node->FirstChildElement(elemName);
    if (!elem) {
        LOGTHROW(err3, std::runtime_error)
            << "XML element \"" << elemName << "\" not found.";
    }
    return elem;
}

void errorAttrNotFound(xml::XMLElement *elem, const char* attrName)
{
    LOGTHROW(err3, std::runtime_error)
        << "XML attribute \"" << attrName
        << "\" not found in element \"" << elem->Name() << "\".";
}

const char* getTextAttr(xml::XMLElement *elem, const char* attrName)
{
    const char* text = elem->Attribute(attrName);
    if (!text) {
        errorAttrNotFound(elem, attrName);
    }
    return text;
}

double getDoubleAttr(xml::XMLElement *elem, const char* attrName)
{
    double a;
    if (elem->QueryDoubleAttribute(attrName, &a) == xml::XML_NO_ATTRIBUTE) {
        errorAttrNotFound(elem, attrName);
    }
    return a;
}

xml::XMLElement* loadLodTreeXml(const fs::path &fname, xml::XMLDocument &doc)
{
    auto err = doc.LoadFile(fname.native().c_str());
    if (err != xml::XML_SUCCESS) {
        LOGTHROW(err3, std::runtime_error)
            << "Error loading " << fname << ": " << doc.ErrorName();
    }

    auto *root = getElement(&doc, "LODTreeExport");

    double version = getDoubleAttr(root, "version");
    if (version > 1.1 + 1e-12) {
        LOGTHROW(err3, std::runtime_error)
            << fname << ": unsupported format version (" << version << ").";
    }

    return root;
}

LodTreeNode::LodTreeNode(tinyxml2::XMLElement *node, const fs::path &dir,
                         const math::Point3 &rootOrigin)
{
    int ok = xml::XML_SUCCESS;
    if (getElement(node, "Radius")->QueryDoubleText(&radius) != ok ||
        getElement(node, "MinRange")->QueryDoubleText(&minRange) != ok)
    {
        LOGTHROW(err3, std::runtime_error) << "Error reading node data";
    }

    auto *ctr = getElement(node, "Center");
    math::Point3 center(getDoubleAttr(ctr, "x"),
                        getDoubleAttr(ctr, "y"),
                        getDoubleAttr(ctr, "z"));
    origin = rootOrigin + center;

    auto* mpath = node->FirstChildElement("ModelPath");
    if (mpath) {
        modelPath = dir / mpath->GetText();
    }

    // load all children
    std::string strNode("Node");
    for (auto *elem = node->FirstChildElement();
         elem;
         elem = elem->NextSiblingElement())
    {
        if (strNode == elem->Name())
        {
            children.emplace_back(elem, dir, rootOrigin);
        }
    }
}

LodTreeExport::LodTreeExport(const fs::path &xmlPath,
                             const math::Point3 &offset)
{
    xml::XMLDocument doc;
    auto *root = loadLodTreeXml(xmlPath, doc);

    srs = getElement(root, "SRS")->GetText();

    auto *local = getElement(root, "Local");
    origin(0) = getDoubleAttr(local, "x");
    origin(1) = getDoubleAttr(local, "y");
    origin(2) = getDoubleAttr(local, "z");
    origin += offset;

    // load all blocks ("Tiles")
    std::string strTile("Tile");
    for (auto *elem = root->FirstChildElement();
         elem;
         elem = elem->NextSiblingElement())
    {
        if (strTile == elem->Name())
        {
            fs::path path(getTextAttr(elem, "path"));
            if (path.is_relative()) {
                path = xmlPath.parent_path() / path;
            }
            LOG(info3) << "Parsing block " << path << ".";

            xml::XMLDocument tileDoc;
            auto *tileRoot = loadLodTreeXml(path, tileDoc);
            auto *rootNode = getElement(tileRoot, "Tile");

            blocks.emplace_back(rootNode, path.parent_path(), origin);
        }
    }
}


//// utility main //////////////////////////////////////////////////////////////

struct InputTile : public tools::InputTile
{
    const LodTreeNode *node;

    mutable int loadCnt; // stats (how many times loaded to cache)

    InputTile(int id, int depth, const LodTreeNode *node)
        : tools::InputTile(id, depth), node(node)
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
    unsigned int ntLodPixelSize;
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

    cmdline.add_options()
        ("input", po::value(&input_)->required()
         , "Path to LODTreeExport.xml input file.")

        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")

        ("overwrite", "Existing tile set gets overwritten if set.")

        ("id", po::value(&config_.tileSetId)->required()
         , "Output tileset ID.")

        ("referenceFrame", po::value(&config_.referenceFrame)->required()
         , "Output reference frame.")

        ("textureQuality", po::value(&config_.textureQuality)
         ->default_value(config_.textureQuality)->required()
         , "Texture quality for JPEG texture encoding (0-100).")

        ("credits", po::value<std::string>()
         , "Comma-separated list of string/numeric credit id.")

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

    createMode_ = (vars.count("overwrite")
                   ? vts::CreateMode::overwrite
                   : vts::CreateMode::failIfExists);

    if (vars.count("credits")) {
        std::vector<std::string> parts;
        for (const auto &value
                 : ba::split(parts, vars["credits"].as<std::string>()
                             , ba::is_any_of(",")))
        {
            vr::Credit credit;
            try {
                credit = vr::system.credits(boost::lexical_cast<int>(value));
            } catch (boost::bad_lexical_cast) {
                credit = vr::system.credits(value);
            }

            config_.credits.insert(credit.numericId);
        }
    }
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
    vts::RawAtlas atlas;
    std::mutex loadMutex;

    void load(const fs::path &path, const math::Point3 &origin);

    typedef std::shared_ptr<Model> pointer;
};


std::string textureFile(const aiScene *scene, const aiMesh *mesh, int channel)
{
    aiString texFile;
    aiMaterial *mat = scene->mMaterials[mesh->mMaterialIndex];
    mat->Get(AI_MATKEY_TEXTURE_DIFFUSE(channel), texFile);
    return {texFile.C_Str()};
}

/// Remove duplicate vertices introduced by AssImp
void optimizeMesh(vts::SubMesh &mesh)
{
    auto hash2 = [](const math::Point2 &p) -> std::size_t {
        return p(0)*218943212 + p(1)*168875421;
    };
    auto hash3 = [](const math::Point3 &p) -> std::size_t {
        return p(0)*218943212 + p(1)*168875421 + p(2)*385120205;
    };

    std::unordered_map<math::Point2, int, decltype(hash2)> map2(1024, hash2);
    std::unordered_map<math::Point3, int, decltype(hash3)> map3(1024, hash3);

    // assign unique indices to vertices and texcoords
    for (const auto &pt : mesh.vertices) {
        int &idx(map3[pt]);
        if (!idx) { idx = map3.size(); }
    }
    for (const auto &pt : mesh.tc) {
        int &idx(map2[pt]);
        if (!idx) { idx = map2.size(); }
    }

    // change face indices
    for (auto &f : mesh.faces) {
        for (int i = 0; i < 3; i++) {
            f(i) = map3[mesh.vertices[f(i)]] - 1;
        }
    }
    for (auto &f : mesh.facesTc) {
        for (int i = 0; i < 3; i++) {
            f(i) = map2[mesh.tc[f(i)]] - 1;
        }
    }

    // update vertices, tc
    mesh.vertices.resize(map3.size());
    for (const auto &item : map3) {
        mesh.vertices[item.second - 1] = item.first;
    }
    mesh.tc.resize(map2.size());
    for (const auto &item : map2) {
        mesh.tc[item.second - 1] = item.first;
    }
}

void Model::load(const fs::path &path, const math::Point3 &origin)
{
    LOG(info2) << "Loading model " << id << " (" << path << ").";

    Assimp::Importer imp;
    const aiScene *scene = imp.ReadFile(path.native(), aiProcess_Triangulate);
    if (!scene) {
        LOGTHROW(err3, std::runtime_error) << "Error loading " << path;
    }

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

        optimizeMesh(submesh);
        mesh.add(submesh);

        fs::path texPath(path.parent_path() / textureFile(scene, aimesh, 0));
        LOG(info2) << "Loading " << texPath;

        try {
            utility::ifstreambuf ifs(texPath.native(), std::ios::binary);
            vts::RawAtlas::Image buffer((std::istreambuf_iterator<char>(ifs)),
                                         std::istreambuf_iterator<char>());
            atlas.add(buffer);
        }
        catch (std::ifstream::failure e) {
            LOG(warn3) << "Error loading image " <<  texPath
                       << ", using empty texture.";
            vts::RawAtlas::Image buffer(data::empty_jpg, data::empty_jpg
                                        + sizeof(data::empty_jpg));
            atlas.add(buffer);
        }
    }
}


class ModelCache
{
public:
    ModelCache(const InputTile::list &input, unsigned cacheLimit)
        : input_(input), cacheLimit_(cacheLimit)
        , hitCnt_(), missCnt_()
    {}

    Model::pointer get(int id);

    ~ModelCache();

private:
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
    ptr->load(intile.node->modelPath, intile.node->origin);
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
    Encoder(const fs::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const InputTile::list &inputTiles
            , const geo::SrsDefinition &inputSrs
            , vs::LodRange &ntLodRange
            , int ntSourceLod, double ntSourceLodPixelSize
            , const Config &config)
        : vts::Encoder(path, properties, mode)
        , inputTiles_(inputTiles)
        , inputSrs_(inputSrs)
        , ntLodRange_(ntLodRange)
        , ntSourceLod_(ntSourceLod)
        , ntSourceLodPixelSize_(ntSourceLodPixelSize)
        , config_(config)
        , tileMap_(inputTiles, inputSrs, referenceFrame(), 1.0/*config.maxClipMargin()*/)
        , modelCache_(inputTiles, 128)
        , hma_(ntSourceLod_)
    {
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

    const InputTile::list &inputTiles_;
    const geo::SrsDefinition &inputSrs_;
    vs::LodRange ntLodRange_;
    int ntSourceLod_;
    double ntSourceLodPixelSize_;
    const Config config_;

    tools::TileMapping tileMap_;
    ModelCache modelCache_;
    vts::HeightMap::Accumulator hma_;
};

/** Constructs transformation matrix that maps everything in extents into a grid
 *  of defined size so that grid (0, 0) matches the upper-left extents corner
 *  and grid(gridSize.width - 1, gridSize.width - 1) matches the lower-right
 *  extents corner.
 */
inline math::Matrix4 mesh2grid(const math::Extents2 &extents
                              , const math::Size2 &gridSize)
{
    math::Matrix4 trafo(ublas::identity_matrix<double>(4));

    auto es(size(extents));

    // scale to grid
    trafo(0, 0) =  (gridSize.width - 1) / es.width;
    trafo(1, 1) = -(gridSize.height - 1) / es.height;

    // place zero to upper-left corner
    trafo(0, 3) = -trafo(0,0)*extents.ll(0);
    trafo(1, 3) = -trafo(1,1)*extents.ll(1) + (gridSize.height - 1);

    return trafo;
}

template <typename Op>
void rasterizeMesh(const vts::SubMesh &submesh, const math::Matrix4 &trafo
                   , const math::Size2 &rasterSize, Op op)
{
    std::vector<imgproc::Scanline> scanlines;
    cv::Point3f tri[3];
    for (const auto &face : submesh.faces) {
        for (int i : { 0, 1, 2 }) {
            auto p(transform(trafo, submesh.vertices[face(i)]));
            tri[i].x = p(0); tri[i].y = p(1); tri[i].z = p(2);
        }

        scanlines.clear();
        imgproc::scanConvertTriangle(tri, 0, rasterSize.height, scanlines);

        for (const auto &sl : scanlines) {
            imgproc::processScanline(sl, 0, rasterSize.width, op);
        }
    }
}

void Encoder::generateHeightMap(const vts::TileId &tileId
                                , const vts::SubMesh &submesh
                                , const math::Extents2 &extents)
{
    cv::Mat *hm;
    UTILITY_OMP(critical)
    hm = &hma_.tile(tileId);

    // invalid heightmap value (i.e. initial value) is +oo and we take minimum
    // of all rasterized heights in given place
    rasterizeMesh(submesh, mesh2grid(extents, hma_.tileSize())
                  , hma_.tileSize()
                  , [&](int x, int y, float z)
    {
        auto &value(hm->at<float>(y, x));
        if (z < value) { value = z; }
    });
}

void warpInPlace(const vts::CsConvertor &conv, vts::SubMesh &sm)
{
    // just convert vertices
    for (auto &v : sm.vertices) {
        // convert vertex in-place
        v = conv(v);
    }
}

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
    vts::RawAtlas::pointer patlas([&]() -> vts::RawAtlas::pointer
    {
        auto atlas(std::make_shared<vts::RawAtlas>());
        tile.atlas = atlas;
        return atlas;
    }());
    vts::RawAtlas &atlas(*patlas);

    // clip and add all source meshes (+atlases) to the output
    for (const auto &model : srcModels) {
        int smIndex(0);
        for (const auto &submesh : model->mesh)
        {
            // copy mesh and convert it to destination SDS...
            vts::SubMesh copy(submesh);
            warpInPlace(src2DstSds, copy);

            // ...where we clip it
            vts::SubMesh clipped(vts::clip(copy, clipExtents));

            if (!clipped.empty()) {
                // update mesh coverage mask
                updateCoverage(mesh, clipped, nodeInfo.extents());

                // rasterize heightmap
                if (tileId.lod == ntSourceLod_) {
                    generateHeightMap(tileId, clipped, nodeInfo.extents());
                }

                // convert to destination physical SRS
                warpInPlace(sds2DstPhy, clipped);

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

    // merge submeshes if allowed
    std::tie(tile.mesh, tile.atlas)
        = mergeSubmeshes(tileId, tile.mesh, patlas, config_.textureQuality);

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
    vts::HeightMap hm(std::move(hma_), referenceFrame()
                 , config_.dtmExtractionRadius / ntSourceLodPixelSize_);

    vts::HeightMap::BestPosition bestPosition;

    for (int lod = ntLodRange_.max; lod >= ntLodRange_.min; --lod)
    {
        // resize heightmap for given lod
        hm.resize(lod);

        // generate and store navtiles
        traverse(ts.tileIndex(), lod
                 , [&](const vts::TileId &tileId, vts::QTree::value_type mask)
        {
            // process only tiles with mesh
            if (!(mask & vts::TileIndex::Flag::mesh)) { return; }

            if (auto nt = hm.navtile(tileId)) {
                ts.setNavTile(tileId, *nt);
            }
        });

        if (lod == ntLodRange_.max) {
            bestPosition = hm.bestPosition();
        }
    }

    // set tileset default position -- TODO
/*    {
        vts::CsConvertor sds2nav(???,
                                 referenceFrame().model.navigationSrs);

        vr::Position pos;
        pos.position = sds2nav(bestPosition.location);

        pos.type = vr::Position::Type::objective;
        pos.heightMode = vr::Position::HeightMode::fixed;
        pos.orientation = { 0.0, -90.0, 0.0 };
        pos.verticalExtent = bestPosition.verticalExtent;
        pos.verticalFov = 90;
        ts.setPosition(pos);
    }*/
}


//// main //////////////////////////////////////////////////////////////////////

void collectInputTiles(const LodTreeNode &node, unsigned depth,
                       unsigned maxDepth, InputTile::list &list)
{
    if (!node.modelPath.empty()) {
        list.emplace_back(list.size(), depth, &node);
    }
    if (depth < maxDepth) {
        for (const auto &ch : node.children) {
            collectInputTiles(ch, depth+1, maxDepth, list);
        }
    }
}

int imageArea(const fs::path &path)
{
    // try to get the size without loading the whole image
    std::string ext(path.extension().string()), jpg(".jpg"), jpeg(".jpeg");
    if (boost::iequals(ext, jpg) || boost::iequals(ext, jpeg))
    {
        try {
            utility::ifstreambuf f(path.native());
            return area(imgproc::jpegSize(f, path));
        }
        catch (...) {}
    }

    // fallback: do load the image
    cv::Mat img = cv::imread(path.native());
    if (img.empty()) {
        LOGTHROW(err3, std::runtime_error) << "Could not load " << path;
    }
    return img.rows * img.cols;
}

/** Calculate and set tile.extents, tile.sdsArea, tile.texArea
 */
void calcModelExtents(InputTile &tile, const vts::CsConvertor &csconv)
{
    fs::path path(tile.node->modelPath);

    Assimp::Importer imp;
    const aiScene *scene = imp.ReadFile(path.native(), aiProcess_Triangulate);
    if (!scene) {
        LOGTHROW(err3, std::runtime_error) << "Error loading " << path
                            << "( " << imp.GetErrorString() << " )";
    }

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
        int imgArea(imageArea(texPath));

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
    math::Point3 offset(config_.offsetX, config_.offsetY, config_.offsetZ);
    if (norm_2(offset) > 0.) {
        LOG(info4) << "Using offset " << offset;
    }

    // parse the XMLs
    LOG(info4) << "Parsing " << input_;
    LodTreeExport lte(input_, offset);

    lte.origin += offset;

    // TODO: error checking (empty?)
    auto inputSrs(geo::SrsDefinition::fromString(lte.srs));

    // find a suitable reference frame division node
    boost::optional<vts::CsConvertor> convToSds;
    vr::ReferenceFrame::Division::Node sdsNode;
    {
        const auto &rf(vr::system.referenceFrames(config_.referenceFrame));
        int maxLod(-1);
        for (const auto &pair : rf.division.nodes) {
            const auto &node(pair.second);
            if (!node.valid()) { continue; }

            const vts::CsConvertor csconv(inputSrs, node.srs);
            if (math::inside(node.extents, csconv(lte.origin))
                && node.id.lod > maxLod)
            {
                convToSds = csconv;
                sdsNode = node;
                maxLod = node.id.lod;
            }
        }
        if (!convToSds) {
            LOGTHROW(err3, std::runtime_error)
                << "Couldn't find reference frame node for " << lte.origin;
        }
    }

    // create a list of InputTiles
    InputTile::list inputTiles;
    for (const auto& block : lte.blocks) {
        collectInputTiles(block, 0, config_.maxLevel, inputTiles);
    }

    // determine extents of the input tiles
    UTILITY_OMP(parallel for)
    for (unsigned i = 0; i < inputTiles.size(); i++)
    {
        auto &tile(inputTiles[i]);
        LOG(info2) << "Getting extents of " << tile.node->modelPath;
        calcModelExtents(tile, *convToSds);

        LOG(info1)
            << "\ntile.extents = " << std::fixed << tile.extents
            << "\ntile.sdsArea = " << tile.sdsArea
            << "\ntile.texArea = " << tile.texArea << "\n";
    }

    // calculate texel size of each tree level
    std::vector<double> texelArea;
    {
        std::vector<std::pair<double, double> > areas;
        for (const auto &tile : inputTiles)
        {
            if (size_t(tile.depth) >= areas.size()) {
                areas.emplace_back(0.0, 0.0);
            }
            auto &pair(areas[tile.depth]);
            pair.first += tile.sdsArea;
            pair.second += tile.texArea;
        }
        for (const auto &pair : areas) {
            texelArea.push_back(pair.first / pair.second);
        }
    }

    // print resolutions and warnings
    {
        int level(0);
        for (double ta : texelArea) {
            if (!level) {
                LOG(info3) << "Tree level " << level
                           << ": avg texel area = " << ta;
            }
            else {
                double factor(texelArea[level-1] / ta);
                LOG(info3) << "Tree level " << level
                           << ": avg texel area = " << ta
                           << " (resolution " << sqrt(factor)
                           << " times previous)";

                if (level && factor < 1.0) {
                    LOG(warn3)
                        << "Warning: level " << level << " has smaller "
                           "resolution than previous level. This level should "
                           "be removed (see also --maxLevel).";
                }
                else if (level && factor < 3.9) {
                    LOG(warn3)
                        << "Warning: level " << level << " does not have "
                           "double the resolution of previous level.";
                }
            }
            ++level;
        }
    }

    // LOD assignment
    vs::LodRange ntLodRange;
    int ntSourceLod(-1);
    double ntSourceLodPixelSize(1.0);
    {
        int level(0), count(0);
        double avgRootLod(0.);
        for (double txa : texelArea)
        {
            // calculate VTS lod assuming 256^2 optimal texture tiles
            double tileArea = 256*256*txa;
            double tileLod = 0.5*log2(sdsNode.extents.area() / tileArea);
            tileLod += sdsNode.id.lod;

            LOG(info3) << "Tree level " << level << " ~ VTS LOD " << tileLod;

            if (!level || (texelArea[level-1] / txa > 3.0)) // skip bad levels
            {
                avgRootLod += tileLod - level;
                ++count;
            }
            ++level;
        }
        avgRootLod /= count;
        LOG(info2) << "avgRootLod = " << avgRootLod;

        int rootLod(round(avgRootLod));
        LOG(info3) << "Placing tree level 0 at VTS LOD " << rootLod << ".";

        for (auto &tile : inputTiles) {
            tile.dstLod = rootLod + tile.depth;
            ntSourceLod = std::max(tile.dstLod, ntSourceLod);
        }

        // determine LOD for heightmap extraction
        level = 0;
        for (double txa : texelArea) {
            if (std::sqrt(txa) <= config_.ntLodPixelSize) { // TODO: correct?
                ntSourceLod = rootLod + level;
                ntSourceLodPixelSize = std::sqrt(txa); // FIXME
                break;
            }
            ++level;
        }
        LOG(info3) << "Navtile data will be extracted at LOD " << ntSourceLod;

        // TODO: is this correct?
        ntLodRange.min = rootLod;
        ntLodRange.max = ntSourceLod;
        LOG(info3) << "Navtile data will be generated in LOD range: "
                   << ntLodRange << ".";
    }

    vts::TileSetProperties properties;
    properties.referenceFrame = config_.referenceFrame;
    properties.id = config_.tileSetId;
    properties.credits.insert(config_.credits.begin(), config_.credits.end());

    // run the encoder
    LOG(info4) << "Building tile mapping.";
    Encoder enc(output_, properties, createMode_, inputTiles, inputSrs,
                ntLodRange, ntSourceLod, ntSourceLodPixelSize,
                config_);

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
