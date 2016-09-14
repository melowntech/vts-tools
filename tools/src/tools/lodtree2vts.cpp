#include <cstdlib>
#include <string>

#include "dbglog/dbglog.hpp"

#include "geometry/mesh.hpp"
#include "geometry/meshop.hpp"

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/openmp.hpp"
#include "utility/progress.hpp"

#include "service/cmdline.hpp"

#include "../vts-libs/vts.hpp"
#include "../vts-libs/vts/encoder.hpp"
#include "../vts-libs/vts/meshop.hpp"
#include "../vts-libs/vts/csconvertor.hpp"
#include "../vts-libs/registry/po.hpp"

#include "../tinyxml2/tinyxml2.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

#include <opencv2/highgui/highgui.hpp>

namespace vs = vadstena::storage;
namespace vr = vadstena::registry;
namespace vts = vadstena::vts;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
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
    math::Point3 center;
    fs::path modelPath;
    std::vector<LodTreeNode> children;

    LodTreeNode(xml::XMLElement *elem, const fs::path &dir);
};

struct LodTreeExport
{
    std::string srs;
    math::Point3 origin;
    std::vector<LodTreeNode> blocks;

    LodTreeExport(const fs::path &xmlPath);
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


LodTreeNode::LodTreeNode(tinyxml2::XMLElement *node, const fs::path &dir)
{
    int ok = xml::XML_SUCCESS;
    if (getElement(node, "Radius")->QueryDoubleText(&radius) != ok ||
        getElement(node, "MinRange")->QueryDoubleText(&minRange) != ok)
    {
        LOGTHROW(err3, std::runtime_error) << "Error reading node data";
    }

    auto *ctr = getElement(node, "Center");
    center(0) = getDoubleAttr(ctr, "x");
    center(1) = getDoubleAttr(ctr, "y");
    center(2) = getDoubleAttr(ctr, "z");

    modelPath = dir / getElement(node, "ModelPath")->GetText();

    // load all children
    std::string strNode("Node");
    for (auto *elem = node->FirstChildElement();
         elem;
         elem = elem->NextSiblingElement())
    {
        if (strNode == elem->Name())
        {
            children.emplace_back(elem, dir);
        }
    }
}


LodTreeExport::LodTreeExport(const fs::path &xmlPath)
{
    xml::XMLDocument doc;
    auto *root = loadLodTreeXml(xmlPath, doc);

    srs = getElement(root, "SRS")->GetText();

    auto *local = getElement(root, "Local");
    origin(0) = getDoubleAttr(local, "x");
    origin(1) = getDoubleAttr(local, "y");
    origin(2) = getDoubleAttr(local, "z");

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

            blocks.emplace_back(rootNode, path.parent_path());
        }
    }
}


//// utility main //////////////////////////////////////////////////////////////

struct Config {
    std::string referenceFrame;
    int textureQuality;

    Config()
        : textureQuality(85)
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

        ("referenceFrame", po::value(&config_.referenceFrame)->required()
         , "Output reference frame.")

        ("textureQuality", po::value(&config_.textureQuality)
         ->default_value(config_.textureQuality)->required()
         , "Texture quality for JPEG texture encoding (0-100).")

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


//// tile mapping //////////////////////////////////////////////////////////////

struct InputTile
{
    int id;
    int depth;  // depth in input LodTree
    int dstLod; // output (vts) LOD

    const LodTreeNode *node;
    math::Extents2 extents; // tile extents in LodTree
    //math::Size2 textureSize;

    typedef std::vector<InputTile> list;

    InputTile(int id, int depth, const LodTreeNode* node)
        : id(id), depth(depth), dstLod(), node(node), extents()
    {}
};


vts::TileRange::point_type
tiled(const math::Size2f &ts, const math::Point2 &origin
      , const math::Point2 &p)
{
    math::Point2 local(p - origin);
    return vts::TileRange::point_type(local(0) / ts.width
                                      , -local(1) / ts.height);
}

vts::TileRange tileRange(const vr::ReferenceFrame::Division::Node &node
                         , vts::Lod localLod, const math::Points2 &points
                         , double margin)
{
    const auto ts(vts::tileSize(node.extents, localLod));
    // NB: origin is in upper-left corner and Y grows down
    const auto origin(math::ul(node.extents));

    math::Size2f isize(ts.width * margin, ts.height * margin);
    std::array<math::Point2, 4> inflates{{
            { -isize.width, +isize.height }
            , { +isize.width, +isize.height }
            , { +isize.width, -isize.height }
            , { -isize.width, -isize.height }
        }};

    vts::TileRange r(math::InvalidExtents{});

    for (const auto &p : points) {
        for (const auto &inflate : inflates) {
            update(r, tiled(ts, origin, p + inflate));
        }
    }

    return r;
}

template <typename Op>
void forEachTile(const vr::ReferenceFrame &referenceFrame
                 , vts::Lod lod, const vts::TileRange &tileRange
                 , Op op)
{
    typedef vts::TileRange::value_type Index;
    for (Index j(tileRange.ll(1)), je(tileRange.ur(1)); j <= je; ++j) {
        for (Index i(tileRange.ll(0)), ie(tileRange.ur(0)); i <= ie; ++i) {
            op(vts::NodeInfo(referenceFrame, vts::TileId(lod, i, j)));
        }
    }
}

template <typename Op>
void rasterizeTiles(const vr::ReferenceFrame &referenceFrame
                    , const vr::ReferenceFrame::Division::Node &rootNode
                    , vts::Lod lod, const vts::TileRange &tileRange
                    , Op op)
{
    // process tile range
    forEachTile(referenceFrame, lod, tileRange
                , [&](const vts::NodeInfo &ni)
    {
        LOG(info1)
            << std::fixed << "dst tile: "
            << ni.nodeId() << ", " << ni.extents();

        // TODO: check for incidence with Q; NB: clip margin must be taken into
        // account

        // check for root
        if (ni.subtree().root().id == rootNode.id) {
            op(vts::tileId(ni.nodeId()));
        }
    });
}

math::Points2 projectCorners(const vr::ReferenceFrame::Division::Node &node
                             , const vts::CsConvertor &conv
                             , const math::Points2 &src)
{
    math::Points2 dst;
    try {
        for (const auto &c : src) {
            dst.push_back(conv(c));
            LOG(info1) << std::fixed << "corner: "
                       << c << " -> " << dst.back();
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


class TileMapping : boost::noncopyable
{
public:
    TileMapping(const std::vector<InputTile> inputTiles
                , const geo::SrsDefinition &inputSRS
                , const vr::ReferenceFrame &dstRf
                , double margin)
    {
        utility::Progress progress(inputTiles.size());

        for (const auto &tile : inputTiles)
        {
            const auto &srcExtents(tile.extents);
            const math::Points2 srcCorners = {
                ul(srcExtents), ur(srcExtents), lr(srcExtents), ll(srcExtents)
            };

            // for each destination division node
            for (const auto &item : dstRf.division.nodes) {
                const auto &node(item.second);
                if (!node.valid()) { continue; }

                const vts::CsConvertor csconv(inputSRS, node.srs);
                auto dstCorners(projectCorners(node, csconv, srcCorners));

                // ignore tiles that cannot be transformed
                if (dstCorners.empty()) { continue; }

                vts::Lod dstLocalLod(tile.dstLod - node.id.lod);

                // generate tile range from bounding box of projected corners
                auto tr(tileRange(node, dstLocalLod, dstCorners, margin));
                LOG(info1) << "tile range: " << tr;

                // TODO: add margin
                rasterizeTiles(dstRf, node, tile.dstLod, tr
                               , [&](const vts::TileId &id)
                {
                    sourceInfo_[id].push_back(tile.id);
                    dstTi_.set(id, 1);
                });
            }

            (++progress).report
                (utility::Progress::ratio_t(5, 1000)
                 , "building tile mapping ");
        }

        // clone dst tile index to valid tree and make it complete
        validTree_ = vts::TileIndex
            (vts::LodRange(0, dstTi_.maxLod()), &dstTi_);
        validTree_.complete();

        /*std::ofstream f("src.dbg");
        for (const auto &pair : sourceInfo_) {
            f << pair.first << ":";
            for (int id : pair.second) {
                f << " " << id;
            }
            f << "\n";
        }*/
    }

    const vts::TileIndex* validTree() const { return &validTree_; }

    /// Get a list of source tiles that project into destination 'tileId'.
    const std::vector<int>& source(const vts::TileId &tileId) const
    {
        auto fsourceInfo(sourceInfo_.find(tileId));
        if (fsourceInfo == sourceInfo_.end()) { return emptySource_; }
        return fsourceInfo->second;
    }

    std::size_t size() const { return sourceInfo_.size(); }

private:
    std::map<vts::TileId, std::vector<int>> sourceInfo_;
    vts::TileIndex dstTi_;
    vts::TileIndex validTree_;

    static const std::vector<int> emptySource_;
};

// keep empty, used as placeholder!
const std::vector<int> TileMapping::emptySource_;


//// encoder ///////////////////////////////////////////////////////////////////

class Encoder : public vts::Encoder
{
public:
    Encoder(const fs::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const LodTreeExport &lte
            , const InputTile::list &inputTiles
            , const geo::SrsDefinition &inputSrs
            , const Config &config)
        : vts::Encoder(path, properties, mode)
        , lte_(lte)
        , inputTiles_(inputTiles)
        , inputSrs_(inputSrs)
        , config_(config)
        , tileMap_(inputTiles, inputSrs, referenceFrame(), 1.0/*config.maxClipMargin()*/)
    {
        setConstraints(Constraints().setValidTree(tileMap_.validTree()));
        setEstimatedTileCount(tileMap_.size());
    }

private:
    virtual TileResult
    generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
             , const TileResult&) UTILITY_OVERRIDE;

    virtual void finish(vts::TileSet&);

    const LodTreeExport &lte_;
    const InputTile::list &inputTiles_;
    const geo::SrsDefinition &inputSrs_;
    const Config config_;

    TileMapping tileMap_;
};


//typedef std::tuple<Mesh::pointer, Atlas::pointer> InputModel;
typedef std::pair<vts::SubMesh, vts::RawAtlas> InputModel;

InputModel loadModel(const fs::path &path, const math::Point3 &center)
{
    LOG(info3) << "Loading " << path;

    Assimp::Importer imp;
    const aiScene *scene = imp.ReadFile(path.native(), 0);
    if (!scene) {
        LOGTHROW(err3, std::runtime_error) << "Error loading " << path;
    }

    // TODO: error checking
    // TODO: multiple meshes

    InputModel result;
    //for (unsigned m = 0; m < scene->mNumMeshes; m++)
    {
        aiMesh *mesh = scene->mMeshes[/*m*/0];
        for (unsigned i = 0; i < mesh->mNumVertices; i++)
        {
            math::Point3 pt(center + point3(mesh->mVertices[i]));
            result.first.vertices.push_back(pt);

            const aiVector3D &tc(mesh->mTextureCoords[0][i]);
            result.first.tc.push_back({tc.x, tc.y});
        }

        for (unsigned i = 0; i < mesh->mNumFaces; i++)
        {
            assert(mesh->mFaces[i].mNumIndices == 3);
            unsigned int* idx(mesh->mFaces[i].mIndices);
            result.first.faces.emplace_back(idx[0], idx[1], idx[2]);
            result.first.facesTc.emplace_back(idx[0], idx[1], idx[2]);
        }

        aiMaterial *mat = scene->mMaterials[mesh->mMaterialIndex];
        aiString texFile;
        mat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), texFile);

        fs::path texPath;
        if (texFile.length) {
            texPath = path.parent_path() / texFile.C_Str();
        } else {
            texPath = "/home/jakub/empty.jpg";
        }

        LOG(info3) << "Loading " << texPath;
        try {
            utility::ifstreambuf ifs(texPath.native(), std::ios::binary);
            vts::RawAtlas::Image buffer((std::istreambuf_iterator<char>(ifs)),
                                         std::istreambuf_iterator<char>());

            result.second.add(buffer);
        }
        catch (std::ifstream::failure e) {
            LOG(warn3) << "Error loading " <<  texPath;
        }
    }

    // TODO: optimize mesh!!!

    return result;
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
    // query which source meshes transform to the destination tileId
    const auto &srcTiles(tileMap_.source(tileId));
    if (srcTiles.empty()) {
        return TileResult::Result::noDataYet;
    }

    LOG(info1) << "Source tiles(" << srcTiles.size() << "): "
               << utility::join(srcTiles, ", ") << ".";

    // CS convertors
    // src -> dst SDS
    const vts::CsConvertor src2DstSds
        (inputSrs_, nodeInfo.srs());

    // dst SDS -> dst physical
    const vts::CsConvertor sds2DstPhy
        (nodeInfo.srs(), referenceFrame().model.physicalSrs);

    // load input models
    std::vector<InputModel> inputModels;
    inputModels.reserve(srcTiles.size());

    for (int id : srcTiles)
    {
        const LodTreeNode &ltnode(*inputTiles_[id].node);
        inputModels.push_back(loadModel(ltnode.modelPath,
                                        lte_.origin + ltnode.center));

        warpInPlace(src2DstSds, inputModels.back().first);
    }

    /*auto clipExtents(vts::inflateTileExtents
                     (nodeInfo.extents(), config_.clipMargin
                      , borderCondition, config_.borderClipMargin));*/
    auto clipExtents(nodeInfo.extents());

    // output
    Encoder::TileResult result;
    auto &tile(result.tile());
    vts::Mesh &mesh
        (*(tile.mesh = std::make_shared<vts::Mesh>(/*config_.forceWatertight*/)));
    vts::RawAtlas::pointer patlas([&]() -> vts::RawAtlas::pointer
    {
        auto atlas(std::make_shared<vts::RawAtlas>());
        tile.atlas = atlas;
        return atlas;
    }());
    vts::RawAtlas &atlas(*patlas);

    // clip and add all source meshes to the output ('mesh')
    for (const auto &model : inputModels)
    {
        vts::SubMesh dstSm(vts::clip(model.first, clipExtents));

        if (!dstSm.empty()) {

            // convert mesh to destination physical SRS
            warpInPlace(sds2DstPhy, dstSm);

            // add mesh
            mesh.add(dstSm);

            // copy texture
            atlas.add(model.second.get(0/*smIndex*/));

            // TODO: update credits
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

    // done:
    return result;
}


void Encoder::finish(vts::TileSet &ts)
{
    (void) ts;
#if 0
    auto position(input_.getProperties().position);

    // convert initial position -- should work
    const vts::CsConvertor nav2nav(input_.referenceFrame().model.navigationSrs
                                   , referenceFrame().model.navigationSrs);
    position.position = nav2nav(position.position);

    // store
    ts.setPosition(position);
#endif
}


//// main //////////////////////////////////////////////////////////////////////

void collectInputTiles(const LodTreeNode &node, unsigned depth,
                       InputTile::list &list)
{
    list.emplace_back(list.size(), depth, &node);
    for (const auto &ch : node.children) {
        collectInputTiles(ch, depth+1, list);
    }
}

math::Extents2 getModelExtents(const fs::path &path
                               , const math::Point3 &center)
{
    Assimp::Importer imp;
    const aiScene *scene = imp.ReadFile(path.native(), 0);
    if (!scene) {
        LOGTHROW(err3, std::runtime_error) << "Error loading " << path;
    }

    math::Extents2 extents(math::InvalidExtents{});
    for (unsigned m = 0; m < scene->mNumMeshes; m++)
    {
        aiMesh *mesh = scene->mMeshes[m];
        for (unsigned i = 0; i < mesh->mNumVertices; i++)
        {
            math::Point3 pt(center + point3(mesh->mVertices[i]));
            math::update(extents, pt);
        }
    }
    return extents;
}


int LodTree2Vts::run()
{
    // parse the XMLs
    LOG(info3) << "Parsing " << input_;
    LodTreeExport lte(input_);

    // TODO: error checking (empty?)
    geo::SrsDefinition inputSrs(geo::SrsDefinition::fromString(lte.srs));

    // create a list of InputTiles
    InputTile::list inputTiles;
    for (const auto& block : lte.blocks) {
        collectInputTiles(block, 0, inputTiles);
    }

    // determine extents of all input tiles
    for (auto &tile : inputTiles) {
        LOG(info2) << "Getting extents of " << tile.node->modelPath;
        tile.extents = getModelExtents(tile.node->modelPath,
                                       tile.node->center + lte.origin);
        // TODO?
        //tile.textureSize = ...
        LOG(info1) << "tile.extents = " << std::fixed << tile.extents;
    }

    // LOD assignment

    // ... ?
    int firstLod = 15; // ?
    //int numLevels = 9;

    for (auto &tile : inputTiles) {
        tile.dstLod = firstLod + tile.depth;
    }

    // TODO
    vts::TileSetProperties properties;
    properties.referenceFrame = config_.referenceFrame;
    properties.id = "TEST";
    properties.credits = {1};

    // run the encoder
    Encoder enc(output_, properties, createMode_,
                lte, inputTiles, inputSrs, config_);
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
