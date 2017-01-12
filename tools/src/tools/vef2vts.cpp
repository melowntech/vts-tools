#include <cstdlib>
#include <string>
#include <iostream>
#include <algorithm>
#include <iterator>

#include <boost/algorithm/string/split.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "dbglog/dbglog.hpp"
#include "utility/streams.hpp"

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/progress.hpp"
#include "utility/streams.hpp"
#include "utility/openmp.hpp"
#include "utility/progress.hpp"

#include "service/cmdline.hpp"

#include "math/transform.hpp"
#include "math/filters.hpp"

#include "imgproc/scanconversion.hpp"
#include "imgproc/jpeg.hpp"

#include "geometry/parse-obj.hpp"

#include "geo/csconvertor.hpp"
#include "geo/coordinates.hpp"

#include "vts-libs/registry/po.hpp"
#include "vts-libs/vts.hpp"
#include "vts-libs/vts/encoder.hpp"
#include "vts-libs/vts/opencv/navtile.hpp"
#include "vts-libs/vts/io.hpp"
#include "vts-libs/vts/csconvertor.hpp"
#include "vts-libs/vts/meshopinput.hpp"
#include "vts-libs/vts/meshop.hpp"
#include "vts-libs/vts/heightmap.hpp"
#include "vts-libs/vts/math.hpp"

#include "vef/vef.hpp"

#include "./tmptileset.hpp"


namespace po = boost::program_options;
namespace vs = vadstena::storage;
namespace vr = vadstena::registry;
namespace vts = vadstena::vts;
namespace tools = vadstena::vts::tools;
namespace vef = vadstena::vef;
namespace ba = boost::algorithm;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;

namespace {

struct Config {
    std::string tilesetId;
    std::string referenceFrame;
    vs::CreditIds credits;
    int textureQuality;
    math::Size2 optimalTextureSize;

    bool forceWatertight;
    int clipMargin;

    Config()
        : textureQuality(85), optimalTextureSize(256, 256)
        , forceWatertight(false)
        , clipMargin(1.0 / 128.)
    {}
};

class Vef2Vts : public service::Cmdline
{
public:
    Vef2Vts()
        : service::Cmdline("vef2vts", BUILD_TARGET_VERSION)
        , createMode_(vts::CreateMode::failIfExists)
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

void Vef2Vts::configuration(po::options_description &cmdline
                             , po::options_description &config
                             , po::positional_options_description &pd)
{
    vr::registryConfiguration(cmdline, vr::defaultPath());

    cmdline.add_options()
        ("input", po::value(&input_)->required()
         , "Path to input vadstena export format (VEF) archive.")
        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")
        ("overwrite", "Existing tile set gets overwritten if set.")

        ("tilesetId", po::value(&config_.tilesetId)->required()
         , "Output tileset ID.")

        ("referenceFrame", po::value(&config_.referenceFrame)->required()
         , "Destination reference frame. Must be different from input "
         "tileset's referenceFrame.")

        ("textureQuality", po::value(&config_.textureQuality)
         ->default_value(config_.textureQuality)->required()
         , "Texture quality for JPEG texture encoding (0-100).")

        ("credits", po::value<std::string>()
         , "Comma-separated list of string/numeric credit id.")

        ("force.watertight", po::value(&config_.forceWatertight)
         ->default_value(false)->implicit_value(true)
         , "Enforces full coverage mask to every generated tile even "
         "when it is holey.")

        ("clipMargin", po::value(&config_.clipMargin)
         ->default_value(config_.clipMargin)
         , "Margin (in fraction of tile dimensions) added to tile extents in "
         "all 4 directions.")

        ("tileExtents", po::value<vts::LodTileRange>()
         , "Optional tile extents specidied in form lod/llx,lly:urx,ury. "
         "When set, only tiles in that range and below are added to "
         "the output.")

        ("tweak.optimalTextureSize", po::value(&config_.optimalTextureSize)
         ->default_value(config_.optimalTextureSize)->required()
         , "Size of ideal tile texture. Used to calculate fitting LOD from"
         "mesh texel size. Do not modify.");
        ;

    pd.add("input", 1);
    pd.add("output", 1);

    (void) config;
}

void Vef2Vts::configure(const po::variables_map &vars)
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

    if ((config_.textureQuality < 0) || (config_.textureQuality > 100)) {
        throw po::validation_error
            (po::validation_error::invalid_option_value, "textureQuality");
    }
}

bool Vef2Vts::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(vef2vts
usage
    vef2vts INPUT OUTPUT [OPTIONS]

)RAW";
    }
    return false;
}

double bestTileArea(const math::Points2 &corners)
{
    return (vts::triangleArea(corners[0], corners[1], corners[2])
            + vts::triangleArea(corners[2], corners[3], corners[0]));
}

int bestLod(const vr::ReferenceFrame::Division::Node &node, double area)
{
    // compute longest of base node tile sizes
    auto rootSize(math::size(node.extents));
    auto rootArea(rootSize.width * rootSize.height);

    // compute number of requested tiles per edge
    auto tileCount(std::sqrt(rootArea / area));

    return int(std::round(std::log2(tileCount)));
}

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

typedef std::map<vts::TileId, vts::TileId::list> SourceInfo;

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
    } catch (std::exception) {
        // whole tile cannot be projected -> ignore
        return {};
    }

    // OK, we could convert whole tile into this reference system
    return dst;
}

class ObjLoader : public geometry::ObjParserBase {
public:
    ObjLoader()
        : textureId_(0), vMap_(), tcMap_()
    {
        // make sure we have at least one valid material
        useMaterial(0);
    }

    vts::Mesh mesh() const { return mesh_; }

private:
    typedef std::vector<int> VertexMap;
    typedef std::vector<VertexMap> VertexMaps;

    virtual void addVertex(const Vector3d &v) {
        vertices_.emplace_back(v.x, v.y, v.z);
    }

    virtual void addTexture(const Vector3d &t) {
        // NB: flip Y coordinate
        tc_.emplace_back(t.x, 1. - t.y);
    }

    template <typename VertexType>
    void addFace(const int f[3], vts::Face &face
                 , const std::vector<VertexType> &vertices
                 , std::vector<VertexType> &out
                 , VertexMap &vmap)
    {
        for (int i(0); i < 3; ++i) {
            const std::size_t src(f[i]);
            // ensure space
            if (vmap.size() <= src) { vmap.resize(src + 1, -1); }

            auto &dst(vmap[src]);
            if (dst < 0) {
                // new mapping
                dst = out.size();
                out.push_back(vertices[src]);
            }
            face(i) = dst;
        }
    }

    virtual void addFacet(const Facet &f) {
        auto &sm(mesh_.submeshes[textureId_]);
        sm.faces.emplace_back();
        addFace(f.v, sm.faces.back(), vertices_, sm.vertices, *vMap_);

        sm.facesTc.emplace_back();
        addFace(f.t, sm.facesTc.back(), tc_, sm.tc, *tcMap_);
    }

    virtual void useMaterial(const std::string &m) {
        // get new material index
        useMaterial(boost::lexical_cast<unsigned int>(m));
    }

    void useMaterial(unsigned int textureId) {
        textureId_ = textureId;

        // ensure space in all lists
        if (mesh_.submeshes.size() <= textureId_) {
            mesh_.submeshes.resize(textureId_ + 1);
            vMaps_.resize(textureId_ + 1);
            tcMaps_.resize(textureId_ + 1);

            vMap_ = &vMaps_[textureId_];
            tcMap_ = &tcMaps_[textureId_];
        }
    }

    virtual void addNormal(const Vector3d&) { /*ignored*/ }
    virtual void materialLibrary(const std::string&) { /*ignored*/ }

    math::Points3 vertices_;
    math::Points2 tc_;
    VertexMaps vMaps_;
    VertexMaps tcMaps_;

    vts::Mesh mesh_;
    unsigned int textureId_;

    VertexMap *vMap_;
    VertexMap *tcMap_;
};

typedef std::vector<cv::Mat> Textures;

class Cutter {
public:
    Cutter(tools::TmpTileset &tmpset, const vef::Manifest &manifest
           , const vr::ReferenceFrame &rf, const Config &config)
        : tmpset_(tmpset), manifest_(manifest), rf_(rf)
        , inputSrs_(*manifest_.srs), config_(config)
    {
        cut();
    }

private:
    void cut();
    void windowCut(const vef::Window &window);
    void splitToTiles(const vts::NodeInfo &root
                      , vts::Lod lod, const vts::TileRange &tr
                      , const vts::Mesh &mesh, const Textures &textures);
    void cutTile(const vts::NodeInfo &node, const vts::Mesh &mesh
                 , const Textures &textures);

    tools::TmpTileset &tmpset_;
    const vef::Manifest &manifest_;
    const vr::ReferenceFrame &rf_;
    const geo::SrsDefinition &inputSrs_;
    const Config &config_;
};

void Cutter::cut()
{
    for (const auto &loddedWindow : manifest_.windows) {
        LOG(info3) << "Processing window LODs from: " << loddedWindow.path;
        for (const auto &window : loddedWindow.lods) {
            windowCut(window);
        }
    }
}

/** Dummy convertor. We work with mesh in local coordinates only.
 */
class DummyMeshVertexConvertor : public vts::MeshVertexConvertor {
public:
    virtual math::Point3d vertex(const math::Point3d &v) const { return v; }
    virtual math::Point2d etc(const math::Point3d&) const { return {}; }
    virtual math::Point2d etc(const math::Point2d&) const { return {}; }
};

vts::TileRange computeTileRange(const vts::RFNode &node, vts::Lod localLod
                                , const vts::Mesh &mesh)
{
    vts::TileRange r(math::InvalidExtents{});
    const auto ts(vts::tileSize(node.extents, localLod));
    const auto origin(math::ul(node.extents));

    for (const auto &sm : mesh.submeshes) {
        for (const auto &p : sm.vertices) {
            update(r, vts::TileRange::point_type
                   ((p(0) - origin(0)) / ts.width
                    , (origin(1) - p(1)) / ts.height));
        }
    }

    return r;
}

void Cutter::windowCut(const vef::Window &window)
{
    // load mesh
    ObjLoader loader;
    LOG(info3) << "Loading window mesh from: " << window.mesh.path;
    if (!loader.parse(window.mesh.path)) {
        LOGTHROW(err2, std::runtime_error)
            << "Unable to load mesh from " << window.mesh.path << ".";
    }

    if (loader.mesh().submeshes.size() != window.atlas.size()) {
        LOGTHROW(err2, std::runtime_error)
            << "Texture/submesh count mismatch in window "
            << window.path << ".";
    }

    std::vector<cv::Mat> inTextures;
    for (const auto &texture : window.atlas) {
        LOG(info3) << "Loading window texture from: " << texture.path;
        inTextures.push_back(cv::imread(texture.path.string()));
        if (!inTextures.back().data) {
            LOGTHROW(err2, std::runtime_error)
                << "Unable to load texture from " << texture.path << ".";
        }
    }

    // get input mesh
    const auto &inMesh(loader.mesh());

    // process all real RF nodes
    for (const auto &node : vts::NodeInfo::nodes(rf_)) {
        const auto &nodeId(node.nodeId());

        // try to convert mesh into node's SRS
        const vts::CsConvertor conv(inputSrs_, node.srs());

        // local mesh and textures
        vts::Mesh mesh;
        std::vector<cv::Mat> textures;
        mesh.submeshes.reserve(inMesh.submeshes.size());
        textures.reserve(inTextures.size());

        auto iinTextures(inTextures.begin());
        for (const auto &sm : inMesh) {
            auto &inTexture(*iinTextures++);

            // project mesh to srs and create mask (full by default)

            // make all faces valid by default
            vts::VertexMask valid(sm.vertices.size(), true);
            math::Points3 projected;
            projected.reserve(sm.vertices.size());

            auto ivalid(valid.begin());
            for (const auto &v : sm.vertices) {
                try {
                    projected.push_back(conv(v));
                    ++ivalid;
                } catch (std::exception) {
                    // failed to convert vertex, mask it and skip
                    projected.emplace_back();
                    *ivalid++ = false;
                }
            }

            // clip mesh to node's extents
            // FIXME: implement mask application in clipping!
            auto osm(vts::clip(sm, projected, node.extents(), valid));
            if (osm.faces.empty()) { continue; }

            // at least one face survived, remember
            mesh.submeshes.push_back(std::move(osm));
            textures.push_back(inTexture);
        }

        if (mesh.submeshes.empty()) {
            // nothing left in the mesh, skip this node
            continue;
        }

        // calculate area (only valid faces)
        const auto a(area(mesh));

        // denormalize texture area
        double textureArea(.0);
        auto iasm(a.submeshes.begin());
        for (const auto &texture : window.atlas) {
            const auto &as(*iasm++);
            textureArea +=
                (as.internalTexture * math::area(texture.size));
        }

        const double texelArea(a.mesh / textureArea);
        const auto optimalTileArea
            (area(config_.optimalTextureSize) * texelArea);
        const auto bestLod
            (0.5 * std::log2(node.extents().area() / optimalTileArea));

        if (bestLod < 0) { continue; }

        // calculate local and global lod (round to closest)
        const vts::Lod localLod(std::round(bestLod));
        const vts::Lod lod(localLod + nodeId.lod);

        // compute mesh (local) tilerange at computed (local) lod
        auto tr(computeTileRange(node.node(), localLod, mesh));
        if (empty(tr)) { continue; }

        // convert local tilerange to global tilerange
        {
            const auto origin(vts::lowestChild(vts::point(nodeId), localLod));
            tr.ll += origin;
            tr.ur += origin;
        }

        /** FIXME: proper way compute valid tile range is to subtract all tile
         * ranges of rf. nodes below this node and process only what is left.
         *
         * For now, just check tr.ll.
         *
         * TODO: check for validity inside parent extents, e.g. inside polar
         * caps in melown2015.
         */

        if (vts::NodeInfo(rf_, vts::tileId(lod, tr.ll)).subtree().root().id
                          != nodeId)
        {
            // oops, some other subtree is down there
            continue;
        }

        // we have some mesh that fits here
        LOG(info3) << "<" << node.srs() << "> texelArea: " << texelArea
                   << ", best lod: " << bestLod << ", lod: " << lod
                   << ", tr: " << tr;

        splitToTiles(node, lod, tr, mesh, textures);
    }
}

void Cutter::splitToTiles(const vts::NodeInfo &root
                          , vts::Lod lod, const vts::TileRange &tr
                          , const vts::Mesh &mesh, const Textures &textures)
{
    typedef vts::TileRange::value_type Index;
    for (Index j = tr.ll(1), je = tr.ur(1); j <= je; ++j) {
        for (Index i = tr.ll(0), ie = tr.ur(0); i <= ie; ++i) {
            vts::TileId tileId(lod, i, j);
            const auto node(root.child(tileId));
            cutTile(node, mesh, textures);
        }
    }
}

void Cutter::cutTile(const vts::NodeInfo &node, const vts::Mesh &mesh
                     , const Textures &textures)
{
    (void) node;
    (void) mesh;
    (void) textures;
    LOG(info4) << "Cutting tile " << node.nodeId();
}

class Encoder : public vts::Encoder {
public:
    Encoder(const boost::filesystem::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const vef::VadstenaArchive &input
            , const Config &config)
        : vts::Encoder(path, properties, mode)
        , config_(config), input_(input)
        , inputSrs_(*input.manifest().srs)
        , tmpset_(path / "tmp")
    {
        Cutter(tmpset_, input.manifest(), referenceFrame(), config_);
    }

private:
    virtual TileResult
    generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
             , const TileResult&)
        UTILITY_OVERRIDE;

    void prepare(const vef::Manifest &manifest);

    virtual void finish(vts::TileSet &ts);

    const Config config_;

    const vef::VadstenaArchive &input_;

    const geo::SrsDefinition inputSrs_;

    tools::TmpTileset tmpset_;
};

void warpInPlace(const vts::CsConvertor &conv, vts::SubMesh &sm)
{
    // just convert vertices
    for (auto &v : sm.vertices) {
        // convert vertex in-place
        v = conv(v);
    }
}

vts::VertexMask warpInPlaceWithMask(const vts::CsConvertor &conv
                                    , vts::SubMesh &sm)
{
    vts::VertexMask mask(sm.vertices.size(), true);

    std::size_t masked(0);
    auto imask(mask.begin());
    for (auto &v : sm.vertices) {
        try {
            // convert vertex in-place
            v = conv(v);
        } catch (std::exception) {
            // cannot convert vertex -> mask out
            *imask = false;
            ++masked;
        }
        ++imask;
    }

    // nothing masked -> no mask
    if (!masked) { return {}; }

    // something masked
    return mask;
}

math::Size2 navpaneSizeInPixels(const math::Size2 &sizeInTiles)
{
    // NB: navtile is in grid system, border pixels are shared between adjacent
    // tiles
    auto s(vts::NavTile::size());
    return { 1 + sizeInTiles.width * (s.width - 1)
            , 1 + sizeInTiles.height * (s.height - 1) };
}

vts::NavTile::pointer
warpNavtiles(const vts::TileId &tileId
             , const vr::ReferenceFrame &referenceFrame
             , const vts::NodeInfo &nodeInfo
             , const vts::MeshOpInput::list &source)
{
    // TODO: Check for different lodding/SDS and process accordingly

    vts::HeightMap hm(tileId, source, referenceFrame);
    if (hm.empty()) { return {}; }
    hm.warp(nodeInfo);

    auto navtile(hm.navtile(tileId));
    if (navtile->coverageMask().empty()) { return {}; }

    return navtile;
}

Encoder::TileResult
Encoder::generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
                  , const TileResult&)
{
    return TileResult::Result::noData;
    (void) tileId;
    (void) nodeInfo;
}

void Encoder::finish(vts::TileSet &ts)
{
    (void) ts;
    // TODO: update position and navtiles
}

int Vef2Vts::run()
{
    vts::TileSetProperties properties;
    properties.referenceFrame = config_.referenceFrame;
    properties.id = config_.tilesetId;

    vef::VadstenaArchive input(input_);
    if (!input.manifest().srs) {
        LOG(fatal)
            << "Vadstena export format archive " << input_
            << " doesn't have assigned an SRS, cannot proceed.";
        return EXIT_FAILURE;
    }

    // run the encoder
    Encoder(output_, properties, createMode_, input, config_).run();

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[])
{
    return Vef2Vts()(argc, argv);
}
