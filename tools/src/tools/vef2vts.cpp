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
#include "utility/openmp.hpp"

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
    double sigmaEditCoef;

    Config()
        : textureQuality(85), optimalTextureSize(256, 256)
        , forceWatertight(false)
        , clipMargin(1.0 / 128.)
        , sigmaEditCoef(1.5)
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
         "mesh texel size. Do not modify.")

        ("tweak.sigmaEditCoef", po::value(&config_.sigmaEditCoef)
         ->default_value(config_.sigmaEditCoef)
         , "Sigma editting coefficient. Meshes with best LOD difference from "
         "mean best LOD lower than sigmaEditCoef * sigma are assigned "
         "round(mean best LOD).")
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

math::Extents2 computeExtents(const vts::Mesh &mesh)
{
    math::Extents2 extents(math::InvalidExtents{});
    for (const auto &sm : mesh) {
        for (const auto &p : sm.vertices) {
            update(extents, p(0), p(1));
        }
    }
    return extents;
}

vts::TileRange computeTileRange(const vts::RFNode &node, vts::Lod localLod
                                , const math::Extents2 &meshExtents)
{
    vts::TileRange r(math::InvalidExtents{});
    const auto ts(vts::tileSize(node.extents, localLod));
    const auto origin(math::ul(node.extents));

    for (const auto &p : vertices(meshExtents)) {
        update(r, vts::TileRange::point_type
               ((p(0) - origin(0)) / ts.width
                , (origin(1) - p(1)) / ts.height));
    }

    return r;
}

typedef std::vector<cv::Mat> Textures;

struct Assignment {
    vts::NodeInfo node;
    double bestLod;
    math::Extents2 meshExtents;

    boost::optional<vts::Lod> lod;

    Assignment(const vts::NodeInfo &node, double bestLod
               , const math::Extents2 &meshExtents)
        : node(node), bestLod(bestLod), meshExtents(meshExtents)
    {}

    void setLod(vts::Lod localLod) {
        const auto& nodeId(node.nodeId());
        auto tileRange(computeTileRange(node.node(), localLod, meshExtents));
        auto lod = localLod + nodeId.lod;

        // convert local tilerange to global tilerange

        {
            const auto origin(vts::lowestChild(vts::point(nodeId), localLod));
            tileRange.ll += origin;
            tileRange.ur += origin;
        }

        /** FIXME: proper way compute valid tile range is to subtract all tile
         * ranges of rf. nodes below this node and process only what is left.
         *
         * For now, just check tileRange.ll.
         */

        if (vts::NodeInfo
            (node.referenceFrame(), vts::tileId(lod, tileRange.ll))
            .subtree().root().id != nodeId)
        {
            return;
        }

        // assign!
        this->lod = lod;
    }

    typedef std::map<vts::TileId, Assignment> map;
    typedef std::vector<Assignment*> plist;
};

class Cutter {
public:
    Cutter(tools::TmpTileset &tmpset, const vef::Manifest &manifest
           , const vr::ReferenceFrame &rf, const Config &config)
        : tmpset_(tmpset), manifest_(manifest), rf_(rf)
        , inputSrs_(*manifest_.srs), config_(config)
        , nodes_(vts::NodeInfo::nodes(rf_))
    {
        cut();
        tmpset_.flush();
    }

private:
    void cut();
    Assignment::map assign(const vef::Window &window);
    void analyze(std::vector<Assignment::map> &assignments);
    void windowCut(const vef::Window &window, vts::Lod lodDiff
                   , const Assignment::map &assignemnts);

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
    const vts::NodeInfo::list nodes_;
};

std::pair<double, double>
statistics(const Assignment::plist &assignments)
{
    std::pair<double, double> res(.0, .0);
    auto &mean(std::get<0>(res));
    auto &stddev(std::get<1>(res));

    // calculate mean
    for (const auto *assignment : assignments) {
        mean += assignment->bestLod;
    }
    mean /= assignments.size();

    // calculate stddev
    for (const auto *assignment : assignments) {
        stddev += math::sqr(assignment->bestLod - mean);
    }
    stddev = std::sqrt(stddev);

    return res;
}

void Cutter::analyze(std::vector<Assignment::map> &assignments)
{
    for (const auto &node : nodes_) {
        Assignment::plist nodeAssignments;
        for (auto &assignment : assignments) {
            auto fassignment(assignment.find(node.nodeId()));
            if (fassignment == assignment.end()) { continue; }
            nodeAssignments.push_back(&fassignment->second);
        }

        while (!nodeAssignments.empty()) {
            double meanLod, stddev;
            std::tie(meanLod, stddev) = statistics(nodeAssignments);

            const double diffLimit(config_.sigmaEditCoef * stddev);
            vts::Lod lod(std::round(meanLod));

            for (auto inodeAssignments(nodeAssignments.begin());
                 inodeAssignments != nodeAssignments.end(); )
            {
                auto *assignment(*inodeAssignments);

                // compute difference from mean best lod
                const double diff(std::abs(assignment->bestLod - meanLod));

                if (diff < diffLimit) {
                    // fits in range -> assign lod and remove from list
                    assignment->setLod(lod);
                    inodeAssignments = nodeAssignments.erase(inodeAssignments);
                } else {
                    ++inodeAssignments;
                }
            }
        }
    }
}

void Cutter::cut()
{
    std::vector<Assignment::map> assignments;

    std::size_t manifestWindowsSize(manifest_.windows.size());
    UTILITY_OMP(parallel for)
    for (std::size_t i = 0; i < manifestWindowsSize; ++i) {
        // calculate assignment
        const auto assignment(assign(manifest_.windows[i].lods.front()));
        // store
        UTILITY_OMP(critical)
            assignments.push_back(assignment);
    }

    analyze(assignments);

    UTILITY_OMP(parallel for)
        for (std::size_t i = 0; i < manifestWindowsSize; ++i) {
            const auto &loddedWindow(manifest_.windows[i]);
            auto &assignment(assignments[i]);

            dbglog::thread_id(loddedWindow.path.filename().string());

            LOG(info3) << "Processing window LODs from: " << loddedWindow.path;

            std::size_t loddedWindowSize(loddedWindow.lods.size());
            for (std::size_t ii = 0; ii < loddedWindowSize; ++ii) {
                windowCut(loddedWindow.lods[ii], ii, assignment);
            }
        }
}

Assignment::map Cutter::assign(const vef::Window &window)
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

    const auto &inMesh(loader.mesh());

    // process all real RF nodes
    Assignment::map assignment;
    std::size_t nodeCount(nodes_.size());
    UTILITY_OMP(parallel for)
    for (std::size_t i = 0; i < nodeCount; ++i) {
        const auto &node(nodes_[i]);

        // try to convert mesh into node's SRS
        const vts::CsConvertor conv(inputSrs_, node.srs());

        // local mesh and textures
        vts::Mesh mesh;
        std::vector<cv::Mat> textures;
        mesh.submeshes.reserve(inMesh.submeshes.size());

        for (const auto &sm : inMesh) {
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
        const auto optimalTileCount(node.extents().area() / optimalTileArea);
        const auto bestLod(0.5 * std::log2(optimalTileCount));

        if (bestLod < 0) { continue; }

        // we have best lod for this window in this SDS node, store info

        UTILITY_OMP(critical)
            assignment.insert
            (Assignment::map::value_type
             (node.nodeId()
              , Assignment(node, bestLod, computeExtents(mesh))));
    }

    // done
    return assignment;
}

void Cutter::windowCut(const vef::Window &window, vts::Lod lodDiff
                       , const Assignment::map &assignemnts)
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

    for (const auto &item : assignemnts) {
        const auto &assignment(item.second);
        if (!assignment.lod) { continue; }
        // grab lod
        auto lod(*assignment.lod);
        // check for underflow
        if (lodDiff > lod) { continue; }
        // fix lod
        lod -= lodDiff;

        const auto &node(assignment.node);
        const auto &nodeId(node.nodeId());

        // out of this node, abandon
        if (lod < nodeId.lod) { continue; }

        // try to convert mesh into node's SRS
        const vts::CsConvertor conv(inputSrs_, node.srs());

        // local mesh and textures
        vts::Mesh mesh;
        std::vector<cv::Mat> textures;
        mesh.submeshes.reserve(inMesh.submeshes.size());

        auto iinTextures(inTextures.begin());
        for (const auto &sm : inMesh) {
            const auto &texture(*iinTextures++);
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
            textures.push_back(texture);
        }

        if (mesh.submeshes.empty()) {
            // nothing left in the mesh, skip this node
            continue;
        }

        const vts::Lod localLod(lod - nodeId.lod);

        // compute local tile range
        auto tr(computeTileRange(node.node(), localLod, computeExtents(mesh)));

        // convert local tilerange to global tilerange
        {
            const auto origin(vts::lowestChild(vts::point(nodeId), localLod));
            tr.ll += origin;
            tr.ur += origin;
        }

        splitToTiles(node, lod, tr, mesh, textures);
    }
}

void Cutter::splitToTiles(const vts::NodeInfo &root
                          , vts::Lod lod, const vts::TileRange &tr
                          , const vts::Mesh &mesh, const Textures &textures)
{
    LOG(info3) << "Splitting to tiles in " << lod << "/" << tr << ".";
    typedef vts::TileRange::value_type Index;
    Index je(tr.ur(1));
    Index ie(tr.ur(0));

    for (Index j = tr.ll(1); j <= je; ++j) {
        for (Index i = tr.ll(0); i <= ie; ++i) {
            vts::TileId tileId(lod, i, j);
            const auto node(root.child(tileId));
            cutTile(node, mesh, textures);
        }
    }
}

void Cutter::cutTile(const vts::NodeInfo &node, const vts::Mesh &mesh
                     , const Textures &textures)
{
    const auto extents(vts::inflateTileExtents
                       (node.extents(), config_.clipMargin));
    vts::Mesh clipped;
    Textures clippedTextures;

    auto itextures(textures.begin());
    for (const auto &sm : mesh) {
        const auto &texture(*itextures++);

        auto m(vts::clip(sm, extents));
        if (m.empty()) { continue; }
        clipped.submeshes.push_back(std::move(m));
        clippedTextures.push_back(texture);
    }

    // TODO: pack mesh atlas and store mesh inside temporaty dataset
    // NB: atlas uses PNG (quality=0)
    vts::opencv::Atlas atlas(0);

    // store in temporary storage
    tmpset_.store(node.nodeId(), mesh, atlas);
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
