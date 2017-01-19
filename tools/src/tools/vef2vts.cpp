#include <cstdlib>
#include <string>
#include <iostream>
#include <algorithm>
#include <iterator>

#include <boost/algorithm/string/split.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

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
#include "vts-libs/vts/opencv/atlas.hpp"

#include "vef/vef.hpp"

#include "./tmptileset.hpp"
#include "./repackatlas.hpp"


namespace po = boost::program_options;
namespace bio = boost::iostreams;
namespace ba = boost::algorithm;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;
namespace vs = vadstena::storage;
namespace vr = vadstena::registry;
namespace vts = vadstena::vts;
namespace tools = vadstena::vts::tools;
namespace vef = vadstena::vef;

namespace {

struct Config {
    std::string tilesetId;
    std::string referenceFrame;
    vs::CreditIds credits;
    int textureQuality;
    math::Size2 optimalTextureSize;
    double ntLodPixelSize;
    double dtmExtractionRadius;

    bool forceWatertight;
    boost::optional<vts::LodTileRange> tileExtents;
    double clipMargin;
    double borderClipMargin;
    double sigmaEditCoef;

    Config()
        : textureQuality(85), optimalTextureSize(256, 256)
        , ntLodPixelSize(1.0), dtmExtractionRadius(40.0)
        , forceWatertight(false), clipMargin(1.0 / 128.)
        , borderClipMargin(clipMargin)
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

        ("navtileLodPixelSize"
         , po::value(&config_.ntLodPixelSize)
         ->default_value(config_.ntLodPixelSize)->required()
         , "Navigation data are generated at first LOD (starting from root) "
         "where pixel size (in navigation grid) is less or "
         "equal to this value.")

        ("dtmExtraction.radius"
         , po::value(&config_.dtmExtractionRadius)
         ->default_value(config_.dtmExtractionRadius)->required()
         , "Radius (in meters) of DTM extraction element (in meters).")

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

        ("borderClipMargin", po::value(&config_.borderClipMargin)
         , "Margin (in fraction of tile dimensions) added to tile extents "
         "where tile touches artificial border definied by tileExtents.")

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

    if (vars.count("tileExtents")) {
        config_.tileExtents = vars["tileExtents"].as<vts::LodTileRange>();
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
        tc_.emplace_back(t.x, t.y);
    }

    template <typename VertexType>
    void addFace(const int f[3], vts::Face &face
                 , const std::vector<VertexType> &vertices
                 , std::vector<VertexType> &out
                 , VertexMap &vmap)
    {
        for (int i(0); i < 3; ++i) {
            const std::size_t src(f[i]);
            // ensure space in map
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

bool loadGzippedObj(ObjLoader &loader, const fs::path &path)
{
    std::ifstream f(path.string());
    if (!f.good()) { return false; }

    bio::filtering_istream gzipped;
    gzipped.push
        (bio::gzip_decompressor(bio::gzip_params().window_bits, 1 << 16));
    gzipped.push(f);

    auto res(loader.parse(gzipped));
    f.close();
    return res;
}

bool loadObj(ObjLoader &loader, const vef::Window &window)
{
    switch (window.mesh.format) {
    case vef::Mesh::Format::obj:
        return loader.parse(window.mesh.path);

    case vef::Mesh::Format::gzippedObj:
        return loadGzippedObj(loader, window.mesh.path);
    }
    throw;
}


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

struct Assignment {
    vts::NodeInfo node;
    double bestLod;
    std::size_t lodCount; // number of LODs (except the most detail one)
    math::Extents2 meshExtents;

    vts::LodRange lodRange;

    Assignment(const vts::NodeInfo &node, double bestLod, std::size_t lodCount
               , const math::Extents2 &meshExtents)
        : node(node), bestLod(bestLod), lodCount(lodCount)
        , meshExtents(meshExtents), lodRange(vts::LodRange::emptyRange())
    {}

    void setLod(vts::Lod localLod) {
        const auto& nodeId(node.nodeId());
        auto tileRange(computeTileRange(node.node(), localLod, meshExtents));
        const auto lod(localLod + nodeId.lod);

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

        // clip LOD count if larger than local LOD
        if (lodCount > localLod) { lodCount = localLod; }

        // assign!
        lodRange.min = lod - lodCount;
        lodRange.max = lod;
    }

    typedef std::map<vts::TileId, Assignment> map;
    typedef std::vector<Assignment*> plist;
};

struct NavtileInfo {
    vts::LodRange lodRange;
    double pixelSize;

    NavtileInfo() : lodRange(vts::LodRange::emptyRange()), pixelSize() {}

    NavtileInfo(const vts::LodRange &lodRange, double pixelSize)
        : lodRange(lodRange), pixelSize(pixelSize)
    {}

    operator bool() const { return !lodRange.empty(); }

    typedef std::map<const vts::RFNode*, NavtileInfo> map;
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

    const NavtileInfo::map& ntMap() const { return ntMap_; }

private:
    void cut();
    Assignment::map assign(const vef::Window &window, std::size_t lodCount);
    void analyze(std::vector<Assignment::map> &assignments);
    void windowCut(const vef::Window &window, vts::Lod lodDiff
                   , const Assignment::map &assignemnts);

    void splitToTiles(const vts::NodeInfo &root
                      , vts::Lod lod, const vts::TileRange &tr
                      , const vts::Mesh &mesh
                      , const vts::opencv::Atlas &atlas);
    void cutTile(const vts::NodeInfo &node, const vts::Mesh &mesh
                 , const vts::opencv::Atlas &atlas);

    tools::TmpTileset &tmpset_;
    const vef::Manifest &manifest_;
    const vr::ReferenceFrame &rf_;
    const geo::SrsDefinition &inputSrs_;
    const Config &config_;
    const vts::NodeInfo::list nodes_;

    NavtileInfo::map ntMap_;
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

Assignment::plist analyzeNodeAssignments(Assignment::plist nodeAssignments
                                         , double sigmaEditCoef)
{
    Assignment::plist out;

    while (!nodeAssignments.empty()) {
        double meanLod, stddev;
        std::tie(meanLod, stddev) = statistics(nodeAssignments);

        const double diffLimit(sigmaEditCoef * stddev);
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
                if (!assignment->lodRange.empty()) {
                    out.push_back(assignment);
                }
            } else {
                ++inodeAssignments;
            }
        }
    }

    return out;
}

NavtileInfo computeNavtileInfo(const vts::NodeInfo &node
                               , Assignment::plist nodeAssignments
                               , const Config &config)
{
    // 1. compute lod range intersection and minimum lod
    // 2. compute union of extents
    vts::LodRange lr(nodeAssignments.front()->lodRange);
    vts::Lod minLod(lr.min);

    math::Extents2 extents(math::InvalidExtents{});
    for (const auto &a : nodeAssignments) {
        update(extents, a->meshExtents.ll);
        update(extents, a->meshExtents.ur);

        const auto &alr(a->lodRange);

        minLod = std::min(minLod, a->lodRange.min);

        if ((alr.max < lr.min) || (alr.min > lr.max)) {
            // ops, no intersection
            lr = vts::LodRange::emptyRange();
            break;
        }

        // fix minimum
        if (alr.min > lr.min) { lr.min = alr.min; }

        // fix maximum
        if (alr.max < lr.max) { lr.max = alr.max; }
    }

    if (lr.empty()) {
        // OOPS, no common lod range...
        return {};
    }

    // 3. find nt lod by nt lod pixelsize
    const auto nodeId(node.nodeId());

    // nt lod
    vts::Lod ntLod(lr.max);

    // sample one tile at bottom lod
    const vts::NodeInfo n(node.child
                          (vts::lowestChild(nodeId, ntLod  - nodeId.lod)));

    // tile size at bottom lod
    const auto ts(math::size(n.extents()));

    // take center of extents
    const auto ntCenter(math::center(extents));

    // navtile size (in pixels)
    auto ntSize(vts::NavTile::size());
    ntSize.width -= 1;
    ntSize.height -= 1;

    // SRS factors at mesh center
    const auto f(geo::SrsFactors(node.srsDef())(ntCenter));

    // calculate pixel size from ration between tile area (in "beans") and
    // navtile size in pixels; ration is down scaled by area of srs factors
    // scales
    auto pixelSize(std::sqrt
                   (math::area(ts)
                    / (math::area(ntSize)
                       * f.meridionalScale * f.parallelScale)));

    // find best matching lod
    while ((ntLod > lr.min) && (pixelSize < config.ntLodPixelSize)) {
        pixelSize *= 2.0;
        --ntLod;
    }

    return NavtileInfo(vts::LodRange(lr.min, ntLod), pixelSize);
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

        const auto analyzed
            (analyzeNodeAssignments(nodeAssignments, config_.sigmaEditCoef));

        if (analyzed.empty()) { continue; }

        // create navtile info mapping for this node
        if (const auto ni = computeNavtileInfo(node, analyzed, config_)) {
            ntMap_.insert(NavtileInfo::map::value_type
                          (&node.subtree().root(), ni));
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
        const auto &loddedWindow(manifest_.windows[i]);
        const auto assignment(assign(loddedWindow.lods.front()
                                     , loddedWindow.lods.size() - 1));
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

Assignment::map Cutter::assign(const vef::Window &window, std::size_t lodCount)
{
    // load mesh
    ObjLoader loader;

    LOG(info3) << "Loading window mesh from: " << window.mesh.path;
    if (!loadObj(loader, window)) {
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

        if (mesh.empty()) {
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
              , Assignment(node, bestLod, lodCount, computeExtents(mesh))));
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
    if (!loadObj(loader, window)) {
        LOGTHROW(err2, std::runtime_error)
            << "Unable to load mesh from " << window.mesh.path << ".";
    }

    if (loader.mesh().submeshes.size() != window.atlas.size()) {
        LOGTHROW(err2, std::runtime_error)
            << "Texture/submesh count mismatch in window "
            << window.path << ".";
    }

    vts::opencv::Atlas inAtlas;
    for (const auto &texture : window.atlas) {
        LOG(info3) << "Loading window texture from: " << texture.path;
        auto tex(cv::imread(texture.path.string()));
        if (!tex.data) {
            LOGTHROW(err2, std::runtime_error)
                << "Unable to load texture from " << texture.path << ".";
        }

        inAtlas.add(tex);
    }

    // get input mesh
    const auto &inMesh(loader.mesh());

    for (const auto &item : assignemnts) {
        const auto &assignment(item.second);

        // compute current lod
        if (assignment.lodRange.empty()) { continue; }
        // grab lod
        auto lod(assignment.lodRange.max);
        // check for absolute underflow
        if (lodDiff > lod) { continue; }
        // fix lod
        lod -= lodDiff;
        // check for underflow in given assignment
        if (lod < assignment.lodRange.min) { continue; }

        const auto &node(assignment.node);
        const auto &nodeId(node.nodeId());

        // out of this node, abandon
        if (lod < nodeId.lod) { continue; }

        // try to convert mesh into node's SRS
        const vts::CsConvertor conv(inputSrs_, node.srs());

        // local mesh and textures
        vts::Mesh mesh;
        vts::opencv::Atlas atlas;
        mesh.submeshes.reserve(inMesh.submeshes.size());

        std::size_t smIndex(0);
        for (const auto &sm : inMesh) {
            const auto &texture(inAtlas.get(smIndex++));
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
            atlas.add(texture);
        }

        if (mesh.empty()) {
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

        splitToTiles(node, lod, tr, mesh, atlas);
    }
}

void Cutter::splitToTiles(const vts::NodeInfo &root
                          , vts::Lod lod, const vts::TileRange &tr
                          , const vts::Mesh &mesh
                          , const vts::opencv::Atlas &atlas)
{
    LOG(info3) << "Splitting to tiles in " << lod << "/" << tr << ".";
    typedef vts::TileRange::value_type Index;
    Index je(tr.ur(1));
    Index ie(tr.ur(0));

    for (Index j = tr.ll(1); j <= je; ++j) {
        for (Index i = tr.ll(0); i <= ie; ++i) {
            vts::TileId tileId(lod, i, j);
            const auto node(root.child(tileId));
            cutTile(node, mesh, atlas);
        }
    }
}

void Cutter::cutTile(const vts::NodeInfo &node, const vts::Mesh &mesh
                     , const vts::opencv::Atlas &atlas)
{
    // compute border condition (defaults to all available)
    vts::BorderCondition borderCondition;
    if (config_.tileExtents) {
        borderCondition = vts::inside(*config_.tileExtents, node.nodeId());
        if (!borderCondition) { return; }
    }

    // compute clip extents
    const auto extents(vts::inflateTileExtents
                       (node.extents(), config_.clipMargin
                        , borderCondition, config_.borderClipMargin));

    vts::Mesh clipped;
    vts::opencv::Atlas clippedAtlas(0); // PNG!

    std::size_t smIndex(0);
    for (const auto &sm : mesh) {
        const auto &texture(atlas.get(smIndex++));

        auto m(vts::clip(sm, extents));
        if (m.empty()) { continue; }

        clipped.submeshes.push_back(std::move(m));
        clippedAtlas.add(texture);
    }

    if (clipped.empty()) { return; }

    // store in temporary storage
    const auto tileId(node.nodeId());
    tools::repack(tileId, clipped, clippedAtlas);
    tmpset_.store(tileId, clipped, clippedAtlas);
}

struct HmAccumulator {
    NavtileInfo ntInfo;
    vts::HeightMap::Accumulator hma;
    vts::CsConvertor toNavSrs;

    HmAccumulator(const vr::ReferenceFrame &rf, const vts::RFNode &node
                  , const NavtileInfo &ntInfo)
        : ntInfo(ntInfo), hma(ntInfo.lodRange.max)
        , toNavSrs(node.srs, rf.model.navigationSrs)
    {}

    typedef std::map<const vts::RFNode*, HmAccumulator> map;
};

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
        Cutter cutter(tmpset_, input.manifest(), referenceFrame(), config_);
        for (const auto &item : cutter.ntMap()) {
            accumulatorMap_.insert
                (HmAccumulator::map::value_type
                 (item.first, HmAccumulator
                  (referenceFrame(), *item.first, item.second)));
        }

        validTree_ = index_ = tmpset_.tileIndex();

        // make valid tree complete from root
        validTree_.makeAbsolute().complete();

        setConstraints(Constraints().setValidTree(&validTree_));
        setEstimatedTileCount(index_.count());
    }

private:
    virtual TileResult
    generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
             , const TileResult&)
        UTILITY_OVERRIDE;

    virtual void finish(vts::TileSet &ts);

    void rasterizeNavTile(const vts::TileId &tileId
                          , const vts::NodeInfo &nodeInfo
                          , const vts::Mesh &mesh);

    const Config config_;

    const vef::VadstenaArchive &input_;

    const geo::SrsDefinition inputSrs_;

    tools::TmpTileset tmpset_;
    vts::TileIndex index_;
    vts::TileIndex validTree_;

    HmAccumulator::map accumulatorMap_;
};

math::Size2 navpaneSizeInPixels(const math::Size2 &sizeInTiles)
{
    // NB: navtile is in grid system, border pixels are shared between adjacent
    // tiles
    auto s(vts::NavTile::size());
    return { 1 + sizeInTiles.width * (s.width - 1)
            , 1 + sizeInTiles.height * (s.height - 1) };
}

void warpInPlace(const vts::CsConvertor &conv, vts::SubMesh &sm)
{
    // just convert vertices
    for (auto &v : sm.vertices) {
        // convert vertex in-place
        v = conv(v);
    }
}

void warpInPlace(const vts::CsConvertor &conv, vts::Mesh &mesh)
{
    for (auto &sm : mesh) {
        warpInPlace(conv, sm);
    }
}


Encoder::TileResult
Encoder::generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
                  , const TileResult&)
{
    if (!index_.get(tileId)) { return TileResult::Result::noDataYet; }

    // dst SDS -> dst physical
    const vts::CsConvertor sds2DstPhy
        (nodeInfo.srs(), referenceFrame().model.physicalSrs);

    TileResult result(TileResult::Result::tile);

    // create tile
    auto &tile(result.tile());
    {
        // load tile
        const auto loaded(tmpset_.load(tileId, config_.textureQuality));

        // merge submeshes
        std::tie(tile.mesh, tile.atlas)
            = vts::mergeSubmeshes
            (tileId, std::get<0>(loaded), std::get<1>(loaded)
             , config_.textureQuality);
    }

    // generate external texture coordinates
    vts::generateEtc(*tile.mesh, nodeInfo.extents()
                     , nodeInfo.node().externalTexture);

    if (!config_.forceWatertight) {
        // generate mesh mask if not asked to make all tiles watertight
        vts::generateCoverage(*tile.mesh, nodeInfo.extents());
    }

    // rasterize navtile (only at defined lod)
    rasterizeNavTile(tileId, nodeInfo, *tile.mesh);

    // warp mesh to physical SRS
    warpInPlace(sds2DstPhy, *tile.mesh);

    // set credits
    tile.credits = config_.credits;

    // done
    return result;
}

/** Constructs transformation matrix that maps everything in extents into a grid
 *  of defined size so the grid (0, 0) matches to upper-left extents corner and
 *  grid(gridSize.width - 1, gridSize.width - 1) matches lower-right extents
 *  corner.
 */
inline math::Matrix4 mesh2grid(const math::Extents2 &extents
                              , const math::Size2 &gridSize)
{
    math::Matrix4 trafo(ublas::identity_matrix<double>(4));

    auto es(size(extents));

    // scales
    math::Size2f scale((gridSize.width - 1) / es.width
                       , (gridSize.height - 1) / es.height);

    // scale to grid
    trafo(0, 0) = scale.width;
    trafo(1, 1) = -scale.height;

    // place zero to upper-left corner
    trafo(0, 3) = -extents.ll(0) * scale.width;
    trafo(1, 3) = extents.ur(1) * scale.height;

    return trafo;
}

template <typename Op>
void rasterizeMesh(const vts::Mesh &mesh, const vts::CsConvertor &toNavSrs
                   , const math::Matrix4 &trafo
                   , const math::Size2 &rasterSize, Op op)
{
    math::Points3 vertices;
    std::vector<imgproc::Scanline> scanlines;
    cv::Point3f tri[3];

    for (const auto &sm : mesh) {
        vertices.reserve(sm.vertices.size());
        vertices.clear();
        for (auto v : sm.vertices) {
            v(2) = toNavSrs(v)(2);
            vertices.push_back(transform(trafo, v));
        }

        for (const auto &face : sm.faces) {
            for (int i(0); i < 3; ++i) {
                const auto &p(vertices[face[i]]);
                tri[i].x = p(0);
                tri[i].y = p(1);
                tri[i].z = p(2);
            }

            scanlines.clear();
            imgproc::scanConvertTriangle(tri, 0, rasterSize.height, scanlines);

            for (const auto &sl : scanlines) {
                imgproc::processScanline(sl, 0, rasterSize.width, op);
            }
        }
    }
}

void Encoder::rasterizeNavTile(const vts::TileId &tileId
                               , const vts::NodeInfo &nodeInfo
                               , const vts::Mesh &mesh)
{
    // grab accumulator
    auto faccumulatorMap(accumulatorMap_.find(&nodeInfo.subtree().root()));
    if (faccumulatorMap == accumulatorMap_.end()) { return; }
    auto &hma(faccumulatorMap->second);
    if (tileId.lod != hma.ntInfo.lodRange.max) { return; }

    auto &hm([&]() -> cv::Mat&
    {
        cv::Mat *hm(nullptr);
        UTILITY_OMP(critical(getnavtile))
            hm = &hma.hma.tile(tileId);
        return *hm;
    }());

    // invalid heightmap value (i.e. initial value) is +oo and we take minimum
    // of all rasterized heights in given place
    rasterizeMesh(mesh, hma.toNavSrs
                  , mesh2grid(nodeInfo.extents(), hma.hma.tileSize())
                  , hma.hma.tileSize(), [&](int x, int y, float z)
    {
        auto &value(hm.at<float>(y, x));
        if (z < value) { value = z; }
    });
}

void Encoder::finish(vts::TileSet &ts)
{
    boost::optional<vts::HeightMap::BestPosition> bestPosition;
    const auto &navigationSrs(referenceFrame().model.navigationSrs);

    for (auto &item : accumulatorMap_) {
        const auto &rfnode(*item.first);
        auto &hma(item.second);

        vts::HeightMap hm
            (std::move(hma.hma), referenceFrame()
             , config_.dtmExtractionRadius / hma.ntInfo.pixelSize);

        // use best position if better than previous
        {
            auto bp(hm.bestPosition());
            if (!bestPosition
                || (bp.verticalExtent > bestPosition->verticalExtent))
            {
                bp.location
                    = vts::CsConvertor(rfnode.srs, navigationSrs)
                    (bp.location);
                bestPosition = bp;
            }
        }

        const auto &lr(hma.ntInfo.lodRange);

        // iterate in nt lod range backwards: iterate from start and invert
        // forward lod into backward lod
        for (auto lod(lr.max); lod >= lr.min; --lod) {
            // resize heightmap for given lod
            hm.resize(lod);

            // generate and store navtiles
            // FIXME: traverse only part covered by current node
            traverse(ts.tileIndex(), lod
                     , [&](const vts::TileId &tileId
                           , vts::QTree::value_type mask)
            {
                // process only tiles with mesh
                if (!(mask & vts::TileIndex::Flag::mesh)) { return; }

                if (auto nt = hm.navtile(tileId)) {
                    ts.setNavTile(tileId, *nt);
                }
            });
        }
    }

    // use best position if available
    if (bestPosition) {
        vr::Position pos;
        pos.position = bestPosition->location;

        pos.type = vr::Position::Type::objective;
        pos.heightMode = vr::Position::HeightMode::fixed;
        pos.orientation = { 0.0, -90.0, 0.0 };
        pos.verticalExtent = bestPosition->verticalExtent;
        pos.verticalFov = 55;
        ts.setPosition(pos);
    }
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
