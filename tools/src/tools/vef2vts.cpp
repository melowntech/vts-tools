#include <cstdlib>
#include <string>
#include <iostream>
#include <algorithm>
#include <iterator>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/optional/optional_io.hpp>

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
#include "utility/limits.hpp"
#include "utility/binaryio.hpp"

#include "service/cmdline.hpp"
#include "service/verbosity.hpp"

#include "math/transform.hpp"
#include "math/filters.hpp"
#include "math/math.hpp"

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
#include "vts-libs/vts/math.hpp"
#include "vts-libs/vts/opencv/atlas.hpp"
#include "vts-libs/vts/ntgenerator.hpp"

#include "vef/vef.hpp"

#include "./tmptileset.hpp"
#include "./repackatlas.hpp"


namespace po = boost::program_options;
namespace bio = boost::iostreams;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;
namespace vs = vtslibs::storage;
namespace vr = vtslibs::registry;
namespace vts = vtslibs::vts;
namespace tools = vtslibs::vts::tools;

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
    bool resume;
    bool keepTmpset;

    Config()
        : textureQuality(85), optimalTextureSize(256, 256)
        , ntLodPixelSize(1.0), dtmExtractionRadius(40.0)
        , forceWatertight(false), clipMargin(1.0 / 128.)
        , borderClipMargin(clipMargin)
        , sigmaEditCoef(1.5), resume(false), keepTmpset(false)
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

    virtual void preNotifyHook(const po::variables_map &vars)
        UTILITY_OVERRIDE;

    virtual bool help(std::ostream &out, const std::string &what) const
        UTILITY_OVERRIDE;

    virtual int run() UTILITY_OVERRIDE;

    int analyze(const po::variables_map &vars);

    fs::path output_;
    std::vector<fs::path> input_;

    vts::CreateMode createMode_;

    Config config_;
};

void Vef2Vts::configuration(po::options_description &cmdline
                             , po::options_description &config
                             , po::positional_options_description &pd)
{
    vr::registryConfiguration(cmdline, vr::defaultPath());
    vr::creditsConfiguration(cmdline);
    service::verbosityConfiguration(cmdline);

    cmdline.add_options()
        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")
        ("input", po::value(&input_)->required()
         , "Path to input VEF archive.")
        ("overwrite", "Existing tile set gets overwritten if set.")

        ("tilesetId", po::value(&config_.tilesetId)->required()
         , "Output tileset ID.")

        ("referenceFrame", po::value(&config_.referenceFrame)->required()
         , "Destination reference frame. Must be different from input "
         "tileset's referenceFrame.")

        ("textureQuality", po::value(&config_.textureQuality)
         ->default_value(config_.textureQuality)->required()
         , "Texture quality for JPEG texture encoding (0-100).")

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

        ("resume"
         , "Resumes interrupted encoding. There must be complete (valid) "
         "temporary tileset inside generated tileset. Use with caution.")
        ("keepTmpset"
         , "Keep temporary tileset intact on exit.")

        ("analyzeOnly"
         , "If set, do not process the file, only analyze it.")
        ;

    pd
        .add("output", 1)
        .add("input", -1);

    (void) config;
}

void Vef2Vts::preNotifyHook(const po::variables_map &vars)
{
    if (vars.count("analyzeOnly")) {
        service::immediateExit(analyze(vars));
    }
}

void Vef2Vts::configure(const po::variables_map &vars)
{
    vr::registryConfigure(vars);
    config_.credits = vr::creditsConfigure(vars);

    createMode_ = (vars.count("overwrite")
                   ? vts::CreateMode::overwrite
                   : vts::CreateMode::failIfExists);

    if ((config_.textureQuality < 0) || (config_.textureQuality > 100)) {
        throw po::validation_error
            (po::validation_error::invalid_option_value, "textureQuality");
    }

    if (vars.count("tileExtents")) {
        config_.tileExtents = vars["tileExtents"].as<vts::LodTileRange>();
    }

    config_.resume = vars.count("resume");
    config_.keepTmpset = vars.count("keepTmpset");
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

bool loadGzippedObj(ObjLoader &loader, const roarchive::RoArchive &archive
                    , const fs::path &path)
{
    auto f(archive.istream(path));
    bio::filtering_istream gzipped;
    gzipped.push
        (bio::gzip_decompressor(bio::gzip_params().window_bits, 1 << 16));
    gzipped.push(f->get());

    auto res(loader.parse(gzipped));
    f->close();
    return res;
}

bool loadObj(ObjLoader &loader, const roarchive::RoArchive &archive
             , const vef::Window &window)
{
    switch (window.mesh.format) {
    case vef::Mesh::Format::obj:
        return loader.parse(*archive.istream(window.mesh.path));

    case vef::Mesh::Format::gzippedObj:
        return loadGzippedObj(loader, archive, window.mesh.path);
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

    Assignment() = default;

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
    typedef std::vector<map> maplist;
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

            if (diff <= diffLimit) {
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
    // FIXME: probably needs to be fixed
    while ((ntLod > lr.min) && (pixelSize < config.ntLodPixelSize)) {
        pixelSize *= 2.0;
        --ntLod;
    }

    return NavtileInfo(vts::LodRange(minLod, ntLod), pixelSize);
}

class Analyzer {
public:
    Analyzer(const std::vector<vef::Archive> &input
             , const vr::ReferenceFrame &rf
             , const Config &config
             , vts::NtGenerator &ntg)
        : rf_(rf), config_(config)
        , nodes_(vts::NodeInfo::nodes(rf_))
    {
        Assignment::maplist assignments;
        // process all input manifests
        for (const auto &archive : input) {
            const auto &manifest(archive.manifest());

            std::size_t manifestWindowsSize(manifest.windows.size());
            auto assignmentsStart(assignments.size());
            assignments.resize(assignmentsStart + manifestWindowsSize);

            UTILITY_OMP(parallel for)
                for (std::size_t i = 0; i < manifestWindowsSize; ++i) {
                    // calculate assignment
                    const auto &loddedWindow(manifest.windows[i]);
                    const auto assignment
                        (assign(*manifest.srs, archive
                                , loddedWindow.lods.front()
                                , loddedWindow.lods.size() - 1));
                    // store
                    assignments[assignmentsStart + i] = assignment;
                }
        }

        analyze(assignments);

        // split assignments per input archive
        auto iassignments(assignments.begin());
        for (const auto &archive : input) {
            auto size(archive.manifest().windows.size());
            assignments_.emplace_back(iassignments, iassignments + size);
            iassignments += size;
        }

        for (const auto &item : ntMap_) {
            ntg.addAccumulator(item.first->srs, item.second.lodRange
                               , item.second.pixelSize);
        }
    }

    const std::vector<Assignment::maplist>& assignments() const {
        return assignments_;
    }

private:
    void analyze(Assignment::maplist &assignments);
    Assignment::map assign(const geo::SrsDefinition &inputSrs
                           , const vef::Archive &archive
                           , const vef::Window &window, std::size_t lodCount);

    const vr::ReferenceFrame &rf_;
    const Config &config_;

    const vts::NodeInfo::list nodes_;
    NavtileInfo::map ntMap_;
    std::vector<Assignment::maplist> assignments_;
};

void Analyzer::analyze(Assignment::maplist &assignments)
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

Assignment::map Analyzer::assign(const geo::SrsDefinition &inputSrs
                                 , const vef::Archive &archive
                                 , const vef::Window &window
                                 , std::size_t lodCount)
{
    // load mesh
    ObjLoader loader;

    LOG(info3) << "Loading window mesh from: " << window.mesh.path;
    if (!loadObj(loader, archive.archive(), window)) {
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
        const vts::CsConvertor conv(inputSrs, node.srs());

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

class Cutter {
public:
    Cutter(tools::TmpTileset &tmpset, const vef::Archive &archive
           , const vr::ReferenceFrame &rf, const Config &config
           , const Assignment::maplist &assignments)
        : tmpset_(tmpset), archive_(archive)
        , manifest_(archive_.manifest()), rf_(rf)
        , inputSrs_(*manifest_.srs), config_(config)
        , nodes_(vts::NodeInfo::nodes(rf_))
    {
        cut(assignments);
    }

private:
    void cut(const Assignment::maplist &assignments);

    void windowCut(const vef::Window &window, vts::Lod lodDiff
                   , const Assignment::map &assignemnts);

    void splitToTiles(const vts::NodeInfo &root
                      , vts::Lod lod, const vts::TileRange &tr
                      , const vts::Mesh &mesh
                      , const vts::opencv::Atlas &atlas);
    void cutTile(const vts::NodeInfo &node, const vts::Mesh &mesh
                 , const vts::opencv::Atlas &atlas);

    cv::Mat loadTexture(const fs::path &path) const;

    tools::TmpTileset &tmpset_;
    const vef::Archive &archive_;
    const vef::Manifest &manifest_;
    const vr::ReferenceFrame &rf_;
    const geo::SrsDefinition &inputSrs_;
    const Config &config_;
    const vts::NodeInfo::list nodes_;

    NavtileInfo::map ntMap_;
};

cv::Mat Cutter::loadTexture(const fs::path &path) const
{
    const auto &archive(archive_.archive());
    if (archive.directio()) {
        // optimized access
        auto tex(cv::imread(archive.path(path).string()));
        if (!tex.data) {
            LOGTHROW(err2, std::runtime_error)
                << "Unable to load texture from " << path << ".";
        }
    }

    auto is(archive.istream(path));
    auto tex(cv::imdecode(is->read(), CV_LOAD_IMAGE_COLOR));

    if (!tex.data) {
        LOGTHROW(err2, std::runtime_error)
            << "Unable to load texture from " << is->path() << ".";
    }

    return tex;
}

void Cutter::cut(const Assignment::maplist &assignments)
{
    std::size_t manifestWindowsSize(manifest_.windows.size());
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


void Cutter::windowCut(const vef::Window &window, vts::Lod lodDiff
                       , const Assignment::map &assignemnts)
{
    // load mesh
    ObjLoader loader;
    LOG(info3) << "Loading window mesh from: " << window.mesh.path;
    if (!loadObj(loader, archive_.archive(), window)) {
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
        inAtlas.add(loadTexture(texture.path));
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

void cutTiles(const std::vector<vef::Archive> &input
              , tools::TmpTileset &tmpset
              , const vr::ReferenceFrame &rf
              , const Config config
              , vts::NtGenerator &ntg)
{
    // analyze whole input
    Analyzer analyzer(input, rf, config, ntg);
    const auto &assignments(analyzer.assignments());

    // cut per archive
    auto iassignments(assignments.begin());
    for (const auto &archive : input) {
        Cutter(tmpset, archive, rf, config
               , *iassignments++);
    }

    tmpset.flush();
    ntg.save(tmpset.root() / "navtile.info");
}

class Encoder : public vts::Encoder {
public:
    /** Regular version.
     */
    Encoder(const boost::filesystem::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const std::vector<vef::Archive> &input
            , const Config &config)
        : vts::Encoder(path, properties, mode)
        , config_(config)
        , tmpset_(path / "tmp")
        , ntg_(&referenceFrame())
    {
        tmpset_.keep(config.keepTmpset);

        // cut tiles to temporary
        cutTiles(input, tmpset_, referenceFrame(), config_, ntg_);

        prepare();
    }

    /** Resume version.
     */
    Encoder(const boost::filesystem::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const Config &config)
        : vts::Encoder(path, properties, mode)
        , config_(config)
        , tmpset_(path / "tmp", false)
        , ntg_(&referenceFrame(), tmpset_.root() / "navtile.info")
    {
        tmpset_.keep(config.keepTmpset);
        prepare();
    }

private:
    void prepare() {
        validTree_ = index_ = tmpset_.tileIndex();

        // make valid tree complete from root
        validTree_.makeAbsolute().complete();

        setConstraints(Constraints().setValidTree(&validTree_));
        setEstimatedTileCount(index_.count());
    }

    virtual TileResult
    generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
             , const TileResult&)
        UTILITY_OVERRIDE;

    virtual void finish(vts::TileSet &ts);

    const Config config_;

    tools::TmpTileset tmpset_;
    vts::TileIndex index_;
    vts::TileIndex validTree_;

    vts::NtGenerator ntg_;
};

inline void warpInPlace(const vts::CsConvertor &conv, vts::SubMesh &sm)
{
    for (auto &v : sm.vertices) { v = conv(v); }
}

inline void warpInPlace(const vts::CsConvertor &conv, vts::Mesh &mesh)
{
    for (auto &sm : mesh) { warpInPlace(conv, sm); }
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

        // mesh in SDS -> pre-compute geom extents
        tile.geomExtents = geomExtents(*tile.mesh);
    }

    // generate external texture coordinates
    vts::generateEtc(*tile.mesh, nodeInfo.extents()
                     , nodeInfo.node().externalTexture);

    if (!config_.forceWatertight) {
        // generate mesh mask if not asked to make all tiles watertight
        vts::generateCoverage(*tile.mesh, nodeInfo.extents());
    }

    // add tile to navtile generator
    ntg_.addTile(tileId, nodeInfo, *tile.mesh);

    // warp mesh to physical SRS
    warpInPlace(sds2DstPhy, *tile.mesh);

    // set credits
    tile.credits = config_.credits;

    // done
    return result;
}

void Encoder::finish(vts::TileSet &ts)
{
    ntg_.generate(ts, config_.dtmExtractionRadius);
}

int Vef2Vts::run()
{
    vts::TileSetProperties properties;
    properties.referenceFrame = config_.referenceFrame;
    properties.id = config_.tilesetId;

    if (config_.resume) {
        // run the encoder
        Encoder(output_, properties, createMode_, config_).run();

        // all done
        LOG(info4) << "All done.";
        return EXIT_SUCCESS;
    }

    // load input manifests
    std::vector<vef::Archive> input;
    for (const auto &path : input_) {
        input.emplace_back(path);
        if (!input.back().manifest().srs) {
            LOG(fatal)
                << "VEF archive " << path
                << " doesn't have assigned an SRS, cannot proceed.";
            return EXIT_FAILURE;
        }
    }

    // run the encoder
    Encoder(output_, properties, createMode_, input, config_).run();

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}

class ExtentsFinder : public geometry::ObjParserBase {
public:
    ExtentsFinder(math::Extents2 &extents
                  , const geo::SrsDefinition &srs
                  , const geo::SrsDefinition &geogcs)
        : extents_(extents), conv_(srs, geogcs)
    {}

private:
    virtual void addVertex(const Vector3d &v) {
        math::update(extents_, conv_(math::Point3(v.x, v.y, v.z)));
    }

    virtual void addTexture(const Vector3d&) {}

    virtual void addFacet(const Facet &) {}

    virtual void useMaterial(const std::string&) {}

    virtual void addNormal(const Vector3d&) { /*ignored*/ }
    virtual void materialLibrary(const std::string&) { /*ignored*/ }

    math::Extents2& extents_;
    geo::CsConvertor conv_;
};


// measures gzipped obj extents
bool measureExtents(ExtentsFinder &loader, const roarchive::RoArchive &archive
                    , const fs::path &path)
{
    auto f(archive.istream(path));
    bio::filtering_istream gzipped;
    gzipped.push
        (bio::gzip_decompressor(bio::gzip_params().window_bits, 1 << 16));
    gzipped.push(f->get());

    auto res(loader.parse(gzipped));
    f->close();
    return res;
}

// measures standard obj extents or uses the other function overload
bool measureExtents(ExtentsFinder &loader, const roarchive::RoArchive &archive
                    , const vef::Window &window)
{
    switch (window.mesh.format) {
    case vef::Mesh::Format::obj:
        return loader.parse(*archive.istream(window.mesh.path));

    case vef::Mesh::Format::gzippedObj:
        return measureExtents(loader, archive, window.mesh.path);
    }
    throw;
}

void updateExtents(const vef::Archive &archive
                   , const geo::SrsDefinition &geogcs
                   , math::Extents2 &extents)
{
    for (const auto &loddedWindow : archive.manifest().windows) {
        if (loddedWindow.lods.empty()) { continue; }

        ExtentsFinder ef(extents, *archive.manifest().srs, geogcs);
        measureExtents(ef, archive.archive(), loddedWindow.lods.back());
    }
}

class QuantileFinder : public geometry::ObjParserBase {
public:
    QuantileFinder(const geo::SrsDefinition &srs
            , const geo::SrsDefinition &geogcs)
            : conv_(srs, geogcs)
    {}

    // returns qunatile from vertices added so far
    void findQuantilePt(float quantile, math::Point3& quantilePt) {
        // return resulting quantile point from pts sorted by Z coord
        std::cout << points.size() << "\n";
        std::nth_element(points.begin()
                , points.begin() + points.size() * quantile
                , points.end(),
                         [](const math::Point3f& lhs, const math::Point3f& rhs) {
                             return lhs(2) < rhs(2);
                         }
        );
        quantilePt = conv_(points[points.size() * quantile]);
    }

private:
    virtual void addVertex(const Vector3d &v) {
        points.push_back(v);
    }

    virtual void addTexture(const Vector3d&) {}

    virtual void addFacet(const Facet &) {}

    virtual void useMaterial(const std::string&) {}

    virtual void addNormal(const Vector3d&) { /*ignored*/ }
    virtual void materialLibrary(const std::string&) { /*ignored*/ }

    math::Points3 points;
    geo::CsConvertor conv_;
};

// measures gzipped obj extents
bool addVerticesToFinder(QuantileFinder &finder, const roarchive::RoArchive &archive
        , const fs::path &path)
{
    auto f(archive.istream(path));
    bio::filtering_istream gzipped;
    gzipped.push
            (bio::gzip_decompressor(bio::gzip_params().window_bits, 1 << 16));
    gzipped.push(f->get());

    auto res(finder.parse(gzipped));
    f->close();
    return res;
}

// measures standard obj extents or uses the other function overload
bool addVerticesToFinder(QuantileFinder &finder, const roarchive::RoArchive &archive
        , const vef::Window &window)
{
    switch (window.mesh.format) {
        case vef::Mesh::Format::obj:
            return finder.parse(*archive.istream(window.mesh.path));

        case vef::Mesh::Format::gzippedObj:
            return addVerticesToFinder(finder, archive, window.mesh.path);
    }
    throw;
}

void measureQuantilePt(const vef::Archive &archive
                      , const geo::SrsDefinition &geogcs
                      , math::Point3 &quantilePt)
{
    QuantileFinder qf(*archive.manifest().srs, geogcs);
    for (const auto &loddedWindow : archive.manifest().windows) {
        if (loddedWindow.lods.empty()) { continue; }

        addVerticesToFinder(qf, archive.archive(), loddedWindow.lods.back());
        //addVerticesToFinder(qf, archive.archive(), *(loddedWindow.lods.end() - 2));
    }
    qf.findQuantilePt(0.5f, quantilePt);
}


bool analyzeInput(const fs::path &input
                  , const boost::optional<geo::SrsDefinition> &geogcs
                  , math::Extents2 &extents
                  , math::Point3 & quantilePt
                  , const service::Verbosity& verbose)
{
    vef::Archive archive(input);
    const auto &manifest(archive.manifest());
    if (!manifest.srs) {
        return false;
    }

    if (geogcs) {
        updateExtents(input, *geogcs, extents);

        if (verbose.level >= 2) {
            measureQuantilePt(input, *geogcs, quantilePt);
        }
    }

    return true;
}

int Vef2Vts::analyze(const po::variables_map &vars)
{
    // process configuration
    vr::registryConfigure(vars);
    auto verbose(service::verbosityConfigure(vars));

    if (!vars.count("input")) {
        throw po::required_option("input");
    }
    const auto input(vars["input"].as<std::vector<fs::path>>());

    boost::optional<geo::SrsDefinition> geogcs;
    if (verbose) {
        if (!vars.count("referenceFrame")) {
            throw po::required_option("referenceFrame");
        }
        const auto referenceFrameId(vars["referenceFrame"].as<std::string>());

        // get reference frame
        const auto &referenceFrame
            (vr::system.referenceFrames(referenceFrameId));

        // get geographic system from physical SRS
        geogcs = vr::system.srs(referenceFrame.model.navigationSrs)
                .srsDef.geographic();
        geogcs = geo::merge(*geogcs, vr::system.srs(referenceFrame.model.publicSrs)
                .srsDef);
    }

    math::Extents2 extents(math::InvalidExtents{});
    math::Point3 quantilePt;

    int referenced(0);
    int unreferenced(0);
    for (const auto file : input) {
        if (analyzeInput(file, geogcs, extents, quantilePt, verbose )) {
            ++referenced;
        } else {
            ++unreferenced;
        }
    }

    if (referenced) {
        if (unreferenced) {
            std::cout << "georeferenced: partial" << std::endl;
        } else {
            std::cout << "georeferenced: true" << std::endl;
            if (verbose) {
                auto center(math::center(extents));
                std::cout << "geogcs: " << *geogcs << std::endl;
                std::cout << "center: " << std::fixed << std::setprecision(9)
                          << center(0) << " " << center(1) << std::endl;
                if (verbose.level >= 2) {

                    std::cout << "elevation_pt: " << std::fixed << std::setprecision(9)
                              << quantilePt(0) << " "
                              << quantilePt(1) << " "
                              << quantilePt(2) << std::endl;
                }
            }
        }
    } else {
        std::cout << "georeferenced: false" << std::endl;
    }

    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[])
{
    utility::unlimitedCoredump();
    return Vef2Vts()(argc, argv);
}
