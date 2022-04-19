/**
 * Copyright (c) 2017 Melown Technologies SE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * *  Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * *  Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
#include "geometry/polygon.hpp"

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
#include "vts-libs/tools-support/progress.hpp"

#include "vef/reader.hpp"

#include "vts-libs/tools-support/tmptsencoder.hpp"
#include "vts-libs/tools-support/repackatlas.hpp"


namespace po = boost::program_options;
namespace bio = boost::iostreams;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;
namespace vs = vtslibs::storage;
namespace vr = vtslibs::registry;
namespace vts = vtslibs::vts;
namespace vt = vtslibs::tools;
namespace tools = vtslibs::vts::tools;

namespace {

struct Config : tools::TmpTsEncoder::Config {
    std::string tilesetId;
    std::string referenceFrame;
    math::Size2 optimalTextureSize;
    double ntLodPixelSize;

    boost::optional<vts::LodTileRange> tileExtents;
    int lodDepth = 0;
    double clipMargin;
    double borderClipMargin;
    double sigmaEditCoef;
    boost::optional<double> nominalResolution;

    double zShift;

    unsigned int revision = 0;

    bool debug_nothreads;

    Config()
        : optimalTextureSize(256, 256)
        , ntLodPixelSize(1.0)
        , clipMargin(1.0 / 128.)
        , borderClipMargin(clipMargin)
        , sigmaEditCoef(1.5)
        , zShift(0.0)
        , debug_nothreads(false)
    {}

    void configuration(po::options_description &config) {
        tools::TmpTsEncoder::Config::configuration(config);

        config.add_options()
            ("tilesetId", po::value(&tilesetId)->required()
             , "Output tileset ID.")

            ("referenceFrame", po::value(&referenceFrame)->required()
             , "Destination reference frame. Must be different from input "
             "tileset's referenceFrame.")

            ("navtileLodPixelSize"
             , po::value(&ntLodPixelSize)
             ->default_value(ntLodPixelSize)->required()
             , "Navigation data are generated at first LOD "
             "(starting from root) "
             "where pixel size (in navigation grid) is less or "
             "equal to this value.")

            ("clipMargin", po::value(&clipMargin)
             ->default_value(clipMargin)
             , "Margin (in fraction of tile dimensions) added to "
             "tile extents in all 4 directions.")

            ("tileExtents", po::value<vts::LodTileRange>()
             , "Optional tile extents specidied in form lod/llx,lly:urx,ury. "
             "When set, only tiles in that range and below are added to "
             "the output.")

            ("lodDepth", po::value(&lodDepth)->default_value(lodDepth)
             , "Limit output only to given depth from bottom (>0)"
             "or top (<0). 0 means no depth limit.")

            ("borderClipMargin", po::value(&borderClipMargin)
             , "Margin (in fraction of tile dimensions) added to tile extents "
             "where tile touches artificial border definied by tileExtents.")

            ("tweak.optimalTextureSize", po::value(&optimalTextureSize)
             ->default_value(optimalTextureSize)->required()
             , "Size of ideal tile texture. Used to calculate fitting LOD from"
             "mesh texel size. Do not modify.")

            ("tweak.sigmaEditCoef", po::value(&sigmaEditCoef)
             ->default_value(sigmaEditCoef)
             , "Sigma editting coefficient. Meshes with best LOD "
             "difference from mean best LOD lower than "
             "sigmaEditCoef * sigma are assigned "
             "round(mean best LOD).")

            ("tweak.nominalResolution", po::value<double>()
             , "Nominal resolution of input data (in input data SRS units). "
             "Will be used to determine output LOD if set. Enforced to all "
             "input windows. Use wisely.")

            ("zShift", po::value(&zShift)
             ->default_value(zShift)->required()
             , "Manual height adjustment (value is "
             "added to z component of all vertices).")

            ("revision", po::value(&revision)->default_value(revision)
             , "Minimum tileset revision. Actual revision might be greater "
             "if there already was an existing tileset at given output path.")

            ("debug.nothreads", po::value(&debug_nothreads)
             ->default_value(false)->implicit_value(true)
             , "Disable threading for debugging purposes. Applies only to "
             "tileset encoding so far.")
            ;
    }

    void configure(const po::variables_map &vars) {
        tools::TmpTsEncoder::Config::configure(vars);

        if (vars.count("tileExtents")) {
            tileExtents = vars["tileExtents"].as<vts::LodTileRange>();

            LOG(info3) << "Limiting output to tile extents "
                       << tileExtents << ".";
        }

        if (vars.count("tweak.nominalResolution")) {
            nominalResolution = vars["tweak.nominalResolution"].as<double>();
        }

        if (lodDepth > 0) {
            LOG(info3) << "Limiting output to first "
                       << lodDepth << " LODs.";
        } else if (lodDepth < 0) {
            LOG(info3) << "Limiting output to last "
                       << -lodDepth << " LODs.";
        }
    }
};

class Vef2Vts : public service::Cmdline
{
public:
    Vef2Vts()
        : service::Cmdline("vef2vts", BUILD_TARGET_VERSION)
        , createMode_(vts::CreateMode::failIfExists)
    {}

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

    fs::path output_;
    std::vector<fs::path> input_;

    vts::CreateMode createMode_;

    Config config_;
    vt::ExternalProgress::Config epConfig_;
};

void Vef2Vts::configuration(po::options_description &cmdline
                             , po::options_description &config
                             , po::positional_options_description &pd)
{
    config_.configuration(cmdline);

    cmdline.add_options()
        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")
        ("input", po::value(&input_)->required()
         , "Path to input VEF archive(s.")
        ("overwrite", "Existing tile set gets overwritten if set.")
        ;

    vt::progressConfiguration(config);

    pd
        .add("output", 1)
        .add("input", -1);

    (void) config;
}

void Vef2Vts::configure(const po::variables_map &vars)
{
    config_.configure(vars);

    createMode_ = (vars.count("overwrite")
                   ? vts::CreateMode::overwrite
                   : vts::CreateMode::failIfExists);

    epConfig_ = vt::configureProgress(vars);
}

bool Vef2Vts::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(vef2vts
usage
    vef2vts OUTPUT INPUT+ [OPTIONS]

)RAW";
    }
    return false;
}

math::Point3 optionalTransform(const vef::OptionalMatrix &trafo
                                     , const math::Point3 &p)
{
    if (!trafo) { return p; }
    return math::transform(*trafo, p);
}

class ObjLoader : public geometry::ObjParserBase {
public:
    ObjLoader(const vef::OptionalMatrix trafo)
        : textureId_(0), vMap_(), tcMap_(), trafo_(trafo)
    {
        // make sure we have at least one valid material
        useMaterial(0);
    }

    vts::Mesh mesh() const { return mesh_; }

private:
    typedef std::vector<int> VertexMap;
    typedef std::vector<VertexMap> VertexMaps;

    virtual void addVertex(const Vector3d &v) {

        vertices_.emplace_back(
            optionalTransform(trafo_, math::Point3(v.x, v.y, v.z)));
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
    vef::OptionalMatrix trafo_;
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

Assignment::plist analyzeNodeAssignments(const std::string &srs
                                         , Assignment::plist nodeAssignments
                                         , double sigmaEditCoef)
{
    Assignment::plist out;

    std::vector<std::pair<vts::Lod, int>> histogram;

    while (!nodeAssignments.empty()) {
        double meanLod, stddev;
        std::tie(meanLod, stddev) = statistics(nodeAssignments);

        const double diffLimit(sigmaEditCoef * stddev);
        vts::Lod lod(std::round(meanLod));

        std::size_t count(0);

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
                    ++count;
                }
            } else {
                ++inodeAssignments;
            }
        }

        if (count) {  histogram.emplace_back(lod, count); }
    }

    if (!histogram.empty()) {
        LOG(info3) << "Input assignment statistics in <" << srs
                   << ">: LOD: count";
        for (const auto &entry : histogram) {
            LOG(info3) << "    " << entry.first << ": " << entry.second;
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

    // fix min applicable lod to tileExtents lod if set
    if (config.tileExtents) {
        const auto &te(*config.tileExtents);
        if (lr.min < te.lod) {
            lr.min = te.lod;
        }
    }

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
             , vts::NtGenerator &ntg
             , vt::ExternalProgress &progress)
        : rf_(rf), config_(config), progress_(progress)
        , nodes_(vts::NodeInfo::nodes(rf_))
    {
        // calculate number of reported events
        // 1 event per mesh read, 1 event per mesh analyze
        progress.expect([&]() -> std::size_t
        {
            std::size_t events(0);
            for (const auto &archive : input) {
                events += archive.manifest().windows.size();
            }
            return 2 * events;
        }());

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
                                , loddedWindow.lods.size() - 1
                                , vef::windowMatrix(manifest, loddedWindow)));
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
                           , const vef::Window &window, std::size_t lodCount
                           , const vef::OptionalMatrix trafo);

    const vr::ReferenceFrame &rf_;
    const Config &config_;
    vt::ExternalProgress &progress_;

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
            (analyzeNodeAssignments(node.srs(), nodeAssignments
                                    , config_.sigmaEditCoef));

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
                                 , std::size_t lodCount
                                 , const vef::OptionalMatrix trafo)
{
    // load mesh
    ObjLoader loader(trafo);

    LOG(info3) << "Loading window mesh from: " << window.mesh.path;
    if (!loadObj(loader, archive.archive(), window)) {
        LOGTHROW(err2, std::runtime_error)
            << "Unable to load mesh from " << window.mesh.path << ".";
    }

    // mesh loaded
    ++progress_;

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
                } catch (const std::exception&) {
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

        // calculate optimal tile area
        double optimalTileArea(0.0);

        if (config_.nominalResolution) {
            // use provided nominal resolution
            math::Size2f thSize(config_.optimalTextureSize.width
                                       * *config_.nominalResolution / 2.0
                                       , config_.optimalTextureSize.height
                                       * *config_.nominalResolution / 2.0);

            // compute center in destination SRS
            math::Point3d dstCenter;
            {
                math::Extents3 e(math::InvalidExtents{});
                for (const auto &sm : mesh) {
                    update(e, computeExtents(sm.vertices));
                }

                dstCenter = math::center(e);
            }

            const auto srcCenter(conv.inverse()(dstCenter));

            // construct a tile in the source SRS around origin mesh center
            math::Points2 src {
                math::Point2(srcCenter(0) - thSize.width
                             , srcCenter(1) + thSize.height)
                , math::Point2(srcCenter(0) + thSize.width
                               , srcCenter(1) + thSize.height)
                , math::Point2(srcCenter(0) + thSize.width
                               , srcCenter(1) - thSize.height)
                , math::Point2(srcCenter(0) - thSize.width
                               , srcCenter(1) - thSize.height)
            };

            math::Points2 dst;
            for (const auto &v : src) { dst.push_back(conv(v)); }
            // make closed
            dst.push_back(dst.front());

            // and compute dst tile area
            optimalTileArea = abs(geometry::area(dst));
        } else {
            // calculate from mesh data

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
            optimalTileArea = area(config_.optimalTextureSize) * texelArea;
        }

        if (optimalTileArea <= 0.0) { continue; }

        const auto optimalTileCount(node.extents().area()
                                    / optimalTileArea);
        auto bestLod(0.5 * std::log2(optimalTileCount));

        if (bestLod < 0) { continue; }

        // we have best lod for this window in this SDS node, store info
        UTILITY_OMP(critical)
            assignment.insert
            (Assignment::map::value_type
             (node.nodeId()
              , Assignment(node, bestLod, lodCount, computeExtents(mesh))));
    }

    // mesh analyzed
    ++progress_;

    // done
    return assignment;
}

class Cutter {
public:
    Cutter(tools::TmpTileset &tmpset, const vef::Archive &archive
           , const vr::ReferenceFrame &rf, const Config &config
           , vt::ExternalProgress &progress
           , const Assignment::maplist &assignments)
        : tmpset_(tmpset), archive_(archive)
        , manifest_(archive_.manifest()), rf_(rf)
        , inputSrs_(*manifest_.srs), config_(config), progress_(progress)
        , nodes_(vts::NodeInfo::nodes(rf_))
    {
        cut(assignments);
    }

private:
    void cut(const Assignment::maplist &assignments);

    void windowCut(const vef::Window &window, vts::Lod lodDiff
                   , const Assignment::map &assignemnts
                   , const vef::OptionalMatrix trafo);

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
    vt::ExternalProgress &progress_;
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
    auto tex(cv::imdecode(is->read(), cv::IMREAD_COLOR));

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

            LOG(info3) << "Processing window LODs from: " << loddedWindow.path
                       << " (" << loddedWindow.lods.size() << " LODs).";

            // start/end lods, defaults to whole datase
            std::size_t bLod(0);
            std::size_t eLod(loddedWindow.lods.size());

            // apply lod depth
            if (config_.lodDepth > 0) {
                // >0 -> only first lodDepth lods
                eLod = std::min(std::size_t(config_.lodDepth), eLod);
            } else if (config_.lodDepth < 0) {
                // <0 -> only last lodDepth lods
                const std::size_t lodDepth(-config_.lodDepth);
                if (lodDepth < eLod) {
                    bLod = eLod - lodDepth;
                }
            }

            for (std::size_t ii = bLod; ii < eLod; ++ii) {
                windowCut( loddedWindow.lods[ii], ii, assignment
                         , vef::windowMatrix(manifest_, loddedWindow));
                ++progress_;
            }
        }
}


void Cutter::windowCut(const vef::Window &window, vts::Lod lodDiff
                       , const Assignment::map &assignemnts
                       , const vef::OptionalMatrix trafo)
{
    // load mesh
    ObjLoader loader(trafo);
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
                    // apply zShift
                    if (config_.zShift) {
                        projected.back()(2) += config_.zShift;
                    }
                    ++ivalid;
                } catch (const std::exception&) {
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
              , vts::NtGenerator &ntg
              , vt::ExternalProgress &progress)
{
    // analyze whole input
    Analyzer analyzer(input, rf, config, ntg, progress);
    const auto &assignments(analyzer.assignments());

    // cut phase
    progress.expect([&]() -> std::size_t
    {
        std::size_t events(0);
        for (const auto &archive : input) {
            for (const auto &window : archive.manifest().windows) {
                events += window.lods.size();
            }
        }
        return events;
    }());

    // cut per archive
    auto iassignments(assignments.begin());
    for (const auto &archive : input) {
        Cutter(tmpset, archive, rf, config, progress
               , *iassignments++);
    }
}

/**
 * External Progress Phases:
 *
 *  * analyze
 *  * cut
 *  * generate tiles
 *  * generate nt tiles
 */
const vt::ExternalProgress::Weights weightsFull{10, 40, 40, 10};
const vt::ExternalProgress::Weights weightsResume{40, 10};

class Encoder : public tools::TmpTsEncoder {
public:
    /** Regular version.
     */
    Encoder(const boost::filesystem::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const std::vector<vef::Archive> &input
            , const ::Config &config
            , vt::ExternalProgress::Config &&epConfig)
        : tools::TmpTsEncoder(path, properties, mode
                              , config, std::move(epConfig)
                              , (config.resume ? weightsResume : weightsFull)
                              , vts::Encoder::Options()
                              .ensureRevision(config.revision))
        , config_(config)
    {
        if (config.resume) { return; }
        if (input.empty()) {
            LOGTHROW(err1, std::runtime_error)
                << "No archive passed while not resuming.";
        }

        // cut tiles to temporary
        cutTiles(input, tmpset(), referenceFrame(), config_, ntg()
                 , progress());
    }

private:
    const ::Config config_;
};

int Vef2Vts::run()
{
    vts::TileSetProperties properties;
    properties.referenceFrame = config_.referenceFrame;
    properties.id = config_.tilesetId;

    std::vector<vef::Archive> input;

    if (!config_.resume) {
        // load input manifests
        for (const auto &path : input_) {
            input.emplace_back(path);
            if (!input.back().manifest().srs) {
                LOG(fatal)
                    << "VEF archive " << path
                    << " doesn't have assigned an SRS, cannot proceed.";
                return EXIT_FAILURE;
            }
        }
    }

    // run the encoder
    Encoder(output_, properties, createMode_, input, config_
            , std::move(epConfig_)).run(!config_.debug_nothreads);

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[])
{
    utility::unlimitedCoredump();
    return Vef2Vts()(argc, argv);
}
