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

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/utility/in_place_factory.hpp>
#include <boost/lexical_cast.hpp>

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
#include "utility/path.hpp"

#include "service/cmdline.hpp"
#include "service/verbosity.hpp"

#include "math/transform.hpp"
#include "math/filters.hpp"
#include "math/math.hpp"

#include "imgproc/scanconversion.hpp"
#include "imgproc/jpeg.hpp"

#include "geo/csconvertor.hpp"
#include "geo/enu.hpp"

#include "vts-libs/tools/progress.hpp"
#include "vts-libs/vts/mesh.hpp"
#include "vts-libs/vts/meshop.hpp"
#include "vts-libs/vts/tileop.hpp"

#include "vef/reader.hpp"
#include "slpk/reader.hpp"
#include "miniball/miniball.hpp"

#include "./tmptileset.hpp"
#include "./repackatlas.hpp"


namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;
namespace vs = vtslibs::storage;
namespace vr = vtslibs::registry;
namespace vt = vtslibs::tools;
namespace vts = vtslibs::vts;
namespace tools = vtslibs::vts::tools;

namespace {

struct Config {
    int textureQuality;
    math::Size2 optimalTextureSize;
    slpk::SpatialReference spatialReference;

    vts::SubmeshMergeOptions smMergeOptions;
    double clipMargin;
    bool resume;
    bool keepTmpset;

    Config()
        : textureQuality(85), optimalTextureSize(256, 256)
        , clipMargin(1.0 / 128.), resume(false), keepTmpset(false)
    {}
};

class Vef2Slpk : public service::Cmdline
{
public:
    Vef2Slpk()
        : service::Cmdline("vef2slpk", BUILD_TARGET_VERSION)
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

    fs::path input_;
    fs::path output_;

    Config config_;
    vt::ExternalProgress::Config epConfig_;
};

void Vef2Slpk::configuration(po::options_description &cmdline
                             , po::options_description &config
                             , po::positional_options_description &pd)
{
    cmdline.add_options()
        ("input", po::value(&input_)->required()
         , "Path to input VEF archive.")
        ("output", po::value(&output_)->required()
         , "Path to output SLPK file.")
        ("overwrite", "Existing tile set gets overwritten if set.")

        ("textureQuality", po::value(&config_.textureQuality)
         ->default_value(config_.textureQuality)->required()
         , "Texture quality for JPEG texture encoding (0-100).")

        // TODO: spatial reference

        ("clipMargin", po::value(&config_.clipMargin)
         ->default_value(config_.clipMargin)
         , "Margin (in fraction of tile dimensions) added to tile extents in "
         "all 4 directions.")

        ("optimalTextureSize", po::value(&config_.optimalTextureSize)
         ->default_value(config_.optimalTextureSize)->required()
         , "Size of ideal tile texture. Used to calculate fitting LOD from"
         "mesh texel size. Do not modify.")

        ("resume"
         , "Resumes interrupted encoding. There must be complete (valid) "
         "temporary tileset inside generated tileset. Use with caution.")
        ("keepTmpset"
         , "Keep temporary tileset intact on exit.")
        ;

    vt::progressConfiguration(config);

    pd
        .add("input", 1)
        .add("output", 1)
        ;

    (void) config;
}

void Vef2Slpk::configure(const po::variables_map &vars)
{
    config_.resume = vars.count("resume");
    config_.keepTmpset = vars.count("keepTmpset");
    epConfig_ = vt::configureProgress(vars);
}

bool Vef2Slpk::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(vef2vts
usage
    vef2vts OUTPUT INPUT+ [OPTIONS]

)RAW";
    }
    return false;
}

struct MeshInfo {
    vts::MeshArea area;
    math::Extents2 extents;
    std::size_t faceCount;

    MeshInfo()
       : extents(math::InvalidExtents{}), faceCount()
    {}

    void update(const MeshInfo &mi) {
        area.mesh += mi.area.mesh;
        area.submeshes.insert(area.submeshes.end(), mi.area.submeshes.begin()
                              , mi.area.submeshes.end());
        math::update(extents, mi.extents.ll);
        math::update(extents, mi.extents.ur);
        faceCount += mi.faceCount;
    }
};

struct Setup {
    math::Extents2 workExtents;
    geo::SrsDefinition srcSrs;
    geo::SrsDefinition workSrs;
    geo::SrsDefinition dstSrs;
    geo::CsConvertor src2work;
    geo::CsConvertor work2dst;
    vts::MeshArea meshArea;
    std::size_t faceCount;
    std::size_t depth;
    vts::Lod maxLod;

    Setup() : faceCount(), depth(), maxLod() {}
};

/** Makes extents prettier:
 *
 *   * center placed at integral coordinates
 *   * integral dimensions
 *   * square footprint
 */
math::Extents2 prettifyExtents(const math::Extents2 &extents)
{
    // measure extents
    auto size(math::size(extents));
    auto center(math::center(extents));
    center(0) = std::round(center(0));
    center(1) = std::round(center(0));

    auto dim(std::max
             (std::ceil(size.width / 2.0) + 1.0
              , std::ceil(size.height / 2.0) + 1.0));

    return { center(0) - dim, center(1) - dim
            , center(0) + dim, center(1) + dim };
}

/** Measures whole mesh extents from coarsest data.
 */
math::Extents3 meshExtents(const vef::Archive &archive)
{
    struct ExtentsMeasurer : public geometry::ObjParserBase {
        ExtentsMeasurer() : extents(math::InvalidExtents{}) {}

        virtual void addVertex(const Vector3d &v) {
            math::update(extents, math::Point3d(v));
        }

        virtual void addTexture(const Vector3d&) {}
        virtual void addNormal(const Vector3d&) {}
        virtual void addFacet(const Facet&) {}
        virtual void materialLibrary(const std::string&) {}
        virtual void useMaterial(const std::string&) {}

        math::Extents3 extents;
    };

    ExtentsMeasurer em;

    for (const auto &lw : archive.manifest().windows) {
        const auto &window(lw.lods.back());
        if (!em.parse(*archive.meshIStream(window.mesh))) {
            LOGTHROW(err2, std::runtime_error)
                << "Unable to load mesh from OBJ file at "
                << window.mesh.path << ".";
        }
    }

    return em.extents;
}

MeshInfo measureMesh(const vef::Archive &archive, const vef::Mesh &mesh
                     , const geo::CsConvertor &conv)
{
    LOG(info2) << "Loading mesh from " << mesh.path << ".";

    MeshInfo mi;

    auto m(vts::loadMeshFromObj(*archive.meshIStream(mesh), mesh.path));
    for (auto &sm : m.submeshes) {
        for (auto &v : sm.vertices) {
            v = conv(v);
            math::update(mi.extents, v(0), v(1));
        }
        mi.faceCount += sm.faces.size();
    }

    mi.area = vts::area(m);

    return mi;
}

MeshInfo measureMeshes(const vef::Archive &archive
                       , const geo::CsConvertor &conv)
{
    MeshInfo mi;

    const auto &windows(archive.manifest().windows);

    UTILITY_OMP(parallel for)
    for (std::size_t i = 0; i < windows.size(); ++i) {
        const auto &window(windows[i].lods.front());

        auto a(measureMesh(archive, window.mesh, conv));

        // expand texture area
        auto iatlas(window.atlas.begin());
        for (auto &sm : a.area.submeshes) {
            // expand by texture size
            const auto &size((*iatlas++).size);
            sm.internalTexture *= size.width;
            sm.internalTexture *= size.height;
        }

        UTILITY_OMP(critical(vef2slpk_measureMeshes_1))
            mi.update(a);
    }
    return mi;
}

Setup toEnu(const Config &config, const vef::Archive &archive)
{
    Setup setup;
    setup.srcSrs = *archive.manifest().srs;
    setup.dstSrs = config.spatialReference.srs();

    if (setup.srcSrs.is(geo::SrsDefinition::Type::enu)) {
        // find, it's ENU -> measure mesh in src/work SRS
        auto mi(measureMeshes(archive, geo::CsConvertor()));

        setup.workExtents = mi.extents;
        setup.workSrs = setup.srcSrs;
        setup.work2dst = geo::CsConvertor(setup.workSrs, setup.dstSrs);
        setup.meshArea = mi.area;
        setup.faceCount = mi.faceCount;

        return setup;
    }

    // not ENU, build one

    // get center of mesh in its source SRS
    const auto center(math::center(meshExtents(archive)));

    // build ENU
    // TODO: extract spheroid and towgs84 from setup.srsSrs
    geo::Enu enu(geo::CsConvertor(setup.srcSrs, setup.srcSrs.geographic())
                 (center));

    setup.workSrs = geo::SrsDefinition::fromEnu(enu);
    setup.src2work = geo::CsConvertor(setup.srcSrs, enu);
    setup.work2dst = geo::CsConvertor(enu, setup.dstSrs);

    // measure mesh in work SRS
    auto mi(measureMeshes(archive, setup.src2work));

    setup.workExtents = mi.extents;
    setup.meshArea = mi.area;
    setup.faceCount = mi.faceCount;

    return setup;
}

double pixelArea(const vts::MeshArea &area) {
    double ta(0);
    for (const auto &sm : area.submeshes) {
        ta += sm.internalTexture;
    }
    return area.mesh / ta;
}

Setup makeSetup(const Config &config, const vef::Archive &archive)
{
    auto setup(toEnu(config, archive));
    LOG(info4) << "setup.workExtents: " << setup.workExtents;
    setup.workExtents = prettifyExtents(setup.workExtents);
    LOG(info4) << "setup.workExtents: " << setup.workExtents;

    for (const auto &lw : archive.manifest().windows) {
        setup.depth = std::max(setup.depth, lw.lods.size());
    }

    // compute area of one pixel (meter^2/pixel)
    auto px(pixelArea(setup.meshArea));

    // optimal tile size in meters^2
    auto tileArea(px
                  * config.optimalTextureSize.width
                  * config.optimalTextureSize.height);

    // we made work extents square -> we can use scene area as is
    const auto sceneSize(math::size(setup.workExtents));
    const auto paneArea(math::area(sceneSize));

    const auto tileCount(paneArea / tileArea);

    const auto optimalLod(0.5 * std::log2(tileCount));

    setup.maxLod = std::round(optimalLod);

    if (setup.depth > (setup.maxLod + 1)) {
        setup.maxLod = setup.depth - 1;
    }

    return setup;
}

/** Per-window record to ease parallel processing.
 */
struct WindowRecord {
    const vef::Window *window;
    vts::Lod lod;

    WindowRecord(const vef::Window &window, vts::Lod lod)
        : window(&window), lod(lod)
    {}

    typedef std::vector<WindowRecord> list;
};

WindowRecord::list windowRecordList(const vef::Archive &archive
                                    , vts::Lod maxLod)
{
    WindowRecord::list list;

    for (const auto &lw : archive.manifest().windows) {
        auto lod(maxLod);
        for (const auto &w : lw.lods) {
            list.emplace_back(w, lod--);
        }
    }

    return list;
}

inline void warpInPlace(vts::SubMesh &mesh, const geo::CsConvertor &conv)
{
    for (auto &v : mesh.vertices) { v = conv(v); }
}

inline void warpInPlace(vts::Mesh &mesh, const geo::CsConvertor &conv)
{
    for (auto &sm : mesh) { warpInPlace(sm, conv); }
}

class Cutter {
public:
    Cutter(tools::TmpTileset &ts, const vef::Archive &archive
           , const Config &config, const Setup &setup)
        : ts_(ts), archive_(archive)
        , config_(config), setup_(setup)
        , windows_(windowRecordList(archive_, setup.maxLod))
    {}

    void operator()(/**vt::ExternalProgress &progress*/);

private:
    void windowCut(const WindowRecord &window);

    void splitToTiles(vts::Lod lod, const vts::TileRange &tr
                      , const vts::Mesh &mesh
                      , const vts::opencv::Atlas &atlas);

    void tileCut(const vts::TileId &tileId, const vts::Mesh &mesh
                 , const vts::opencv::Atlas &atlas);

    cv::Mat loadTexture(const fs::path &path) const;

    tools::TmpTileset &ts_;
    const vef::Archive &archive_;
    const Config &config_;
    const Setup &setup_;

    WindowRecord::list windows_;
};

void Cutter::operator()(/**vt::ExternalProgress &progress*/)
{
    UTILITY_OMP(parallel for)
        for (std::size_t i = 0; i < windows_.size(); ++i) {
            windowCut(windows_[i]);
        }

    ts_.flush();
}

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

vts::TileRange computeTileRange(const Setup &setup, vts::Lod lod
                                , const math::Extents2 &meshExtents)
{
    vts::TileRange r(math::InvalidExtents{});
    const auto ts(vts::tileSize(setup.workExtents, lod));
    const auto origin(math::ul(setup.workExtents));

    for (const auto &p : vertices(meshExtents)) {
        update(r, vts::TileRange::point_type
               ((p(0) - origin(0)) / ts.width
                , (origin(1) - p(1)) / ts.height));
    }

    return r;
}

void Cutter::windowCut(const WindowRecord &wr)
{
    const auto &window(*wr.window);
    const auto &wm(window.mesh);
    LOG(info2) << "Cutting window mesh from " << wm.path << ".";
    auto mesh(vts::loadMeshFromObj(*archive_.meshIStream(wm), wm.path));
    warpInPlace(mesh, setup_.src2work);

    if (mesh.submeshes.size() != window.atlas.size()) {
        LOGTHROW(err2, std::runtime_error)
            << "Texture/submesh count mismatch in window "
            << window.path << ".";
    }

    vts::opencv::Atlas atlas;
    for (const auto &texture : window.atlas) {
        LOG(info3) << "Loading window texture from: " << texture.path;
        atlas.add(loadTexture(texture.path));
    }

    auto tr(computeTileRange(setup_, wr.lod, computeExtents(mesh)));
    LOG(info3) << "Splitting window " << window.path
               << " to tiles in " << wr.lod << "/" << tr << ".";
    splitToTiles(wr.lod, tr, mesh, atlas);
}

void Cutter::splitToTiles(vts::Lod lod, const vts::TileRange &tr
                          , const vts::Mesh &mesh
                          , const vts::opencv::Atlas &atlas)
{
    for (auto j(tr.ll(1)), je(tr.ur(1)); j <= je; ++j) {
        for (auto i(tr.ll(0)), ie(tr.ur(0)); i <= ie; ++i) {
            tileCut(vts::TileId(lod, i, j), mesh, atlas);
        }
    }
}

math::Extents2 tileExtents(const math::Extents2 &rootExtents
                           , const vts::TileId &tileId)
{
    auto tc(vts::tileCount(tileId.lod));
    auto rs(size(rootExtents));
    math::Size2f ts(rs.width / tc, rs.height / tc);

    return math::Extents2
        (rootExtents.ll(0) + tileId.x * ts.width
         , rootExtents.ur(1) - (tileId.y + 1) * ts.height
         , rootExtents.ll(0) + (tileId.x + 1) * ts.width
         , rootExtents.ur(1) - tileId.y * ts.height);
}

void Cutter::tileCut(const vts::TileId &tileId, const vts::Mesh &mesh
                     , const vts::opencv::Atlas &atlas)
{
    auto extents
        (vts::inflateTileExtents
         (tileExtents(setup_.workExtents, tileId), config_.clipMargin));

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
    tools::repack(tileId, clipped, clippedAtlas);
    ts_.store(tileId, clipped, clippedAtlas);
}

struct NodeId {
    std::vector<int> path;

    NodeId() {}

    NodeId child(int which) const {
        NodeId nodeId(*this);
        nodeId.path.push_back(which);
        return nodeId;
    }
};

template<typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits> &os, const NodeId &nodeId)
{
    return os << utility::join(nodeId.path, "-", "root");
}

class Generator {
public:
    Generator(const tools::TmpTileset &ts, const Config &config
              , const Setup &setup)
        : ts_(ts), config_(config), setup_(setup)
        , ti_(ts.tileIndex()), fullTree_(ti_)
    {
        fullTree_.complete();
    }

    void operator()(/**vt::ExternalProgress &progress*/);

private:
    boost::optional<slpk::NodeReference>
    process(const vts::TileId &tileId, NodeId nodeId);

    const tools::TmpTileset &ts_;
    const Config &config_;
    const Setup &setup_;

    vts::TileIndex ti_;
    vts::TileIndex fullTree_;
};

struct MeshVertices {
    typedef double value_type;
    typedef miniball::Point3_<value_type> Point3;

    MeshVertices(const vts::Mesh &mesh)
        : mesh(mesh)
    {
        for (const auto &sm : mesh) {
            for (const auto &v : sm.vertices) {
                points.push_back(v);
            }
        }
    }

    std::size_t size() const { return points.size(); }
    Point3 operator[](std::size_t i) const { return points[i]; }

    const vts::Mesh &mesh;
    std::vector<Point3> points;
};

boost::optional<slpk::NodeReference>
Generator::process(const vts::TileId &tileId, NodeId nodeId)
{
    struct TIDGuard {
        TIDGuard(const std::string &id)
            : old(dbglog::thread_id())
        {
            dbglog::thread_id(id);
        }
        ~TIDGuard() { try { dbglog::thread_id(old); } catch (...) {} }

        const std::string old;
    };

    if (!fullTree_.get(tileId)) { return boost::none; }

    TIDGuard tg(str(boost::format("tile:%s") % tileId));

    boost::optional<slpk::NodeReference> onr;
    if (ti_.get(tileId)) {
        LOG(info3)
            << "Generating node <" << nodeId << "> from tile " << tileId
            << ".";

        vts::Mesh::pointer mesh;
        vts::Atlas::pointer atlas;

        {
            // TODO: generate
            const auto loaded(ts_.load(tileId, config_.textureQuality));

            // merge submeshes
            std::tie(mesh, atlas)
                = vts::mergeSubmeshes
                (tileId, std::get<0>(loaded), std::get<1>(loaded)
                 , config_.textureQuality, config_.smMergeOptions);
        }

        auto &nodeReference(*(onr = boost::in_place()));
        nodeReference.id = boost::lexical_cast<std::string>(nodeId);

        const auto mbs(miniball::minimumBoundingSphere(MeshVertices(*mesh)));
        LOG(info4) << "mbs: " << mbs.center << ", " << mbs.radius;
        nodeReference.mbs.x = mbs.center(0);
        nodeReference.mbs.y = mbs.center(1);
        nodeReference.mbs.z = mbs.center(2);
        nodeReference.mbs.r = mbs.radius;

        // TODO: fill node reference
        (void) nodeReference;
    }

    // proces children -> go down
    slpk::NodeReference::list nodeReferences;
    int childIndex(0);
    for (auto child : vts::children(tileId)) {
        UTILITY_OMP(task)
        {
            if (auto nodeReference
                = process(child, nodeId.child(childIndex)))
            {
                UTILITY_OMP(critical(vef2slpk_process_1))
                   nodeReferences.push_back(std::move(*nodeReference));
            }
        }
        ++childIndex;
    }

    // done
    return onr;
}

void Generator::operator()(/**vt::ExternalProgress &progress*/)
{
    UTILITY_OMP(parallel)
    UTILITY_OMP(single)
    {
        process({}, {});
    }
}

int Vef2Slpk::run()
{
    const auto tmpTilesetPath(utility::addExtension(output_, ".tmpts"));
    tools::TmpTileset ts(tmpTilesetPath, !config_.resume);
    ts.keep(config_.keepTmpset);

    if (config_.resume) {
#if 0
        // TODO: implement me

        // all done
        LOG(info4) << "All done.";
        return EXIT_SUCCESS;
#endif
        LOG(fatal) << "Resume not implemented yet.";
        return EXIT_FAILURE;
    }

    // load input manifests
    vef::Archive input(input_);
    if (!input.manifest().srs) {
        LOG(fatal)
            << "VEF archive " << input_
            << " doesn't have assigned an SRS, cannot proceed.";
        return EXIT_FAILURE;
    }

    // measure mesh extents
    auto setup(makeSetup(config_, input));

    Cutter(ts, input, config_, setup)(/* progress */);
    Generator(ts, config_, setup)(/* progress */);

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[])
{
    utility::unlimitedCoredump();
    return Vef2Slpk()(argc, argv);
}
