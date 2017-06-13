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

#include <boost/utility/in_place_factory.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/limits.hpp"
#include "utility/path.hpp"
#include "utility/openmp.hpp"

#include "service/cmdline.hpp"

#include "geometry/meshop.hpp"

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
#include "vts-libs/tools/progress.hpp"

#include "slpk/reader.hpp"

#include "./tmptsencoder.hpp"
#include "./repackatlas.hpp"


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
    double clipMargin;
    double borderClipMargin;
    double sigmaEditCoef;
    double zShift;

    Config()
        : optimalTextureSize(256, 256)
        , ntLodPixelSize(1.0)
        , clipMargin(1.0 / 128.)
        , borderClipMargin(clipMargin)
        , sigmaEditCoef(1.5)
        , zShift(0.0)
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
             , "Navigation data are generated at first LOD (starting "
             "from root) where pixel size (in navigation grid) is less or "
             "equal to this value.")

            ("clipMargin", po::value(&clipMargin)
             ->default_value(clipMargin)
             , "Margin (in fraction of tile dimensions) added to tile extents "
             "in all 4 directions.")

            ("tileExtents", po::value<vts::LodTileRange>()
             , "Optional tile extents specidied in form lod/llx,lly:urx,ury. "
             "When set, only tiles in that range and below are added to "
             "the output.")

            ("borderClipMargin", po::value(&borderClipMargin)
             , "Margin (in fraction of tile dimensions) added to tile extents "
             "where tile touches artificial border definied by tileExtents.")

            ("tweak.optimalTextureSize", po::value(&optimalTextureSize)
             ->default_value(optimalTextureSize)->required()
             , "Size of ideal tile texture. Used to calculate fitting LOD from"
             "mesh texel size. Do not modify.")

            ("tweak.sigmaEditCoef", po::value(&sigmaEditCoef)
             ->default_value(sigmaEditCoef)
             , "Sigma editting coefficient. Meshes with best LOD difference "
             "from mean best LOD lower than sigmaEditCoef * sigma are "
             "assigned round(mean best LOD).")

            ("zShift", po::value(&zShift)
             ->default_value(zShift)->required()
             , "Manual height adjustment (value is "
             "added to z component of all vertices).")
            ;
    }

    void configure(const po::variables_map &vars) {
        tools::TmpTsEncoder::Config::configure(vars);

        if (vars.count("tileExtents")) {
            tileExtents = vars["tileExtents"].as<vts::LodTileRange>();
        }
    }
};

class Slpk2Vts : public service::Cmdline
{
public:
    Slpk2Vts()
        : service::Cmdline("slpk2vts", BUILD_TARGET_VERSION)
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

    fs::path output_;
    fs::path input_;

    vts::CreateMode createMode_;

    Config config_;
    vt::ExternalProgress::Config epConfig_;
};

void Slpk2Vts::configuration(po::options_description &cmdline
                             , po::options_description &config
                             , po::positional_options_description &pd)
{
    vr::registryConfiguration(cmdline, vr::defaultPath());
    vr::creditsConfiguration(cmdline);

    config_.configuration(cmdline);

    cmdline.add_options()
        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")
        ("input", po::value(&input_)->required()
         , "Path to input SLPK archive.")
        ("overwrite", "Existing tile set gets overwritten if set.")
        ;

    vt::progressConfiguration(config);

    pd
        .add("input", 1)
        .add("output", 1);

    (void) config;
}

void Slpk2Vts::configure(const po::variables_map &vars)
{
    vr::registryConfigure(vars);

    config_.configure(vars);

    createMode_ = (vars.count("overwrite")
                   ? vts::CreateMode::overwrite
                   : vts::CreateMode::failIfExists);

    epConfig_ = vt::configureProgress(vars);
}

bool Slpk2Vts::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(slpk2vts
usage
    slpk2vts INPUT OUTPUT [OPTIONS]

)RAW";
    }
    return false;
}

void writeMtl(const fs::path &path, const std::string &name)
{
    LOG(info1) << "Writing " << path;
    std::ofstream f(path.string());

    f << "newmtl 0\n"
      << "map_Kd " << name
      << "\n";
}

void debug(const slpk::Archive &input)
{
    const vts::CsConvertor conv(input.sceneLayerInfo().spatialReference.srs()
                                , "pseudomerc-va");

    const auto tree(input.loadTree());
    for (const auto &n : tree.nodes) {
        const auto &node(n.second);
        LOG(info4) << n.first;
        LOG(info4) << "    geometry:";
        for (const auto &r : node.geometryData) {
            LOG(info4) << "        " << r.href << " " << r.encoding->mime;
        }
        LOG(info4) << "    texture:";
        for (const auto &r : node.textureData) {
            LOG(info4) << "        " << r.href << " " << r.encoding->mime;
        }

        auto geometry(input.loadGeometry(node));

        {
            auto igd(node.geometryData.begin());
            int meshIndex(0);
            for (auto &mesh : geometry) {
                for (auto &v : mesh.vertices) { v = conv(v); }
                const fs::path path((*igd++).href);
                const auto meshPath(utility::addExtension(path, ".obj"));
                create_directories(meshPath.parent_path());

                // TODO get extension from internals
                const auto texPath(utility::addExtension(path, ".jpg"));
                const auto mtlPath(utility::addExtension(path, ".mtl"));

                // save mesh
                {
                    utility::ofstreambuf os(meshPath.string());
                    os.precision(12);
                    saveAsObj(mesh, os, mtlPath.filename().string());
                    os.flush();
                }

                // copy texture as-is
                copy(input.texture(node, meshIndex), texPath);

                writeMtl(mtlPath, texPath.filename().string());

                ++meshIndex;
            }
        }
    }
}

// ------------------------------------------------------------------------

/** Loads SLPK geometry as a list of submeshes.
 */
class VtsMeshLoader
    : public slpk::GeometryLoader
    , public geometry::ObjParserBase
{
public:
    virtual geometry::ObjParserBase& next() {
        mesh_.submeshes.emplace_back();
        current_ = &mesh_.submeshes.back();
        return *this;
    }

    const vts::Mesh& mesh() { return mesh_; }

    virtual void addVertex(const Vector3d &v) {
        current_->vertices.emplace_back(v.x, v.y, v.z);
    }

    virtual void addTexture(const Vector3d &t) {
        current_->tc.emplace_back(t.x, t.y);
    }

    virtual void addFacet(const Facet &f) {
        current_->faces.emplace_back(f.v[0], f.v[1], f.v[2]);
        current_->facesTc.emplace_back(f.t[0], f.t[1], f.t[2]);
    }

private:
    virtual void addNormal(const Vector3d&) {}
    virtual void materialLibrary(const std::string&) {}
    virtual void useMaterial(const std::string&) {}

    vts::Mesh mesh_;
    vts::SubMesh *current_;
};

// ------------------------------------------------------------------------

double bestLod(const Config &config
               , const vts::NodeInfo &rfNode
               , const vts::SubMeshArea &area)
{
    const double texelArea(area.mesh / area.internalTexture);

    const auto optimalTileArea
        (math::area(config.optimalTextureSize) * texelArea);
    const auto optimalTileCount(rfNode.extents().area()
                                / optimalTileArea);
    return (0.5 * std::log2(optimalTileCount));
}

inline void updateExtents(math::Extents2 &extents, const vts::SubMesh &sm)
{
    for (const auto &p : sm.vertices) {
        update(extents, p(0), p(1));
    }
}

inline void updateExtents(math::Extents2 &extents, const vts::Mesh &mesh)
{
    for (const auto &sm : mesh) {
        updateExtents(extents, sm);
    }
}

inline math::Extents2 computeExtents(const vts::Mesh &mesh)
{
    math::Extents2 extents(math::InvalidExtents{});
    updateExtents(extents, mesh);
    return extents;
}

struct MeshInfo {
    vts::SubMeshArea area;
    math::Extents2 extents;

    typedef std::map<const vts::NodeInfo*, MeshInfo> map;

    MeshInfo() : extents(math::InvalidExtents{}) {}

    operator bool() const { return area.internalTexture; }

    void update(const vts::SubMesh &mesh, const math::Size2 &txSize) {
        const auto a(vts::area(mesh));
        area.internalTexture += (a.internalTexture * math::area(txSize));
        area.mesh += a.mesh;
        updateExtents(extents, mesh);
    }

    MeshInfo& operator+=(const MeshInfo &o) {
        if (!o) { return *this; }

        area.mesh += o.area.mesh;
        area.internalTexture += o.area.internalTexture;
        extents = unite(extents, o.extents);
        return *this;
    }
};

MeshInfo measureMesh(const vts::NodeInfo &rfNode
                     , const vts::CsConvertor conv
                     , const vts::Mesh &mesh
                     , const std::vector<math::Size2> &sizes)
{
    MeshInfo mi;

    auto isizes(sizes.begin());
    for (const auto &sm : mesh) {
        const auto &size(*isizes++);

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
        auto osm(vts::clip(sm, projected, rfNode.extents(), valid));
        if (osm.faces.empty()) { continue; }

        // at least one face survived remember
        mi.update(osm, size);
    }

    return mi;
}

struct LodParams {
    math::Extents2 meshExtents;
    vts::Lod lod;

    operator vts::Lod() const { return lod; }

    LodParams(const math::Extents2 &meshExtents = math::Extents2()
              , vts::Lod lod = 0)
        : meshExtents(meshExtents), lod(lod)
    {}
};

struct LodInfo {
    /** Rf subtree root to bottom lod mapping.
     */
    std::map<const vts::NodeInfo*, LodParams> localLods;

    /** Common min tree depth.
     *  This is depth where there are data available in all nodes.
     */
    int topDepth;

    /** Max tree depth.
     */
    int bottomDepth;

    LodInfo() : topDepth(), bottomDepth() {}

    int levelDiff() const { return bottomDepth - topDepth; }
};

class Analyzer {
public:
    Analyzer(const Config &config
             , const vts::NodeInfo::list &nodes
             , const slpk::Tree &tree
             , const slpk::Archive &archive)
        : config_(config)
        , nodes_(nodes), tree_(tree), archive_(archive)
        , inputSrs_(archive_.srs())
    {}

    LodInfo run(vt::ExternalProgress &progress) const;

private:
    typedef std::pair<int, int> DepthRange;

    DepthRange bottomMeshArea(const std::string &nodeId
                              , MeshInfo::map &mim) const;

    DepthRange bottomMeshAreaImpl(const std::string *nodeId
                                  , MeshInfo::map *mim) const;

    const Config &config_;
    const vts::NodeInfo::list &nodes_;
    const slpk::Tree &tree_;
    const slpk::Archive &archive_;
    const geo::SrsDefinition inputSrs_;
};

LodInfo Analyzer::run(vt::ExternalProgress &progress) const
{
    LOG(info3) << "Analyzing input dataset (" << tree_.nodes.size()
               << " I3S nodes).";
    progress.expect(tree_.nodes.size());

    MeshInfo::map mim;
    LodInfo lodInfo;
    std::tie(lodInfo.topDepth, lodInfo.bottomDepth)
        = bottomMeshArea(tree_.rootNodeId, mim);

    for (const auto &item : mim) {
        const auto bl(bestLod(config_, *item.first, item.second.area));
        lodInfo.localLods[item.first]
            = LodParams(item.second.extents, std::round(bl));
    }

    return lodInfo;
}

Analyzer::DepthRange
Analyzer::bottomMeshArea(const std::string &nodeId, MeshInfo::map &mim) const
{
    Analyzer::DepthRange out;
    auto *mimp(&mim);
    auto outp(&out);
    auto *nodeIdp(&nodeId);

    // run in parallel
    UTILITY_OMP(parallel)
    UTILITY_OMP(single)
    {
        *outp = bottomMeshAreaImpl(nodeIdp, mimp);
    }

    return out;
}

Analyzer::DepthRange
Analyzer::bottomMeshAreaImpl(const std::string *nodeId, MeshInfo::map *mim)
    const
{
    const auto *node(tree_.find(*nodeId));
    if (!node) {
        LOG(warn3) << "Referenced node <" << *nodeId << "> not found.";
    }

    if (!node->children.empty()) {
        int bottom(-1);
        int top(-1);

        std::vector<DepthRange> ranges;

        // process all children
        for (const auto &child : node->children) {
            const std::string *childId(&child.id);

            // descend
            UTILITY_OMP(task shared(ranges))
            {
                const auto dr(bottomMeshAreaImpl(childId, mim));
                UTILITY_OMP(critical(slpk2vts_meshInfo_1))
                    ranges.push_back(dr);
            }
        }

        // wait for task in the above loop
        UTILITY_OMP(taskwait)

        for (const auto &dr : ranges) {
            // update bottom depth
            if (dr.second > bottom) { bottom = dr.second; }

            // compute max common top depth
            if (dr.first > top) { top = dr.first; }
        }

        if (node->hasGeometry() && ((node->level + 1) == top)) {
            // valid node sitting above 4 valid nodes
            top = node->level;
        }

        return { top, bottom };
    }

    // load geometry
    VtsMeshLoader loader;
    archive_.loadGeometry(loader, *node);

    // measure textures
    std::vector<math::Size2> sizes;
    {
        for (std::size_t i(0), e(loader.mesh().size()); i != e; ++i) {
            sizes.push_back(archive_.textureSize(*node, i));
        }
    }

    if (!node->hasGeometry()) { return { -1, -1 }; }

    // bottom level -> compute best lod in each node
    for (const auto &rfNode : nodes_) {
        const vts::CsConvertor conv(inputSrs_, rfNode.srs());
        const auto mi(measureMesh(rfNode, conv, loader.mesh(), sizes));
        if (mi) {
            UTILITY_OMP(critical(slpk2vts_meshInfo_2))
                (*mim)[&rfNode] += mi;
        }
    }

    // default range for this node
    return { node->level, node->level };
}

void computeNavtileInfo(const Config &config, const vts::NodeInfo &node
                        , const LodParams lodParams, const LodInfo &lodInfo
                        , vts::NtGenerator &ntg)
{
    // find nt lod by nt lod pixelsize
    const auto nodeId(node.nodeId());

    // build LOD range
    vts::LodRange lr(0, nodeId.lod + lodParams.lod);

    auto lodDiff(lodInfo.levelDiff());
    if (lodDiff > lodParams.lod) {
        lr.min = nodeId.lod;
    } else {
        lr.min = nodeId.lod + lodParams.lod - lodDiff;
    }

    // nt lod, start with maximum lod
    vts::Lod ntLod(lodParams.lod + nodeId.lod);

    // sample one tile at bottom lod
    const vts::NodeInfo n(node.child
                          (vts::lowestChild(nodeId, lodParams.lod)));

    // tile size at bottom lod
    const auto ts(math::size(n.extents()));

    // take center of extents
    const auto ntCenter(math::center(lodParams.meshExtents));

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

    ntg.addAccumulator(node.srs(), vts::LodRange(lr.min, ntLod), pixelSize);
}

// ------------------------------------------------------------------------

class Cutter {
public:
    Cutter(const Config &config, const vr::ReferenceFrame &rf
           , tools::TmpTileset &tmpset, vts::NtGenerator &ntg
           , const slpk::Archive &archive)
        : config_(config), rf_(rf), tmpset_(tmpset), ntg_(ntg)
        , archive_(archive), nodes_(vts::NodeInfo::leaves(rf_))
        , inputSrs_(archive_.srs())
    {}

    void run(vt::ExternalProgress &progress);

private:
    void cutNode(const slpk::Node &node, const LodInfo &lodInfo);

    void splitToTiles(const vts::NodeInfo &root
                      , vts::Lod lod, const vts::TileRange &tr
                      , const vts::Mesh &mesh
                      , const vts::opencv::Atlas &atlas);

    void cutTile(const vts::NodeInfo &node, const vts::Mesh &mesh
                 , const vts::opencv::Atlas &atlas);

    cv::Mat loadTexture(const slpk::Node &node, int index) const;

    const Config &config_;
    const vr::ReferenceFrame &rf_;
    tools::TmpTileset &tmpset_;
    vts::NtGenerator &ntg_;
    const slpk::Archive &archive_;

    const vts::NodeInfo::list nodes_;
    const geo::SrsDefinition inputSrs_;
};

void Cutter::run(vt::ExternalProgress &progress)
{
    // load all available nodes
    const auto tree(archive_.loadTree());

    // analyze first
    const auto lodInfo(Analyzer(config_, nodes_, tree, archive_)
                       .run(progress));

    // compute navtile information (adds accumulators)
    for (const auto &item : lodInfo.localLods) {
        computeNavtileInfo(config_, *item.first, item.second, lodInfo, ntg_);
    }

    // update progress
    progress.expect(tree.nodes.size());

    // convert node map to node (pointer) list (needed by OpenMP to iterate over
    // nodes)
    auto nl([&]() -> std::vector<const slpk::Node*>
    {
        std::vector<const slpk::Node*> nl;
        nl.reserve(tree.nodes.size());
        for (const auto &item : tree.nodes) {
            nl.push_back(&item.second);
        }
        return nl;
    }());

    const std::size_t nlSize(nl.size());
    UTILITY_OMP(parallel for)
    for (std::size_t i = 0; i < nlSize; ++i) {
        const auto &node(*nl[i]);
        cutNode(node, lodInfo);
        ++progress;
    }
}

cv::Mat Cutter::loadTexture(const slpk::Node &node, int index) const
{
    const auto is(archive_.texture(node, index));
    LOG(info1) << "Loading texture from " << is->path() << ".";
    auto tex(cv::imdecode(is->read(), CV_LOAD_IMAGE_COLOR));

    if (!tex.data) {
        LOGTHROW(err2, std::runtime_error)
            << "Unable to load texture from " << is->path() << ".";
    }

    return tex;
}

vts::TileRange computeTileRange(const math::Extents2 &nodeExtents
                                , vts::Lod localLod
                                , const math::Extents2 &meshExtents)
{
    vts::TileRange r(math::InvalidExtents{});
    const auto ts(vts::tileSize(nodeExtents, localLod));
    const auto origin(math::ul(nodeExtents));

    for (const auto &p : vertices(meshExtents)) {
        update(r, vts::TileRange::point_type
               ((p(0) - origin(0)) / ts.width
                , (origin(1) - p(1)) / ts.height));
    }

    return r;
}

void Cutter::cutNode(const slpk::Node &node, const LodInfo &lodInfo)
{
    VtsMeshLoader loader;
    archive_.loadGeometry(loader, node);
    vts::opencv::Atlas inAtlas;
    for (std::size_t i(0), e(loader.mesh().size()); i != e; ++i) {
        inAtlas.add(loadTexture(node, i));
    }

    // for each valid rfnode
    for (const auto &item : lodInfo.localLods) {
        const auto rfNode(*item.first);
        const auto bottomLod(item.second);
        const vts::CsConvertor conv(inputSrs_, rfNode.srs());

        // compute local lod + sanity check
        const auto fromBottom(lodInfo.bottomDepth - node.level);
        if (fromBottom > bottomLod) {
            // out of reference frame -> skip
            continue;
        }

        const vts::Lod localLod(bottomLod - fromBottom);
        const auto lod(localLod + rfNode.nodeId().lod);

        // projested mesh/atlas
        vts::Mesh mesh;
        vts::opencv::Atlas atlas;

        // and for each submesh
        std::size_t meshIndex(0);
        for (auto &sm : loader.mesh()) {
            const auto &texture(inAtlas.get(meshIndex++));

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
            // FIXME: implement actual mask application in clipping!
            auto osm(vts::clip(sm, projected, rfNode.extents(), valid));
            if (osm.faces.empty()) { continue; }

            // at least one face survived, remember
            mesh.submeshes.push_back(std::move(osm));
            atlas.add(texture);

        }

        // anything there?
        if (mesh.empty()) { continue; }

        // compute local tile range
        auto tr(computeTileRange(rfNode.extents(), localLod
                                 , computeExtents(mesh)));

        // convert local tilerange to global tilerange
        {
            const auto origin
                (vts::lowestChild(vts::point(rfNode.nodeId()), localLod));
            tr.ll += origin;
            tr.ur += origin;
        }

        // split to tiles
        LOG(info3) << "Splitting I3S node <" << node.id << "> to tiles in "
                   << lod << "/" << tr << ".";
        splitToTiles(rfNode, lod, tr, mesh, atlas);
    }
}

void Cutter::splitToTiles(const vts::NodeInfo &root
                          , vts::Lod lod, const vts::TileRange &tr
                          , const vts::Mesh &mesh
                          , const vts::opencv::Atlas &atlas)
{
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

// ------------------------------------------------------------------------

const vt::ExternalProgress::Weights weightsFull{10, 40, 40, 10};
const vt::ExternalProgress::Weights weightsResume{40, 10};

class Encoder : public tools::TmpTsEncoder {
public:
    Encoder(const boost::filesystem::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const ::Config &config
            , vt::ExternalProgress::Config &&epConfig
            , const boost::optional<slpk::Archive> &input)
        : tools::TmpTsEncoder(path, properties, mode
                              , config, std::move(epConfig)
                              , weightsFull)
        , config_(config)
    {
        if (config.resume) { return; }
        if (!input) {
            LOGTHROW(err1, std::runtime_error)
                << "No archive passed while not resuming.";
        }

        Cutter(config_, referenceFrame(), tmpset(), ntg(), *input)
            .run(progress());
    }

private:
    const ::Config config_;
};

int Slpk2Vts::run()
{
#if 0
    // debug :)
    debug(slpk::Archive(input_));
    return EXIT_SUCCESS;
#endif

    vts::TileSetProperties properties;
    properties.referenceFrame = config_.referenceFrame;
    properties.id = config_.tilesetId;

    // open input if in non-resume mode
    boost::optional<slpk::Archive> input;
    if (!config_.resume) {
        input = boost::in_place(input_);

        // TODO: sanity check: mesh-pyramids, non-local
    }

    // run the encoder
    Encoder(output_, properties, createMode_, config_
            , std::move(epConfig_), input).run();

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[])
{
    utility::unlimitedCoredump();
    return Slpk2Vts()(argc, argv);
}
