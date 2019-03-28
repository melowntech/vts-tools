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
#include "vts-libs/tools-support/progress.hpp"

#include "slpk/reader.hpp"

#include "vts-libs/tools-support/tmptsencoder.hpp"
#include "vts-libs/tools-support/repackatlas.hpp"
#include "vts-libs/tools-support/analyze.hpp"

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
    double zShift;

    Config()
        : optimalTextureSize(256, 256)
        , ntLodPixelSize(1.0)
        , clipMargin(1.0 / 128.)
        , borderClipMargin(clipMargin)
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

// ------------------------------------------------------------------------

typedef tools::TextureRegionInfo RegionInfo;

/** Loads SLPK geometry as a list of submeshes.
 */
class VtsMeshLoader
    : public slpk::GeometryLoader
    , public slpk::MeshLoader
{
public:
    VtsMeshLoader() : current_(nullptr), currentRInfo_(nullptr) {}

    virtual slpk::MeshLoader& next() {
        mesh_.submeshes.emplace_back();
        current_ = &mesh_.submeshes.back();

        regions_.emplace_back();
        currentRInfo_ = &regions_.back();
        return *this;
    }

    const vts::Mesh& mesh() const { return mesh_; }
    const RegionInfo::list& regions() const { return regions_; }

    virtual void addVertex(const math::Point3d &v) {
        current_->vertices.push_back(v);
    }

    virtual void addTexture(const math::Point2d &t) {
        current_->tc.push_back(t);
    }

    virtual void addFace(const Face &mesh, const FaceTc &tc, const Face&) {
        current_->faces.push_back(mesh);
        current_->facesTc.push_back(tc);
        currentRInfo_->faces.push_back(tc.region);
    }

    virtual void addTxRegion(const Region &region) {
        currentRInfo_->regions.emplace_back(region);
    }

private:
    virtual void addNormal(const math::Point3d&) {}

    vts::Mesh mesh_;
    vts::SubMesh *current_;
    RegionInfo::list regions_;
    RegionInfo *currentRInfo_;
};

// ------------------------------------------------------------------------

void remapTcToRegion(vts::SubMesh &sm, const vts::FaceOriginList &faceOrigin
                     , const RegionInfo &ri)
{
    if (ri.regions.empty()) {
        // nothing to inflate
        return;
    }

    // remap texture coordinates from region coordinates to texture coordinates

    std::vector<char> seen(sm.tc.size(), false);
    const auto &remap([&](int index, const RegionInfo::Region &region) -> void
    {
        auto &iseen(seen[index]);
        if (iseen) { return; }

        // remap from region space to texture space
        auto &tc(sm.tc[index]);
        tc(0) *= region.size.width;
        tc(1) *= region.size.height;

        iseen = true;
    });

    auto ifaceOrigin(faceOrigin.begin());
    for (const auto &face : sm.facesTc) {
        // grab index to regions
        const auto regionIndex(ri.faces[*ifaceOrigin++]);
        const auto &region(ri.regions[regionIndex]);

        for (const auto &tc : face) { remap(tc, region); }
    }
}

tools::LodInfo analyze(vt::ExternalProgress &progress
                       , const Config &config
                       , const vts::NodeInfo::list &nodes
                       , const slpk::Tree &tree
                       , const slpk::Archive &archive)
{
    LOG(info3) << "Analyzing input dataset (" << tree.nodes.size()
               << " I3S nodes).";
    progress.expect(tree.nodes.size());

    const geo::SrsDefinition inputSrs(archive.srs());

    // find limits for data nodes: top/bottom and bottom common to all subtrees
    tools::LodInfo lodInfo;
    lodInfo.topDepth = std::numeric_limits<int>::max();
    lodInfo.commonBottom = std::numeric_limits<int>::max();
    lodInfo.bottomDepth = -1;

    {
        for (const auto item : tree.nodes) {
            const auto &node(item.second);
            if (!node.hasGeometry()) { continue; }
            if (node.children.empty()) {
                // leaf
                lodInfo.commonBottom
                    = std::min(lodInfo.commonBottom, node.level);
                lodInfo.bottomDepth
                    = std::max(lodInfo.bottomDepth, node.level);
            }

            // update top
            lodInfo.topDepth = std::min(lodInfo.topDepth, node.level);
        }

        LOG(info2) << "Found top/common-bottom/bottom: "
                   << lodInfo.topDepth << "/" << lodInfo.commonBottom
                   << "/" << lodInfo.bottomDepth << ".";
    }

    tools::MeshInfo::map mim;

    // accumulate mesh area (both 3D and 2D) in all nodes at common bottom depth

    // collect nodes for OpenMP
    std::vector<const slpk::Node*> treeNodes;
    for (const auto &item : tree.nodes) {
        const auto &node(item.second);
        if ((node.level == lodInfo.commonBottom) && (node.hasGeometry())) {
            treeNodes.push_back(&node);
        }
    }

    auto *pmim(&mim);
    const auto *pnodes(&treeNodes);

    UTILITY_OMP(parallel for shared(pmim) schedule(dynamic))
    for (std::size_t i = 0; i < pnodes->size(); ++i) {
        const auto &node(*(*pnodes)[i]);

        // load geometry
        VtsMeshLoader loader;
        archive.loadGeometry(loader, node);

        // measure textures
        std::vector<math::Size2> sizes;
        {
            for (std::size_t i(0), e(loader.mesh().size()); i != e; ++i) {
                sizes.push_back(archive.textureSize(node, i));
            }
        }

        // compute mesh area in each RF node
        for (const auto &rfNode : nodes) {
            const vts::CsConvertor conv(inputSrs, rfNode.srs());
            const auto mi(tools::measureMesh(rfNode, conv, loader.mesh()
                                             , loader.regions(), sizes));
            if (mi) {
                UTILITY_OMP(critical(slpk2vts_meshInfo_1))
                    (*pmim)[&rfNode] += mi;
            }
        }
    }

    // shift between common depth and bottom depth
    const auto lodShift(lodInfo.bottomDepth - lodInfo.commonBottom);

    for (const auto &item : mim) {
        const auto bl
            (lodShift + tools::bestLod(*item.first, item.second.area
                                       , config.optimalTextureSize));
        const auto &lod(lodInfo.localLods[item.first]
                        = tools::LodParams(item.second.extents
                                           , std::round(bl)));
        LOG(info3)
            << "Assigned LOD " << (item.first->nodeId().lod + lod)
            << " (local LOD " << lod
            << ") for bottom depth (" << lodInfo.bottomDepth
            << ") in subtree " << item.first->srs() << ".";
    }

    return lodInfo;
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
    void cutNode(const slpk::Node &node, const tools::LodInfo &lodInfo);

    void splitToTiles(const slpk::Node &slpkNode
                      , const vts::NodeInfo &root
                      , vts::Lod lod, const vts::TileRange &tr
                      , const vts::Mesh &mesh
                      , const RegionInfo::list &textureRegions
                      , const vts::opencv::Atlas &atlas
                      , vts::TileIndex::Flag::value_type tileFlags);

    void cutTile(const slpk::Node &slpkNode, const vts::NodeInfo &node
                 , const vts::Mesh &mesh
                 , const RegionInfo::list &textureRegions
                 , const vts::opencv::Atlas &atlas
                 , vts::TileIndex::Flag::value_type tileFlags);

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
    const auto lodInfo(analyze(progress, config_, nodes_, tree, archive_));

    // compute navtile information (adds accumulators)
    for (const auto &item : lodInfo.localLods) {
        tools::computeNavtileInfo(*item.first, item.second, lodInfo, ntg_
                                  , config_.tileExtents
                                  , config_.ntLodPixelSize);
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
    UTILITY_OMP(parallel for schedule(dynamic))
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

void Cutter::cutNode(const slpk::Node &node, const tools::LodInfo &lodInfo)
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

        /** (extra) Tile flags to be stored along generate tiles from this node
         */
        vts::TileIndex::Flag::value_type tileFlags(0);
        if (node.level <= lodInfo.commonBottom) {
            // mark all tiles not below common bottom level as watertight
            tileFlags |= vts::TileIndex::Flag::watertight;

            if (node.level == lodInfo.commonBottom) {
                // mark all tiles at commom bottom level as alien
                tileFlags |= vts::TileIndex::Flag::alien;
            }
        }

        // projected mesh/atlas
        vts::Mesh mesh;
        RegionInfo::list textureRegions;
        vts::opencv::Atlas atlas;

        // and for each submesh
        std::size_t meshIndex(0);
        auto iregions(loader.regions().begin());
        for (auto &sm : loader.mesh()) {
            const auto &texture(inAtlas.get(meshIndex++));
            const auto &srcRi(*iregions++);

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
            vts::FaceOriginList faceOrigin;
            auto osm(vts::clip(sm, projected, rfNode.extents(), valid
                               , &faceOrigin));
            if (osm.faces.empty()) { continue; }

            // at least one face survived, remember

            // get texturing info
            textureRegions.emplace_back(srcRi.regions);
            auto &tr(textureRegions.back());
            // remap face regions (if any)
            if (!srcRi.regions.empty()) {
                for (const auto fo : faceOrigin) {
                    tr.faces.push_back(srcRi.faces[fo]);
                }
            }

            mesh.submeshes.push_back(std::move(osm));
            atlas.add(texture);
        }

        // anything there?
        if (mesh.empty()) { continue; }

        // compute local tile range
        auto tr(tools::computeTileRange(rfNode.extents(), localLod
                                        , tools::computeExtents(mesh)));

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
        splitToTiles(node, rfNode, lod, tr, mesh, textureRegions, atlas
                     , tileFlags);
    }
}

void Cutter::splitToTiles(const slpk::Node &slpkNode
                          , const vts::NodeInfo &root
                          , vts::Lod lod, const vts::TileRange &tr
                          , const vts::Mesh &mesh
                          , const RegionInfo::list &textureRegions
                          , const vts::opencv::Atlas &atlas
                          , vts::TileIndex::Flag::value_type tileFlags)
{
    if (config_.tileExtents) {
        // check for range validity
        const auto &rootId(root.nodeId());
        if (lod < config_.tileExtents->lod) {
            LOG(info2)
                << "Nothing to cut from SLPK node <" << slpkNode.id << ">.";
        }

        const auto gtr(vts::global(rootId, lod, tr));
        const auto extents(vts::shiftRange(*config_.tileExtents, lod));

        if (!vts::tileRangesOverlap(gtr, extents)) {
            LOG(info2)
                << "Nothing to cut from SLPK node <" << slpkNode.id << ">"
                << ", gtr: " << gtr << ", extents: " << extents << ".";
            return;
        }
    }

    typedef vts::TileRange::value_type Index;
    Index je(tr.ur(1));
    Index ie(tr.ur(0));

    for (Index j = tr.ll(1); j <= je; ++j) {
        for (Index i = tr.ll(0); i <= ie; ++i) {
            vts::TileId tileId(lod, i, j);
            const auto node(root.child(tileId));
            cutTile(slpkNode, node, mesh, textureRegions, atlas, tileFlags);
        }
    }
}

void Cutter::cutTile(const slpk::Node &slpkNode, const vts::NodeInfo &node
                     , const vts::Mesh &mesh
                     , const RegionInfo::list &textureRegions
                     , const vts::opencv::Atlas &atlas
                     , vts::TileIndex::Flag::value_type tileFlags)
{

    // compute border condition (defaults to all available)
    vts::BorderCondition borderCondition;
    if (config_.tileExtents) {
        borderCondition = vts::inside(*config_.tileExtents, node.nodeId());
        if (!borderCondition) {
            LOG(info1)
                << node.nodeId() << ": Nothing to cut from SLPK node <"
                << slpkNode.id << ">.";
            return;
        }
    }

    // compute clip extents
    const auto extents(vts::inflateTileExtents
                       (node.extents(), config_.clipMargin
                        , borderCondition, config_.borderClipMargin));

    vts::Mesh clipped;
    vts::opencv::Atlas clippedAtlas(0); // PNG!

    std::size_t smIndex(0);
    std::size_t faces(0);
    for (const auto &sm : mesh) {
        const auto &texture(atlas.get(smIndex++));

        auto m(vts::clip(sm, extents));
        if (m.empty()) { continue; }

        clipped.submeshes.push_back(std::move(m));
        clippedAtlas.add(texture);
        faces += clipped.submeshes.back().faces.size();
    }

    if (clipped.empty()) {
        LOG(info1)
            << node.nodeId() << ": Nothing cut from SLPK node <"
            << slpkNode.id << ">.";
        return;
    }

    LOG(info2)
        << node.nodeId() << ": Cut " << faces
        << " faces from SLPK node <" << slpkNode.id << ">.";

    // store in temporary storage
    const auto tileId(node.nodeId());

    tools::repack(tileId, clipped, clippedAtlas, textureRegions);
    tmpset_.store(tileId, clipped, clippedAtlas, tileFlags);
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
                              , (config.resume ? weightsResume : weightsFull))
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
