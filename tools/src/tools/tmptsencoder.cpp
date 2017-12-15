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

#include <boost/logic/tribool.hpp>

#include "vts-libs/vts/meshop.hpp"
#include "vts-libs/vts/tileset/merge.hpp"
#include "vts-libs/registry/po.hpp"

#include "./tmptsencoder.hpp"

namespace fs = boost::filesystem;
namespace vs = vtslibs::storage;
namespace vr = vtslibs::registry;
namespace vts = vtslibs::vts;
namespace po = boost::program_options;

namespace vtslibs { namespace vts { namespace tools {

namespace {

boost::optional<fs::path> optionalPath(bool valid, const fs::path &path)
{
    if (valid) { return path; }
    return boost::none;
}

inline void warpInPlace(const CsConvertor &conv, SubMesh &sm)
{
    for (auto &v : sm.vertices) { v = conv(v); }
}

inline void warpInPlace(const CsConvertor &conv, Mesh &mesh)
{
    for (auto &sm : mesh) { warpInPlace(conv, sm); }
}

} // namespace

TmpTsEncoder::TmpTsEncoder(const boost::filesystem::path &path
                           , const TileSetProperties &properties
                           , CreateMode mode
                           , const Config &config
                           , ExternalProgress::Config &&epConfig
                           , const ExternalProgress::Weights &weights)
    : Encoder(path, properties, mode)
    , config_(config)
    , progress_(std::move(epConfig), weights)
    , tmpset_(path / "tmp", !config_.resume)
    , ntg_(&referenceFrame()
           , optionalPath(config_.resume, tmpset_.root() / "navtile.info"))
{
    tmpset_.keep(config.keepTmpset);
}

TmpTsEncoder::~TmpTsEncoder()
{
    if (!std::uncaught_exception()) {
        progress_.done();
    }
}

void TmpTsEncoder::run()
{
    prepare();
    Encoder::run();
}

void TmpTsEncoder::prepare()
{
    if (!config_.resume) {
        tmpset_.flush();
        ntg_.save(tmpset_.root() / "navtile.info");
    }

    // base trees
    deriveTree_ = index_ = tmpset_.tileIndex();

    // derive tree: every tile not in index_ is to be derived
    // every alien marked tile will have its children marked as well
    deriveTree_
        .conditionalRound
        (TileIndex::Flag::mesh | TileIndex::Flag::watertight
         , TileIndex::Flag::mesh)
        .distributeFlags(TileIndex::Flag::alien)
        ;

    // make valid tree complete from root
    validTree_ = deriveTree_;
    validTree_.makeAbsolute().complete();

    // cherry pick path to debug tileId (if configured)
    if (config_.debug_tileId) {
        TileIndex constraint(LodRange(0, validTree_.maxLod()));

        auto tileId(*config_.debug_tileId);
        for (int i(tileId.lod); i >= 0; --i) {
            constraint.set(tileId, 1);
            tileId = parent(tileId);
        }

        validTree_ = validTree_.intersect(constraint);
    }

    setConstraints(Constraints().setValidTree(&validTree_));

    // set estimated count including tiles to derive
    const auto count(deriveTree_.count());
    setEstimatedTileCount(count);
    progress_.expect(count);
}

namespace {

typedef opencv::HybridAtlas HybridAtlas;

class DataSource : public MeshOpInput::DataSource {
public:
    DataSource(const NodeInfo &nodeInfo
               , const Mesh::pointer &mesh
               , const HybridAtlas::pointer &atlas)
        : MeshOpInput::DataSource({}), nodeInfo_(nodeInfo)
        , mesh_(mesh), atlas_(atlas)
    {}

    virtual ~DataSource() {}

private:
    void check(const TileId &tileId) const {
        if (tileId == nodeInfo_.nodeId()) { return; }
        LOGTHROW(err2, std::runtime_error)
            << "This datasource provides data only for "
            << nodeInfo_.nodeId() << ", not " << tileId << ".";
    }

    virtual TileIndex::Flag::value_type flags_impl(const TileId &tileId)
        const
    {
        check(tileId);
        return (TileIndex::Flag::mesh | TileIndex::Flag::atlas);
    }

    virtual const MetaNode* findMetaNode_impl(const TileId &tileId)
        const
    {
        check(tileId);
        return &node_;
    }

    virtual Mesh::pointer
    getMesh_impl(const TileId &tileId, TileIndex::Flag::value_type) const
    {
        check(tileId);
        return mesh_;
    }

    virtual HybridAtlas::pointer
    getAtlas_impl(const TileId &tileId, TileIndex::Flag::value_type) const
    {
        check(tileId);
        return atlas_;
    }

    virtual opencv::NavTile::pointer
    getNavTile_impl(const TileId &tileId, const MetaNode*)
        const
    {
        check(tileId);
        LOGTHROW(err2, std::runtime_error)
            << "No navtile for tile " << tileId << ".";
        throw;
    }

    virtual NodeInfo nodeInfo_impl(const TileId &tileId) const
    {
        check(tileId);
        return nodeInfo_;
    }

    NodeInfo nodeInfo_;
    Mesh::pointer mesh_;
    HybridAtlas::pointer atlas_;
    MetaNode node_;
};

const auto mergeOptions([]() -> MergeOptions
{
    MergeOptions mo;
    mo.glueMode = GlueMode::simpleClip;
    return mo;
}());

const auto mergeExtraOptions([]() -> merge::ExtraOptions
{
    merge::ExtraOptions eo;
    eo.meshesInSds = true;
    return eo;
}());

const merge::TileSource emptyTileSource;

typedef std::tuple<Mesh::pointer, HybridAtlas::pointer> MeshResult;

struct MergeConstraints : merge::MergeConstraints {
    MergeConstraints(TileIndex::Flag::value_type flags
                     , Encoder::TileResult &tileResult
                     , bool influencedOnly)
        : merge::MergeConstraints(!(flags & TileIndex::Flag::watertight))
        , tileResult(tileResult)
        , influencedOnly(influencedOnly)
    {
        if (influencedOnly) {
            // default to no data
            tileResult.noDataYet();
        }
    }

    virtual bool feasible(const merge::Output&) const {
        if (influencedOnly) {
            // ha! this tile would have some content but we just need to know
            // about it
            tileResult.influenced();
            return false;
        }

        return true;
    }

    Encoder::TileResult &tileResult;
    bool influencedOnly;
};

MeshResult glueHolesFromParent(const TileId &tileId
                               , const NodeInfo &nodeInfo
                               , const Encoder::TileResult &parent
                               , const Mesh::pointer &mesh
                               , const HybridAtlas::pointer &atlas
                               , TileIndex::Flag::value_type flags
                               , bool influencedOnly
                               , Encoder::TileResult &tileResult)
{
    merge::Input::list input;
    if (mesh && atlas) {
        input.emplace_back
            (tileId.lod
             , std::make_shared<DataSource>(nodeInfo, mesh, atlas)
             , tileId, &nodeInfo);
    }

    // merge tile; tile is generated only when it's not marked as a watertight
    auto result
        (merge::mergeTile
         (tileId, nodeInfo, input
          , parent.userDataWithDefault(emptyTileSource)
          , MergeConstraints(flags, tileResult, influencedOnly)
          , mergeOptions, mergeExtraOptions));

    // store used sources
    tileResult.userData(result.source);

    // we are processing single tileset -> reset surface reference
    if (result.mesh) { result.mesh->resetSurfaceReference(); }

    return MeshResult(std::move(result.mesh), std::move(result.atlas));
}

} // namespace

Encoder::TileResult
TmpTsEncoder::generate(const TileId &tileId, const NodeInfo &nodeInfo
                       , const TileResult &parent)
{
    boost::tribool derive(false);
    if (!index_.get(tileId)) {
        const auto dv(deriveTree_.get(tileId));
        if (dv & TileIndex::Flag::mesh) {
            // we are trying to fully derive a tile from parent data
            derive = true;
        } else if (dv & TileIndex::Flag::alien) {
            // deriving but not generating tile, only flag "influenced" will be
            // stored in the output
            derive = boost::indeterminate;
        } else {
            return TileResult::Result::noDataYet;
        }
    }

    // dst SDS -> dst physical
    const CsConvertor sds2DstPhy
        (nodeInfo.srs(), referenceFrame().model.physicalSrs);

    TileResult result(TileResult::Result::tile);

    // create tile
    auto &tile(result.tile());

    // holding original mesh and atlas
    Mesh::pointer mesh;
    HybridAtlas::pointer atlas;
    TileIndex::Flag::value_type flags(TileIndex::Flag::none);

    if (!derive) {
        // we have data, load them and process

        // load tile
        std::tie(mesh, atlas, flags)
            = tmpset_.load(tileId, config_.textureQuality);

        // generate external texture coordinates
        generateEtc(*mesh, nodeInfo.extents()
                    , nodeInfo.node().externalTexture);

        // generate mesh mask (may be overrideny by full mask if asked to do so)
        generateCoverage(*mesh, nodeInfo.extents());
    }

    // glue holes from parent
    std::tie(mesh, atlas)
        = glueHolesFromParent(tileId, nodeInfo, parent, mesh, atlas
                              , flags, boost::indeterminate(derive)
                              , result);

    // influenced tile -> stop here
    if (boost::indeterminate(derive)) {
        ++progress_;
        if (result.result() != TileResult::Result::influenced) {
            updateEstimatedTileCount(-1);
        }
        return result;
    }

    // sanity check
    if (!mesh || mesh->empty()) {
        if (!derive) {
            LOG(warn3) << "Generated non-derived empty tile!";
        }

        ++progress_;
        updateEstimatedTileCount(-1);
        return TileResult::Result::noDataYet;
    }

    // fuse individual submeshes if asked to
    if (config_.fuseSubmeshes) {
        // merge submeshes
        std::tie(tile.mesh, tile.atlas)
            = mergeSubmeshes
            (tileId, mesh, atlas
             , config_.textureQuality, config_.smMergeOptions);
    } else {
        tile.mesh = std::make_shared<Mesh>(*mesh);
        tile.atlas = atlas;
    }

    // mesh in SDS -> pre-compute geom extents
    tile.geomExtents = geomExtents(*tile.mesh);

    // add tile to navtile generator
    ntg_.addTile(tileId, nodeInfo, *tile.mesh);

    // generate coverage
    if (config_.forceWatertight) {
        tile.mesh->createCoverage(true);
    } else {
        generateCoverage(*tile.mesh, nodeInfo.extents());
    }

    // finally, warp mesh to physical SRS
    warpInPlace(sds2DstPhy, *tile.mesh);

    // set credits
    tile.credits = config_.credits;

    // done
    ++progress_;

    // done
    return result;
}

void TmpTsEncoder::finish(TileSet &ts)
{
    ntg_.generate(ts, config_.dtmExtractionRadius, progress_);
}

void TmpTsEncoder::Config::configuration(po::options_description &config)
{
    config.add_options()
        ("textureQuality", po::value(&textureQuality)
         ->default_value(textureQuality)->required()
         , "Texture quality for JPEG texture encoding (1-100). "
         "NB: Special value 0 causes generation lossless PNG textures.")

        ("dtmExtraction.radius"
         , po::value(&dtmExtractionRadius)
         ->default_value(dtmExtractionRadius)->required()
         , "Radius (in meters) of DTM extraction element (in meters).")

        ("force.watertight", po::value(&forceWatertight)
         ->default_value(false)->implicit_value(true)
         , "Enforces full coverage mask to every generated tile even "
         "when it is holey.")

        ("resume"
         , "Resumes interrupted encoding. There must be complete (valid) "
         "temporary tileset inside generated tileset. Use with caution.")
        ("keepTmpset"
         , "Keep temporary tileset intact on exit.")

        ("fuseSubmeshes", po::value(&fuseSubmeshes)
         ->default_value(fuseSubmeshes)->required()
         , "Fuse submeshes into bigger submeshes (ideally one).")

        ("debug.tileId", po::value<TileId>()
         , "Limits output to tiles in the path to "
         "given tileId (optional, for debug purposes).")
        ;
}

void TmpTsEncoder::Config::configure(const po::variables_map &vars)
{
    credits = registry::creditsConfigure(vars);
    if ((textureQuality < 0) || (textureQuality > 100)) {
        throw po::validation_error
            (po::validation_error::invalid_option_value, "textureQuality");
    }

    resume = vars.count("resume");
    keepTmpset = vars.count("keepTmpset");

    if (vars.count("debug.tileId")) {
        debug_tileId = vars["debug.tileId"].as<TileId>();
    }
}

} } } // namespace vtslibs::vts::tools
