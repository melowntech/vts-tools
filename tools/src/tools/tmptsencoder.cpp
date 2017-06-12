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

#include "vts-libs/vts/meshop.hpp"
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

inline void warpInPlace(const vts::CsConvertor &conv, vts::SubMesh &sm)
{
    for (auto &v : sm.vertices) { v = conv(v); }
}

inline void warpInPlace(const vts::CsConvertor &conv, vts::Mesh &mesh)
{
    for (auto &sm : mesh) { warpInPlace(conv, sm); }
}

} // namespace

TmpTsEncoder::TmpTsEncoder(const boost::filesystem::path &path
                           , const vts::TileSetProperties &properties
                           , vts::CreateMode mode
                           , const Config &config
                           , ExternalProgress::Config &&epConfig
                           , const ExternalProgress::Weights &weights)
    : vts::Encoder(path, properties, mode)
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
    vts::Encoder::run();
}

void TmpTsEncoder::prepare()
{
    if (!config_.resume) {
        tmpset_.flush();
        ntg_.save(tmpset_.root() / "navtile.info");
    }

    validTree_ = index_ = tmpset_.tileIndex();

    // make valid tree complete from root
    validTree_.makeAbsolute().complete();

    setConstraints(Constraints().setValidTree(&validTree_));
    const auto count(index_.count());
    setEstimatedTileCount(count);
    progress_.expect(count);
}

Encoder::TileResult
TmpTsEncoder::generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
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
    ++progress_;

    // done
    return result;
}

void TmpTsEncoder::finish(vts::TileSet &ts)
{
    ntg_.generate(ts, config_.dtmExtractionRadius, progress_);
}

void TmpTsEncoder::Config::configuration(po::options_description &config)
{
    config.add_options()
        ("textureQuality", po::value(&textureQuality)
         ->default_value(textureQuality)->required()
         , "Texture quality for JPEG texture encoding (0-100).")

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

}

} } } // namespace vtslibs::vts::tools
