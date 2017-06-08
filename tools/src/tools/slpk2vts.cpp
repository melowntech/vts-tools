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

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/limits.hpp"
#include "utility/path.hpp"

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

    cmdline.add_options()
        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")
        ("input", po::value(&input_)->required()
         , "Path to input SLPK archive.")
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

        ("zShift", po::value(&config_.zShift)
         ->default_value(config_.zShift)->required()
         , "Manual height adjustment (value is "
         "added to z component of all vertices).")

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

    for (const auto &n : input.loadTree()) {
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

#if 0
const vt::ExternalProgress::Weights weightsFull{10, 40, 40, 10};
const vt::ExternalProgress::Weights weightsResume{40, 10};
#else
const vt::ExternalProgress::Weights weightsFull{40, 10};
const vt::ExternalProgress::Weights weightsResume{40, 10};
#endif

class Encoder : public tools::TmpTsEncoder {
public:
    Encoder(const boost::filesystem::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const Config &config
            , vt::ExternalProgress::Config &&epConfig
            , const slpk::Archive &input)
        : tools::TmpTsEncoder(path, properties, mode
                              , config, std::move(epConfig)
                              , weightsFull)
        , config_(config)
    {
        // process input
        (void) input;
    }

    Encoder(const boost::filesystem::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const Config &config
            , vt::ExternalProgress::Config &&epConfig)
        : tools::TmpTsEncoder(path, properties, mode
                              , config, std::move(epConfig)
                              , weightsResume)
        , config_(config)
    {}

private:
    void prepareTiles(tools::TmpTileset &tmpset
                      , vt::ExternalProgress &progress);

    const Config config_;
};

void Encoder::prepareTiles(tools::TmpTileset &tmpset
                           , vt::ExternalProgress &progress)
{
    (void) tmpset;
    (void) progress;
}

int Slpk2Vts::run()
{
    vts::TileSetProperties properties;
    properties.referenceFrame = config_.referenceFrame;
    properties.id = config_.tilesetId;

    if (config_.resume) {
        // run the encoder
        Encoder(output_, properties, createMode_, config_
                , std::move(epConfig_)).run();

        // all done
        LOG(info4) << "All done.";
        return EXIT_SUCCESS;
    }

    // open archive
    slpk::Archive input(input_);

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
