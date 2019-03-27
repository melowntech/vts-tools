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
#include <boost/algorithm/string/predicate.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/limits.hpp"
#include "utility/path.hpp"
#include "utility/openmp.hpp"

#include "service/cmdline.hpp"

#include "geometry/meshop.hpp"

#include "imgproc/imagesize.hpp"

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

#include "vts-libs/tools-support/tmptsencoder.hpp"
#include "vts-libs/tools-support/repackatlas.hpp"
#include "vts-libs/tools-support/analyze.hpp"

#include "3dtiles/3dtiles.hpp"
#include "3dtiles/reader.hpp"
#include "3dtiles/b3dm.hpp"
#include "3dtiles/io.hpp"

namespace po = boost::program_options;
namespace bio = boost::iostreams;
namespace fs = boost::filesystem;
namespace ba = boost::algorithm;
namespace ublas = boost::numeric::ublas;
namespace vs = vtslibs::storage;
namespace vr = vtslibs::registry;
namespace vts = vtslibs::vts;
namespace vt = vtslibs::tools;
namespace tools = vtslibs::vts::tools;
namespace tdt = threedtiles;

namespace {

struct Config : tools::TmpTsEncoder::Config {
    geo::SrsDefinition inputSrs;

    std::string tilesetId;
    std::string referenceFrame;
    math::Size2 optimalTextureSize;
    double ntLodPixelSize;

    boost::optional<vts::LodTileRange> tileExtents;
    double clipMargin;
    double borderClipMargin;
    double zShift;

    Config()
        : inputSrs(4328)
        , optimalTextureSize(256, 256)
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

            ("clipMargin"
             , po::value(&clipMargin)->default_value(clipMargin)
             , "Margin (in fraction of tile dimensions) added to tile extents "
             "in all 4 directions.")

            ("tileExtents", po::value<vts::LodTileRange>()
             , "Optional tile extents specidied in form lod/llx,lly:urx,ury. "
             "When set, only tiles in that range and below are added to "
             "the output.")

            ("borderClipMargin"
             , po::value(&borderClipMargin)->default_value(borderClipMargin)
             , "Margin (in fraction of tile dimensions) added to tile extents "
             "where tile touches artificial border definied by tileExtents.")

            ("inputSrs", po::value(&inputSrs)->default_value(inputSrs)
             , "SRS of input 3D Tiles tileset. It is usually "
             "geocent (EPSG:4328).")

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

class Tdt2Vts : public service::Cmdline
{
public:
    Tdt2Vts()
        : service::Cmdline("3dtiles2vts", BUILD_TARGET_VERSION)
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

void Tdt2Vts::configuration(po::options_description &cmdline
                             , po::options_description &config
                             , po::positional_options_description &pd)
{
    config_.configuration(cmdline);

    cmdline.add_options()
        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")
        ("input", po::value(&input_)->required()
         , "Path to input 3D Tileset archive.")
        ("overwrite", "Existing tile set gets overwritten if set.")
        ;

    vt::progressConfiguration(config);

    pd
        .add("input", 1)
        .add("output", 1);

    (void) config;
}

void Tdt2Vts::configure(const po::variables_map &vars)
{
    config_.configure(vars);

    createMode_ = (vars.count("overwrite")
                   ? vts::CreateMode::overwrite
                   : vts::CreateMode::failIfExists);

    epConfig_ = vt::configureProgress(vars);
}

bool Tdt2Vts::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(3dtiles2vts
usage
    3dtiles2vts INPUT OUTPUT [OPTIONS]

)RAW";
    }
    return false;
}

class SizeOnlyAtlas : public vts::Atlas {
public:
    typedef std::shared_ptr<SizeOnlyAtlas> pointer;

    virtual std::size_t size() const { return images_.size(); }

    using Image = math::Size2;
    using Images = std::vector<Image>;

    const Image& get(std::size_t index) const { return images_[index]; }

    void add(const Image &image) { images_.push_back(image); }

    /** Access internal data.
     */
    const Images& get() const { return images_; }

private:
    virtual vts::multifile::Table serialize_impl(std::ostream&) const {
        LOGTHROW(err2, vs::Unimplemented)
            << "SizeOnlyAtlas cannot be serialized because it is a "
            "phony atlas.";
        throw;
    }

    virtual void deserialize_impl(std::istream&, const fs::path&
                                  , const vts::multifile::Table&)
    {
        LOGTHROW(err2, vs::Unimplemented)
            << "SizeOnlyAtlas cannot be deserialized because it is a "
            "phony atlas.";
    }

    virtual math::Size2 imageSize_impl(std::size_t index) const {
        return get(index);
    }

    virtual void write_impl(std::ostream&, std::size_t) const {
        LOGTHROW(err2, vs::Unimplemented)
            << "Cannot write image because SizeOnlyAtlas is a phony atlas.";
    }

    Images images_;
};

void addImage(vts::RawAtlas &atlas, const gltf::DataView &data
              , const std::string&)
{
    atlas.add(vts::RawAtlas::Image(data.first, data.second));
}

void addImage(SizeOnlyAtlas &atlas, const gltf::DataView &data
              , const std::string &filename)
{
    atlas.add(imgproc::imageSize(data.first, gltf::size(data), filename));
}

template <typename Atlas>
class VtsMeshLoader : public gltf::MeshLoader {
public:
    VtsMeshLoader(const std::string &filename)
        : filename_(filename)
        , sm_(nullptr)
    {
        LOG(info2) << "Loading mesh from " << filename << ".";
    }

    std::pair<vts::Mesh&, Atlas&> get() {
        // TODO: optimize mesh
        if (mesh_.submeshes.size() != atlas_.size()) {
            LOGTHROW(err2, std::runtime_error)
                << "Some submeshes are not textured in tile <"
                << filename_ << ">.";
        }
        return std::pair<vts::Mesh&, Atlas&>(mesh_, atlas_);
    }

protected:
    virtual void mesh() {
        mesh_.submeshes.emplace_back();
        sm_ = &mesh_.submeshes.back();
    }

    virtual void vertices(math::Points3d &&v) {
        sm_->vertices = std::move(v);
    }

    virtual void tc(math::Points2d &&tc) {
        sm_->tc = std::move(tc);
    }

    virtual void faces(Faces &&faces) {
        // use the same indices for both 3D and 2D faces
        sm_->faces.assign(faces.begin(), faces.end());
        sm_->facesTc.assign(faces.begin(), faces.end());
    }

    virtual void image(const gltf::DataView &imageData) {
        addImage(atlas_, imageData, filename_);
    }

    std::string filename_;

    vts::Mesh mesh_;
    Atlas atlas_;
    vts::SubMesh *sm_;
};

// ------------------------------------------------------------------------

class Cutter {
public:
    Cutter(const Config &config, const vr::ReferenceFrame &rf
           , tools::TmpTileset &tmpset, vts::NtGenerator &ntg
           , const tdt::Archive &archive)
        : config_(config), rf_(rf), tmpset_(tmpset), ntg_(ntg)
        , archive_(archive), tileset_(archive.tileset())
        , nodes_(vts::NodeInfo::leaves(rf_))
    {}

    void run(vt::ExternalProgress &progress);

private:
    void cutNode(const tdt::Tile &node
                 , const tools::LodInfo &lodInfo);

    void splitToTiles(const tdt::Tile &tdtTile
                      , const vts::NodeInfo &root
                      , vts::Lod lod, const vts::TileRange &tr
                      , const vts::Mesh &mesh
                      , const vts::opencv::Atlas &atlas
                      , vts::TileIndex::Flag::value_type tileFlags);

    const Config &config_;
    const vr::ReferenceFrame &rf_;
    tools::TmpTileset &tmpset_;
    vts::NtGenerator &ntg_;
    const tdt::Archive &archive_;
    const tdt::Tileset &tileset_;

    const vts::NodeInfo::list nodes_;
};

struct TileInfo {
    const tdt::Tile *tile;
    const tdt::TilePath path;

    TileInfo(const tdt::Tile *tile, const tdt::TilePath &path)
        : tile(tile), path(path)
    {}

    std::string makePath() const;

    using list = std::vector<TileInfo>;
};

std::string TileInfo::makePath() const
{
    std::ostringstream os;
    os << tile->content->uri << "#" << path;
    return os.str();
}

tools::LodInfo analyze(vt::ExternalProgress &progress
                       , const Config &config
                       , const vts::NodeInfo::list &nodes
                       , const tdt::Archive &archive)
{
    LOG(info3) << "Analyzing input dataset (" << archive.treeSize()
               << " 3D Tiles).";

    const auto &root(*archive.tileset().root);
    if (!root.refine || (root.refine != tdt::Refinement::replace)) {
        LOGTHROW(err2, std::runtime_error)
            << "Only <" << tdt::Refinement::replace
            << "> refinement is supported.";
    }

    auto lodInfo(tools::LodInfo::invalid());

    traverse(root, [&](const tdt::Tile &tile, const tdt::TilePath &path)
             -> void
    {
        LOG(info4) << "Processing tile <" << path << ">";

        if (!tile.content) { return; }

        if (!ba::iends_with(tile.content->uri, ".b3dm")) {
            LOGTHROW(err2, std::runtime_error)
                << "Unsupported file content type: <"
                << tile.content->uri << "> in tile <" << path << ">.";
        }

        const auto depth(path.depth());

        if (tile.children.empty()) {
            // leaf
            lodInfo.commonBottom
                = std::min(lodInfo.commonBottom, depth);
            lodInfo.bottomDepth
                = std::max(lodInfo.bottomDepth, depth);
        }

        // update top
        lodInfo.topDepth = std::min(lodInfo.topDepth, depth);
    });

    LOG(info2) << "Found top/common-bottom/bottom: "
               << lodInfo.topDepth << "/" << lodInfo.commonBottom
               << "/" << lodInfo.bottomDepth << ".";

    // collect info for OpenMP
    TileInfo::list tiles;
    traverse(root, [&](const tdt::Tile &tile, const tdt::TilePath &path)
             -> void
    {
        if ((path.depth() == lodInfo.commonBottom) && tile.content) {
            tiles.emplace_back(&tile, path);
        }
    });
    progress.expect(tiles.size());

    tools::MeshInfo::map mim;

    UTILITY_OMP(parallel for shared(tiles, mim) schedule(dynamic))
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        const auto &ti(tiles[i]);
        const auto &tile(*ti.tile);
        const auto path(ti.makePath());

        VtsMeshLoader<SizeOnlyAtlas> loader(path);
        archive.loadMesh(loader, tile.content->uri);

        auto m(loader.get());
        const auto &mesh(m.first);
        const auto &atlas(m.second);

        // compute mesh area in each RF node
        for (const auto &rfNode : nodes) {
            const vts::CsConvertor conv(config.inputSrs, rfNode.srs());
            if (const auto mi = tools::measureMesh
                (rfNode, conv, mesh, atlas.get()))
            {
                UTILITY_OMP(critical(slpk2vts_meshInfo_1))
                    mim[&rfNode] += mi;
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

void Cutter::run(vt::ExternalProgress &progress)
{
    (void) progress;

    // analyze first
    const auto lodInfo(analyze(progress, config_, nodes_, archive_));

#if 0
    // analyze first
    const auto lodInfo(analyze(progress, config_, nodes_, tlsTiles, archive_));

    // compute navtile information (adds accumulators)
    for (const auto &item : lodInfo.localLods) {
        tools::computeNavtileInfo(*item.first, item.second, lodInfo, ntg_
                                  , config_.tileExtents
                                  , config_.ntLodPixelSize);
    }

    // update progress
    progress.expect(tlsTiles.size());


    const std::size_t nlSize(tlsTiles.size());
    UTILITY_OMP(parallel for schedule(dynamic))
    for (std::size_t i = 0; i < nlSize; ++i) {
        const auto &node(*tlsTiles[i]);
        cutNode(node, lodInfo);
        ++progress;
    }

#endif
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
            , const boost::optional<tdt::Archive> &input)
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

int Tdt2Vts::run()
{
    vts::TileSetProperties properties;
    properties.referenceFrame = config_.referenceFrame;
    properties.id = config_.tilesetId;

    // open 3D Tiles archive, let it recusively load tileset
    boost::optional<tdt::Archive> input;
    if (!config_.resume) {
        input = boost::in_place(input_, "", true);
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
    return Tdt2Vts()(argc, argv);
}
