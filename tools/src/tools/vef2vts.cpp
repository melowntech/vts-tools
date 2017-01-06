#include <cstdlib>
#include <string>
#include <iostream>
#include <algorithm>
#include <iterator>

#include <boost/algorithm/string/split.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "dbglog/dbglog.hpp"
#include "utility/streams.hpp"

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/progress.hpp"
#include "utility/streams.hpp"
#include "utility/openmp.hpp"
#include "utility/progress.hpp"

#include "service/cmdline.hpp"

#include "math/transform.hpp"
#include "math/filters.hpp"

#include "imgproc/scanconversion.hpp"
#include "imgproc/jpeg.hpp"

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

#include "vef/vef.hpp"


namespace po = boost::program_options;
namespace vs = vadstena::storage;
namespace vr = vadstena::registry;
namespace vts = vadstena::vts;
namespace vef = vadstena::vef;
namespace ba = boost::algorithm;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;

namespace {

struct Config {
    std::string tilesetId;
    std::string referenceFrame;
    vs::CreditIds credits;
    int textureQuality;

    bool forceWatertight;
    boost::optional<vts::LodTileRange> tileExtents;
    double clipMargin;
    double borderClipMargin;

    Config()
        : textureQuality(85), forceWatertight(false)
        , clipMargin(1.0 / 128.), borderClipMargin(1.0 / 128.)
    {}

    double maxClipMargin() const {
        // no tile extents: use only clip margin
        if (!tileExtents) { return clipMargin; }
        // use maximum of clipe extents and border clip extents
        return std::max(clipMargin, borderClipMargin);
    }
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

double triangleArea(const math::Point2 &a, const math::Point2 &b,
                    const math::Point2 &c)
{
    return std::abs
        (math::crossProduct(math::Point2(b - a), math::Point2(c - a)))
        / 2.0;
}

double bestTileArea(const math::Points2 &corners)
{
    return (triangleArea(corners[0], corners[1], corners[2])
            + triangleArea(corners[2], corners[3], corners[0]));
}

int bestLod(const vr::ReferenceFrame::Division::Node &node, double area)
{
    // compute longest of base node tile sizes
    auto rootSize(math::size(node.extents));
    auto rootArea(rootSize.width * rootSize.height);

    // compute number of requested tiles per edge
    auto tileCount(std::sqrt(rootArea / area));

    return int(std::round(std::log2(tileCount)));
}

vts::TileRange::point_type
tiled(const math::Size2f &ts, const math::Point2 &origin
      , const math::Point2 &p)
{
    math::Point2 local(p - origin);
    return vts::TileRange::point_type(local(0) / ts.width
                                      , -local(1) / ts.height);
}

vts::TileRange tileRange(const vr::ReferenceFrame::Division::Node &node
                         , vts::Lod localLod, const math::Points2 &points
                         , double margin)
{
    const auto ts(vts::tileSize(node.extents, localLod));
    // NB: origin is in upper-left corner and Y grows down
    const auto origin(math::ul(node.extents));

    math::Size2f isize(ts.width * margin, ts.height * margin);
    std::array<math::Point2, 4> inflates{{
            { -isize.width, +isize.height }
            , { +isize.width, +isize.height }
            , { +isize.width, -isize.height }
            , { -isize.width, -isize.height }
        }};

    vts::TileRange r(math::InvalidExtents{});

    for (const auto &p : points) {
        for (const auto &inflate : inflates) {
            update(r, tiled(ts, origin, p + inflate));
        }
    }

    return r;
}
template <typename Op>
void forEachTile(const vr::ReferenceFrame &referenceFrame
                 , vts::Lod lod, const vts::TileRange &tileRange
                 , Op op)
{
    typedef vts::TileRange::value_type Index;
    for (Index j(tileRange.ll(1)), je(tileRange.ur(1)); j <= je; ++j) {
        for (Index i(tileRange.ll(0)), ie(tileRange.ur(0)); i <= ie; ++i) {
            op(vts::NodeInfo(referenceFrame, vts::TileId(lod, i, j)));
        }
    }
}

template <typename Op>
void rasterizeTiles(const vr::ReferenceFrame &referenceFrame
                    , const vr::ReferenceFrame::Division::Node &rootNode
                    , vts::Lod lod, const vts::TileRange &tileRange
                    , Op op)
{
    // process tile range
    forEachTile(referenceFrame, lod, tileRange
                , [&](const vts::NodeInfo &ni)
    {
        LOG(info1)
            << std::fixed << "dst tile: "
            << ni.nodeId() << ", " << ni.extents();

        // TODO: check for incidence with Q; NB: clip margin must be taken into
        // account

        // check for root
        if (ni.subtree().root().id == rootNode.id) {
            op(vts::tileId(ni.nodeId()));
        }
    });
}

typedef std::map<vts::TileId, vts::TileId::list> SourceInfo;

math::Points2 projectCorners(const vr::ReferenceFrame::Division::Node &node
                             , const vts::CsConvertor &conv
                             , const math::Points2 &src)
{
    math::Points2 dst;
    try {
        for (const auto &c : src) {
            dst.push_back(conv(c));
            LOG(info1) << std::fixed << "corner: "
                       << c << " -> " << dst.back();
            if (!inside(node.extents, dst.back())) {
                // projected dst tile cannot fit inside this node's
                // extents -> ignore
                return {};
            }
        }
    } catch (std::exception) {
        // whole tile cannot be projected -> ignore
        return {};
    }

    // OK, we could convert whole tile into this reference system
    return dst;
}

class SourceInfoBuilder : boost::noncopyable {
public:
    SourceInfoBuilder(const vef::VadstenaArchive &archive
                      , const vr::ReferenceFrame &rf
                      , double margin)
    {
        (void) archive;
        (void) rf;
        (void) margin;

#if 0
        const auto &srcRf(tileset.referenceFrame());
        utility::Progress progress(tileset.tileIndex().count());

        traverse(tileset.tileIndex()
                 , [&](const vts::TileId &srcId, vts::QTree::value_type flags)
        {
            (++progress).report
                (utility::Progress::ratio_t(5, 1000)
                 , "building tile mapping ");
            if (!vts::TileIndex::Flag::isReal(flags)) { return; }

            vts::NodeInfo ni(srcRf, srcId);
            if (!ni.valid()) { return; }

            const auto &srcExtents(ni.extents());
            const math::Points2 srcCorners = {
                ul(srcExtents), ur(srcExtents), lr(srcExtents), ll(srcExtents)
            };

            // for each destination node
            for (const auto &item : dstRf.division.nodes) {
                const auto &node(item.second);
                if (!node.valid()) { continue; }
                const vts::CsConvertor csconv(ni.srs(), node.srs);

                auto dstCorners(projectCorners(node, csconv, srcCorners));

                // ignore tiles that cannot be transformed
                if (dstCorners.empty()) { continue; }

                // find best tile size
                auto bta(bestTileArea(dstCorners));

                // find such the closest tile to the best tile size
                auto dstLocalLod(bestLod(node, bta));
                auto dstLod(node.id.lod + dstLocalLod);

                LOG(info1)
                    << "Best tile area: " << bta << " -> LOD: " << dstLod
                    << " (node's local LOD: " << dstLocalLod << ").";

                // generate tile range from corners
                auto tr(tileRange(node, dstLocalLod, dstCorners, margin));
                LOG(info1) << "tile range: " << tr;

                // TODO: add margin
                rasterizeTiles(dstRf, node, dstLod, tr
                               , [&](const vts::TileId &id)
                {
                    sourceInfo_[id].push_back(srcId);
                    dstTi_.set(id, 1);
                });
            }
        });

        // clone dst tile index to valid tree and make it complete
        validTree_ = vts::TileIndex
            (vts::LodRange(0, dstTi_.maxLod()), &dstTi_);
        validTree_.complete();
#endif
    }

    const vts::TileIndex* validTree() const { return &validTree_; }

    const vts::TileId::list& source(const vts::TileId &tileId) const {
        auto fsourceInfo(sourceInfo_.find(tileId));
        if (fsourceInfo == sourceInfo_.end()) { return emptySource_; }
        return fsourceInfo->second;
    }

    std::size_t size() const { return sourceInfo_.size(); }

private:
    SourceInfo sourceInfo_;
    vts::TileIndex dstTi_;
    vts::TileIndex validTree_;

    static const vts::TileId::list emptySource_;
};

// keep empty, used as placeholder!
const vts::TileId::list SourceInfoBuilder::emptySource_;

class Encoder : public vts::Encoder {
public:
    Encoder(const boost::filesystem::path &path
            , const vts::TileSetProperties &properties
            , vts::CreateMode mode
            , const vef::VadstenaArchive &input
            , const Config &config)
        : vts::Encoder(path, properties, mode)
        , config_(config), input_(input)
        , srcInfo_(input_, referenceFrame(), config.maxClipMargin())
        , inputSrs_(*input.manifest().srs)
    {
        setConstraints(Constraints().setValidTree(srcInfo_.validTree()));
        setEstimatedTileCount(srcInfo_.size());
    }

private:
    virtual TileResult
    generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
             , const TileResult&)
        UTILITY_OVERRIDE;

    virtual void finish(vts::TileSet&);

    const Config config_;

    const vef::VadstenaArchive &input_;

    SourceInfoBuilder srcInfo_;

    const geo::SrsDefinition inputSrs_;
};

void warpInPlace(const vts::CsConvertor &conv, vts::SubMesh &sm)
{
    // just convert vertices
    for (auto &v : sm.vertices) {
        // convert vertex in-place
        v = conv(v);
    }
}

vts::VertexMask warpInPlaceWithMask(const vts::CsConvertor &conv
                                    , vts::SubMesh &sm)
{
    vts::VertexMask mask(sm.vertices.size(), true);

    std::size_t masked(0);
    auto imask(mask.begin());
    for (auto &v : sm.vertices) {
        try {
            // convert vertex in-place
            v = conv(v);
        } catch (std::exception) {
            // cannot convert vertex -> mask out
            *imask = false;
            ++masked;
        }
        ++imask;
    }

    // nothing masked -> no mask
    if (!masked) { return {}; }

    // something masked
    return mask;
}

math::Size2 navpaneSizeInPixels(const math::Size2 &sizeInTiles)
{
    // NB: navtile is in grid system, border pixels are shared between adjacent
    // tiles
    auto s(vts::NavTile::size());
    return { 1 + sizeInTiles.width * (s.width - 1)
            , 1 + sizeInTiles.height * (s.height - 1) };
}

vts::NavTile::pointer
warpNavtiles(const vts::TileId &tileId
             , const vr::ReferenceFrame &referenceFrame
             , const vts::NodeInfo &nodeInfo
             , const vts::MeshOpInput::list &source)
{
    // TODO: Check for different lodding/SDS and process accordingly

    vts::HeightMap hm(tileId, source, referenceFrame);
    if (hm.empty()) { return {}; }
    hm.warp(nodeInfo);

    auto navtile(hm.navtile(tileId));
    if (navtile->coverageMask().empty()) { return {}; }

    return navtile;
}

Encoder::TileResult
Encoder::generate(const vts::TileId &tileId, const vts::NodeInfo &nodeInfo
                  , const TileResult&)
{
    vts::BorderCondition borderCondition;

    if (config_.tileExtents) {
        borderCondition = vts::inside(*config_.tileExtents, tileId);
        if (!borderCondition) {
            // outside of range
            return TileResult::Result::noDataYet;
        }
    }

    const auto &src(srcInfo_.source(tileId));
    if (src.empty()) {
        return TileResult::Result::noDataYet;
    }

    LOG(info1) << "Source tiles(" << src.size() << "): "
               << utility::join(src, ", ") << ".";

    vts::MeshOpInput::list source;
#if 0
    {
        vts::MeshOpInput::Id id(0);
        for (const auto &srcId : src) {
            UTILITY_OMP(critical)
            {
                // build input for tile transformation:
                //     * node info is generated on the fly
                //     * this cannot be a lazy operation
                vts::MeshOpInput t(id++, input_, srcId, nullptr, false);
                if (t) { source.push_back(t); }
            }
        }
    }
#endif

    // CS convertors
    // src physical -> dst SDS
    const vts::CsConvertor srcPhy2Sds(inputSrs_, nodeInfo.srs());

    // dst SDS -> dst physical
    const vts::CsConvertor sds2DstPhy
        (nodeInfo.srs(), referenceFrame().model.physicalSrs);

    auto clipExtents(vts::inflateTileExtents
                     (nodeInfo.extents(), config_.clipMargin
                      , borderCondition, config_.borderClipMargin));

    // output
    Encoder::TileResult result;
    auto &tile(result.tile());
    vts::Mesh &mesh
        (*(tile.mesh = std::make_shared<vts::Mesh>(config_.forceWatertight)));
    vts::RawAtlas::pointer patlas([&]() -> vts::RawAtlas::pointer
    {
        auto atlas(std::make_shared<vts::RawAtlas>());
        tile.atlas = atlas;
        return atlas;
    }());
    auto &atlas(*patlas);

    for (const auto &input : source) {
        const auto &inMesh(input.mesh());
        for (std::size_t smIndex(0), esmIndex(inMesh.size());
             smIndex != esmIndex; ++smIndex)
        {
            auto sm(inMesh[smIndex]);

            auto mask(warpInPlaceWithMask(srcPhy2Sds, sm));

            // clip submesh
            auto dstSm(vts::clip(sm, clipExtents, mask));

            if (!dstSm.empty()) {
                // re-generate external tx coordinates (if division node allows)
                generateEtc(dstSm, nodeInfo.extents()
                            , nodeInfo.node().externalTexture);

                // update mesh coverage mask
                if (!config_.forceWatertight) {
                    updateCoverage(mesh, dstSm, nodeInfo.extents());
                }

                // convert mesh to destination physical SRS
                warpInPlace(sds2DstPhy, dstSm);

                // add mesh
                mesh.add(dstSm);

                // copy texture if submesh has atlas
                if (input.hasAtlas() && input.atlas().valid(smIndex)) {
                    atlas.add(input.atlas().get(smIndex));
                }

                // update credits
                const auto &credits(input.node().credits());
                tile.credits.insert(credits.begin(), credits.end());
            }
        }
    }

    if (mesh.empty()) {
        // no mesh
        // decrement number of estimated tiles
        updateEstimatedTileCount(-1);
        // tell that there is nothing yet
        return TileResult::Result::noDataYet;
    }

    // merge submeshes if allowed
    std::tie(tile.mesh, tile.atlas)
        = mergeSubmeshes(tileId, tile.mesh, patlas, config_.textureQuality);

    if (atlas.empty()) {
        // no atlas -> disable
        tile.atlas.reset();
    }

    // done:
    return result;
}

void Encoder::finish(vts::TileSet &ts)
{
    (void) ts;
    // TODO: update position and navtiles
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
