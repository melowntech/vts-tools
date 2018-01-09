#include <cstdlib>
#include <string>

#include <tinyxml2.h>

#include <opencv2/highgui/highgui.hpp>

#include "dbglog/dbglog.hpp"

#include "math/transform.hpp"

#include "geometry/mesh.hpp"
#include "geometry/meshop.hpp"

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/openmp.hpp"
#include "utility/progress.hpp"
#include "utility/limits.hpp"

#include "service/cmdline.hpp"
#include "service/verbosity.hpp"

#include "imgproc/imagesize.hpp"

#include "roarchive/roarchive.hpp"
#include "lodtree/lodtreefile.hpp"

#include "vts-libs/vts.hpp"
#include "vts-libs/vts/encoder.hpp"
#include "vts-libs/vts/meshop.hpp"
#include "vts-libs/vts/csconvertor.hpp"
#include "vts-libs/registry/po.hpp"
#include "vts-libs/vts/ntgenerator.hpp"
#include "vts-libs/vts/opencv/atlas.hpp"
#include "vts-libs/vts/io.hpp"
#include "vts-libs/tools-support/progress.hpp"
#include "vts-libs/tools-support/assimp.hpp"

#include "vts-libs/tools-support/tmptsencoder.hpp"
#include "vts-libs/tools-support/repackatlas.hpp"
#include "vts-libs/tools-support/analyze.hpp"

namespace vs = vtslibs::storage;
namespace vr = vtslibs::registry;
namespace vts = vtslibs::vts;
namespace vt = vtslibs::tools;
namespace tools = vtslibs::vts::tools;
namespace fs = boost::filesystem;
namespace po = boost::program_options;
namespace ublas = boost::numeric::ublas;

typedef vts::opencv::HybridAtlas HybridAtlas;

namespace {

struct Config : tools::TmpTsEncoder::Config {
    std::string tilesetId;
    std::string referenceFrame;
    math::Size2 optimalTextureSize;
    double ntLodPixelSize;

    boost::optional<vts::LodTileRange> tileExtents;
    double clipMargin;
    double borderClipMargin;
    double offsetX, offsetY, offsetZ;
    double zShift;

    Config()
        : optimalTextureSize(256, 256)
        , ntLodPixelSize(1.0)
        , clipMargin(1.0 / 128.)
        , borderClipMargin(clipMargin)
        , offsetX(), offsetY(), offsetZ()
        , zShift()
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

            ("offsetX"
             , po::value(&offsetX)->default_value(offsetX),
             "Force X shift of the model. "
             "NB: shift is performed in model's SRS.")
            ("offsetY"
             , po::value(&offsetY)->default_value(offsetY),
             "Force Y shift of the model. "
             "NB: shift is performed in model's SRS.")
            ("offsetZ"
             , po::value(&offsetZ)->default_value(offsetZ),
             "Force Z shift of the model. "
             "NB: shift is performed in model's SRS.")

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

class LodTree2Vts : public service::Cmdline
{
public:
    LodTree2Vts()
        : service::Cmdline("lodtree2vts", BUILD_TARGET_VERSION)
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

void LodTree2Vts::configuration(po::options_description &cmdline
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
         , "Path to input LODTree archive.")
        ("overwrite", "Existing tile set gets overwritten if set.")
        ;

    vt::progressConfiguration(config);

    pd
        .add("input", 1)
        .add("output", 1);

    (void) config;
}

void LodTree2Vts::configure(const po::variables_map &vars)
{
    vr::registryConfigure(vars);

    config_.configure(vars);

    createMode_ = (vars.count("overwrite")
                   ? vts::CreateMode::overwrite
                   : vts::CreateMode::failIfExists);

    epConfig_ = vt::configureProgress(vars);
}

bool LodTree2Vts::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(lodtree2vts
usage
    lodtree2vts INPUT OUTPUT [OPTIONS]

)RAW";
    }
    return false;
}

// ------------------------------------------------------------------------

tools::MeshInfo measureMesh(const vts::NodeInfo &rfNode
                            , const vts::CsConvertor conv
                            , const vts::Mesh &mesh
                            , const std::vector<math::Size2> &sizes)
{
    tools::MeshInfo mi;

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

tools::LodInfo analyze(vt::ExternalProgress &progress
                       , const Config &config
                       , const vts::NodeInfo::list &nodes
                       , const lodtree::Node::list &ltNodes
                       , const lodtree::LodTreeExport &archive)
{
    LOG(info3) << "Analyzing input dataset (" << ltNodes.size()
               << " LODTree nodes).";
    progress.expect(ltNodes.size());

    const geo::SrsDefinition inputSrs(archive.srs);

    // find limits for data nodes: top/bottom and bottom common to all subtrees
    tools::LodInfo lodInfo;
    lodInfo.topDepth = std::numeric_limits<int>::max();
    lodInfo.commonBottom = std::numeric_limits<int>::max();
    lodInfo.bottomDepth = -1;

    {
        for (const auto &node : ltNodes) {
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
    std::vector<const lodtree::Node*> treeNodes;
    for (const auto &node : ltNodes) {
        if (node.level == lodInfo.commonBottom) {
            treeNodes.push_back(&node);
        }
    }

    auto *pmim(&mim);
    const auto *pnodes(&treeNodes);

    UTILITY_OMP(parallel for shared(pmim) schedule(dynamic))
    for (std::size_t i = 0; i < pnodes->size(); ++i) {
        const auto &node(*(*pnodes)[i]);

        Assimp::Importer imp;
        imp.SetPropertyBool(AI_CONFIG_IMPORT_NO_SKELETON_MESHES, true);

        // load geometry
        vts::Mesh mesh;
        std::vector<math::Size2> sizes;

        {
            tools::TextureStreams ts;
            std::tie(mesh, ts) = tools::loadAssimpScene
                (imp, archive.archive(), node.modelPath, node.origin);

            // measure textures
            for (const auto &is : ts) {
                sizes.push_back(imgproc::imageSize(*is, is->path()));
            }
        }

        // compute mesh are in each RF node
        for (const auto &rfNode : nodes) {
            const vts::CsConvertor conv(inputSrs, rfNode.srs());
            const auto mi(measureMesh(rfNode, conv, mesh, sizes));
            if (mi) {
                UTILITY_OMP(critical(lodtree2vts_meshInfo_1))
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
        LOG(info2) << "Assigned LOD " << lod << " for bottom in subtree <"
                   << item.first->srs() << ">.";
    }

    return lodInfo;
}

// ------------------------------------------------------------------------

class Cutter {
public:
    Cutter(const Config &config, const vr::ReferenceFrame &rf
           , tools::TmpTileset &tmpset, vts::NtGenerator &ntg
           , const lodtree::LodTreeExport &archive)
        : config_(config), rf_(rf), tmpset_(tmpset), ntg_(ntg)
        , archive_(archive), nodes_(vts::NodeInfo::leaves(rf_))
        , inputSrs_(archive_.srs)
    {}

    void run(vt::ExternalProgress &progress);

private:
    void cutNode(const lodtree::LodTreeNode &node
                 , const tools::LodInfo &lodInfo);

    void splitToTiles(const lodtree::Node &ltNode
                      , const vts::NodeInfo &root
                      , vts::Lod lod, const vts::TileRange &tr
                      , const vts::Mesh &mesh
                      , const vts::opencv::Atlas &atlas
                      , vts::TileIndex::Flag::value_type tileFlags);

    void cutTile(const lodtree::Node &ltNode, const vts::NodeInfo &node
                 , const vts::Mesh &mesh
                 , const vts::opencv::Atlas &atlas
                 , vts::TileIndex::Flag::value_type tileFlags);

    const Config &config_;
    const vr::ReferenceFrame &rf_;
    tools::TmpTileset &tmpset_;
    vts::NtGenerator &ntg_;
    const lodtree::LodTreeExport &archive_;

    const vts::NodeInfo::list nodes_;
    const geo::SrsDefinition inputSrs_;
};

void Cutter::run(vt::ExternalProgress &progress)
{
    // load all available nodes
    const auto ltNodes(archive_.nodes());

    // analyze first
    const auto lodInfo(analyze(progress, config_, nodes_, ltNodes, archive_));

    // compute navtile information (adds accumulators)
    for (const auto &item : lodInfo.localLods) {
        tools::computeNavtileInfo(*item.first, item.second, lodInfo, ntg_
                                  , config_.tileExtents
                                  , config_.ntLodPixelSize);
    }

    // update progress
    progress.expect(ltNodes.size());

    // convert node map to node (pointer) list (needed by OpenMP to iterate over
    // nodes)
    auto nl([&]() -> std::vector<const lodtree::Node*>
    {
        std::vector<const lodtree::Node*> nl;
        nl.reserve(ltNodes.size());
        for (const auto &node : ltNodes) {
            nl.push_back(&node);
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

void Cutter::cutNode(const lodtree::Node &node, const tools::LodInfo &lodInfo)
{
    const auto &roArchive(archive_.archive());

    Assimp::Importer imp;
    imp.SetPropertyBool(AI_CONFIG_IMPORT_NO_SKELETON_MESHES, true);

    // load geometry
    vts::Mesh inMesh;
    vts::opencv::Atlas inAtlas;

    {
        tools::TextureStreams ts;
        std::tie(inMesh, ts) = tools::loadAssimpScene
            (imp, roArchive, node.modelPath, node.origin);

        // load textures
        for (const auto &is : ts) {
            LOG(info1) << "Loading texture from " << is->path() << ".";
            auto tex(cv::imdecode(is->read(), CV_LOAD_IMAGE_COLOR));
            inAtlas.add(tex);
        }
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

        const vts::Lod localLod(bottomLod - fromBottom);
        const auto lod(localLod + rfNode.nodeId().lod);

        // projested mesh/atlas
        vts::Mesh mesh;
        vts::opencv::Atlas atlas;

        // and for each submesh
        std::size_t meshIndex(0);
        for (auto &sm : inMesh) {
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
            vts::FaceOriginList faceOrigin;
            auto osm(vts::clip(sm, projected, rfNode.extents(), valid
                               , &faceOrigin));
            if (osm.faces.empty()) { continue; }

            // at least one face survived, remember

            mesh.submeshes.push_back(std::move(osm));
            atlas.add(texture);
        }

        // anything there?
        if (mesh.empty()) {
            continue;
        }

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
        LOG(info3) << "Splitting LODTree node " << node.modelPath
                   << " to tiles in "
                   << lod << "/" << tr << ".";
        splitToTiles(node, rfNode, lod, tr, mesh, atlas, tileFlags);
    }
}

void Cutter::splitToTiles(const lodtree::Node &ltNode
                          , const vts::NodeInfo &root
                          , vts::Lod lod, const vts::TileRange &tr
                          , const vts::Mesh &mesh
                          , const vts::opencv::Atlas &atlas
                          , vts::TileIndex::Flag::value_type tileFlags)
{
    if (config_.tileExtents) {
        // check for range validity
        const auto &rootId(root.nodeId());
        if (lod < config_.tileExtents->lod) {
            LOG(info2)
                << "Nothing to cut from LODTree node "
                << ltNode.modelPath << ".";
        }

        const auto gtr(vts::global(rootId, lod, tr));
        const auto extents(vts::shiftRange(*config_.tileExtents, lod));

        if (!vts::tileRangesOverlap(gtr, extents)) {
            LOG(info2)
                << "Nothing to cut from LODTree node " << ltNode.modelPath
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
            cutTile(ltNode, node, mesh, atlas, tileFlags);
        }
    }
}

void Cutter::cutTile(const lodtree::Node &ltNode, const vts::NodeInfo &node
                     , const vts::Mesh &mesh
                     , const vts::opencv::Atlas &atlas
                     , vts::TileIndex::Flag::value_type tileFlags)
{

    // compute border condition (defaults to all available)
    vts::BorderCondition borderCondition;
    if (config_.tileExtents) {
        borderCondition = vts::inside(*config_.tileExtents, node.nodeId());
        if (!borderCondition) {
            LOG(info1)
                << node.nodeId() << ": Nothing to cut from LODTree node "
                << ltNode.modelPath << ".";
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
            << node.nodeId() << ": Nothing cut from LODTree node "
            << ltNode.modelPath << ".";
        return;
    }

    LOG(info2)
        << node.nodeId() << ": Cut " << faces
        << " faces from LODTree node " << ltNode.modelPath << ".";

    // store in temporary storage
    const auto tileId(node.nodeId());

    tools::repack(tileId, clipped, clippedAtlas);
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
            , const boost::optional<lodtree::LodTreeExport> &input)
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

int LodTree2Vts::run()
{
    vts::TileSetProperties properties;
    properties.referenceFrame = config_.referenceFrame;
    properties.id = config_.tilesetId;

    // open input if in non-resume mode
    boost::optional<lodtree::LodTreeExport> input;
    if (!config_.resume) {
        math::Point3 offset(config_.offsetX, config_.offsetY, config_.offsetZ);
        if (norm_2(offset) > 0.) {
            LOG(info2) << "Using offset " << offset << ".";
        }

        // parse the XMLs
        input = boost::in_place(input_, offset);

        // TODO: sanity check
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
    return LodTree2Vts()(argc, argv);
}

