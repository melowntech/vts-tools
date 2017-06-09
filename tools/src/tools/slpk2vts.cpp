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

boost::optional<double> computeBestLod(const Config &config
                                       , const vts::NodeInfo &rfNode
                                       , const vts::CsConvertor conv
                                       , const vts::Mesh &mesh
                                       , const std::vector<math::Size2> &sizes)
{
    double meshArea(0.0);
    double textureArea(0.0);

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
        if (osm.faces.empty()) { return boost::none; }

        // at least one face survived remember

        // calculate area (only valid faces)
        const auto a(area(osm));
        textureArea += (a.internalTexture * math::area(size));
        meshArea += a.mesh;
    }

    // anything?
    if (!textureArea) { return boost::none; }

    const double texelArea(meshArea / textureArea);

    const auto optimalTileArea
        (area(config.optimalTextureSize) * texelArea);
    const auto optimalTileCount(rfNode.extents().area()
                                / optimalTileArea);
    return (0.5 * std::log2(optimalTileCount));
}

struct SrcNodeInfo {
    /** Node
     */
    const slpk::Node *node;
    double bestLod;

    SrcNodeInfo(const slpk::Node *node, double bestLod)
        : node(node), bestLod(bestLod)
    {}

    typedef std::map<const vts::NodeInfo*, SrcNodeInfo> map;
};

class Analyzer {
public:
    Analyzer(const Config &config
             , const vts::NodeInfo::list &nodes
             , const slpk::Tree &tree
             , const slpk::Archive &archive
             , SrcNodeInfo::map &nodeLodMapping)
        : config_(config)
        , nodes_(nodes), tree_(tree), archive_(archive)
        , inputSrs_(archive_.srs())
        , nodeLodMapping_(nodeLodMapping)
    {}

    void run(vt::ExternalProgress &progress);

private:
    void collect(const std::string &nodeId);

    const Config config_;
    const vts::NodeInfo::list &nodes_;
    const slpk::Tree &tree_;
    const slpk::Archive &archive_;
    const geo::SrsDefinition inputSrs_;

    SrcNodeInfo::map &nodeLodMapping_;
};

void Analyzer::run(vt::ExternalProgress &progress)
{
    progress.expect(tree_.nodes.size());

    collect(tree_.rootNodeId);

    // TODO: sigma-edit best lods at collected nodes
    // TODO: distribute lods through hierarchy
}

void Analyzer::collect(const std::string &nodeId)
{
    const auto *node(tree_.find(nodeId));
    if (!node) {
        LOG(warn3) << "Referenced node <" << nodeId << "> not found.";
    }

    LOG(info4) << "node: " << node->id << " level=" << node->level << ".";
    if (!node->children.empty()) {
        for (const auto &child : node->children) {
            collect(child.id);
        }
        return;
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

    // bottom level -> compute best lod in each node
    for (const auto &rfNode : nodes_) {
        const vts::CsConvertor conv(inputSrs_, rfNode.srs());

        auto bestLod(computeBestLod(config_, rfNode, conv
                                    , loader.mesh(), sizes));
        if (!bestLod) { continue; }
        LOG(info4) << "<" << rfNode.srs() << ">: " << nodeId << " "
                   << *bestLod;

        // TODO: remember best lod
    }
}

// ------------------------------------------------------------------------

class Cutter {
public:
    Cutter(const Config &config, const vr::ReferenceFrame &rf
           , tools::TmpTileset &tmpset, const slpk::Archive &archive)
        : config_(config), rf_(rf), tmpset_(tmpset), archive_(archive)
        , nodes_(vts::NodeInfo::leaves(rf_))
        , inputSrs_(archive_.srs())
    {}

    void run(vt::ExternalProgress &progress);

private:
    void cutNode(const slpk::Node &node);

    cv::Mat loadTexture(const slpk::Node &node, int index) const;

    const Config &config_;
    const vr::ReferenceFrame &rf_;
    tools::TmpTileset &tmpset_;
    const slpk::Archive &archive_;

    const vts::NodeInfo::list nodes_;
    const geo::SrsDefinition inputSrs_;

    SrcNodeInfo::map nodeLodMapping_;
};

void Cutter::run(vt::ExternalProgress &progress)
{
    // load all available nodes
    const auto tree(archive_.loadTree());

    // analyze first
    Analyzer(config_, nodes_, tree, archive_, nodeLodMapping_).run(progress);

    // update progress
    progress.expect(tree.nodes.size());

    for (const auto &ni : tree.nodes) {
        cutNode(ni.second);
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

void Cutter::cutNode(const slpk::Node &node)
{
    (void) node;

#if 0
    VtsMeshLoader loader;
    archive_.loadGeometry(loader, node);

    // for each submesh
    std::size_t meshIndex(0);
    for (auto &sm : loader.mesh()) {
        const auto texture(loadTexture(node, meshIndex));

        // and for each RF node
        for (const auto &rfNode : nodes_) {
            const vts::CsConvertor conv(inputSrs_, rfNode.srs());

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
            auto osm(vts::clip(sm, projected, rfNode.extents(), valid));
            if (osm.faces.empty()) { continue; }

            // at least one face survived, remember
            vts::Mesh mesh;
            vts::opencv::Atlas atlas;
            mesh.submeshes.push_back(std::move(osm));
            atlas.add(texture);

            // calculate area (only valid faces)
            const auto a(area(mesh));
            const double textureArea(a.submeshes.front().internalTexture
                                     * texture.cols * texture.rows);
            const double texelArea(a.mesh / textureArea);

            const auto optimalTileArea
                (area(config_.optimalTextureSize) * texelArea);
            const auto optimalTileCount(rfNode.extents().area()
                                        / optimalTileArea);
            const auto bestLod(0.5 * std::log2(optimalTileCount));

            LOG(info4) << rfNode.srs()
                       << ": id=" << node.id
                       << ", faces=" << osm.faces.size()
                       << ", textureArea=" << textureArea
                       << ", bestLod=" << bestLod;
        }

        ++meshIndex;
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

        Cutter(config_, referenceFrame(), tmpset(), *input).run(progress());
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
