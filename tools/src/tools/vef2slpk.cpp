/**
 * Copyright (c) 2017-2022 Melown Technologies SE
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
#include <ctime>
#include <string>
#include <iostream>
#include <algorithm>
#include <iterator>

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/utility/in_place_factory.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

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
#include "utility/stl-helpers.hpp"
#include "utility/config.hpp"

#include "utility/zip.hpp"

#include "service/cmdline.hpp"
#include "service/verbosity.hpp"

#include "math/transform.hpp"
#include "math/filters.hpp"
#include "math/math.hpp"

#include "imgproc/scanconversion.hpp"
#include "imgproc/jpeg.hpp"

#include "geometry/parse-obj.hpp"

#include "geo/csconvertor.hpp"
#include "geo/enu.hpp"

#include "vts-libs/vts/mesh.hpp"
#include "vts-libs/vts/meshop.hpp"
#include "vts-libs/vts/tileop.hpp"
#include "vts-libs/tools-support/progress.hpp"

#include "vef/reader.hpp"
#include "vef/tiling.hpp"
#include "vef/tilecutter.hpp"
#include "slpk/writer.hpp"
#include "miniball/miniball.hpp"

#include "vts-libs/tools-support/tmptileset.hpp"
#include "vts-libs/tools-support/repackatlas.hpp"


namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;
namespace vs = vtslibs::storage;
namespace vr = vtslibs::registry;
namespace vt = vtslibs::tools;
namespace vts = vtslibs::vts;
namespace tools = vtslibs::vts::tools;

namespace {

struct SrsOverride {
    enum class Source {
        default_
        , input
        , custom
    };

    std::string name;

    Source source = Source::default_;
    boost::optional<geo::SrsDefinition> srs;

    SrsOverride(const std::string &name) : name(name) {}

    void configuration(po::options_description &config);

    void configure(const po::variables_map &vars);

    std::ostream& dump(std::ostream &os, const std::string &prefix = {}) const;
};

void SrsOverride::configuration(po::options_description &config)
{
    config.add_options()
        (name.c_str(), po::value<std::string>()
         , "SRS override. Leave unset to use default handling, "
         "set to actual SRS to provide custom SRS and "
         "use special value \"input\" to use input SRS "
         "from VEF.");
}

void SrsOverride::configure(const po::variables_map &vars)
{
    if (!vars.count(name)) {
        source = Source::default_;
        srs = boost::none;
        return;
    }

    const auto &value(vars[name].as<std::string>());
    if (value == "input") {
        source = Source::input;
        srs = boost::none;
    } else {
        source = Source::custom;
        srs = value;
    }
}

std::ostream& SrsOverride::dump(std::ostream &os, const std::string &prefix)
    const
{
    os << prefix << name << " = " ;

    switch (source) {
    case Source::default_: os << "default"; break;
    case Source::input: os << "input"; break;
    case Source::custom: os << *srs; break;
    }

    return os;
}

struct Config {
    int textureQuality;
    math::Size2 optimalTextureSize;
    int binaryOrder = -1;
    std::string layerName;
    boost::optional<std::string> alias;
    boost::optional<std::string> copyrightText;

    bool fuseSubmeshes;
    vts::SubmeshMergeOptions smMergeOptions;
    double clipMargin;
    bool resume;
    bool keepTmpset;

    SrsOverride workSrs = SrsOverride("workSrs");
    SrsOverride dstSrs = SrsOverride("dstSrs");

    Config()
        : textureQuality(85), optimalTextureSize(256, 256)
        , fuseSubmeshes(true)
        , clipMargin(1.0 / 128.), resume(false), keepTmpset(false)
    {
        // set to inlimited, we want one submesh since multiple texture support
        // is missing
        smMergeOptions.maxVertexCount = smMergeOptions.maxFaceCount
            = std::numeric_limits<std::size_t>::max();
    }
};

class Vef2Slpk : public service::Cmdline
{
public:
    Vef2Slpk()
        : service::Cmdline("vef2slpk", BUILD_TARGET_VERSION)
    {}

private:
    void configuration(po::options_description &cmdline
                       , po::options_description &config
                       , po::positional_options_description &pd) override;

    void configure(const po::variables_map &vars) override;

    bool help(std::ostream &out, const std::string &what) const override;

    int run() override;

    fs::path output_;
    std::vector<fs::path> input_;

    fs::path tmptsPath_;

    bool overwrite_;
    Config config_;
    vt::ExternalProgress::Config epConfig_;
};

void Vef2Slpk::configuration(po::options_description &cmdline
                             , po::options_description &config
                             , po::positional_options_description &pd)
{
    cmdline.add_options()
        ("output", po::value(&output_)->required()
         , "Path to output SLPK file.")
        ("input", po::value(&input_)
         , "Path to input VEF archive(s).")
        ("tmptsPath", po::value(&tmptsPath_)
         , "Path to temporary tileset, defautls to output + \".tmpts\".")
        ("overwrite", "Existing tile set gets overwritten if set.")

        ("textureQuality", po::value(&config_.textureQuality)
         ->default_value(config_.textureQuality)->required()
         , "Texture quality for JPEG texture encoding (0-100).")

        ("clipMargin", po::value(&config_.clipMargin)
         ->default_value(config_.clipMargin)
         , "Margin (in fraction of tile dimensions) added to tile extents in "
         "all 4 directions.")

        ("optimalTextureSize", po::value(&config_.optimalTextureSize)
         ->default_value(config_.optimalTextureSize)->required()
         , "Size of ideal tile texture. Used to calculate fitting LOD from"
         "mesh texel size. Do not modify.")

        ("binaryOrder", po::value(&config_.binaryOrder)
         ->default_value(config_.binaryOrder)->required()
         , "Size of tilar archive (2^binary order * 2^binary order tiles). "
         "Value <0 uses tilar's built-in default.")

        ("fuseSubmeshes", po::value(&config_.fuseSubmeshes)
         ->default_value(config_.fuseSubmeshes)->required()
         , "Fuse submeshes into one bigger submesh. "
         "NB: multi texture bundling is not supported, yet.")

        ("resume"
         , "Resumes interrupted encoding. There must be complete (valid) "
         "temporary tileset inside generated tileset. Use with caution.")
        ("keepTmpset"
         , "Keep temporary tileset intact on exit.")

        ("layerName", po::value(&config_.layerName)
         , "SLPK layer name. Defaults to output path stem "
         "(filename without extentsion).")

        ("alias", po::value<std::string>()
         , "Optional display alias for generated SLPK layer.")

        ("copyrightText", po::value<std::string>()
         , "Optional copyright text for generated SLPK layer.")
        ;

    config_.workSrs.configuration(cmdline);
    config_.dstSrs.configuration(cmdline);

    vt::progressConfiguration(config);

    pd
        .add("output", 1)
        .add("input", -1)
        ;

    (void) config;
}

void Vef2Slpk::configure(const po::variables_map &vars)
{
    overwrite_ = vars.count("overwrite");
    config_.resume = vars.count("resume");
    config_.keepTmpset = vars.count("keepTmpset");
    epConfig_ = vt::configureProgress(vars);
    if (vars.count("alias")) {
        config_.alias = vars["alias"].as<std::string>();
    }
    if (vars.count("copyrightText")) {
        config_.copyrightText = vars["copyrightText"].as<std::string>();
    }

    config_.workSrs.configure(vars);
    config_.dstSrs.configure(vars);

    if (!vars.count("tmptsPath")) {
        tmptsPath_ = utility::addExtension(output_, ".tmpts");
    }

    if (!config_.resume && input_.empty()) {
        throw po::required_option("input");
    }

    if (config_.resume) {
        if (!input_.empty()) {
            LOG(warn2) << "NB: input ignored when resuming.";
        }
    }

    LOG(info3)
        << "Configuration:\n"
        << utility::dump(config_.workSrs, "\t") << "\n"
        << utility::dump(config_.dstSrs, "\t") << "\n"
        ;
}

bool Vef2Slpk::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(vef2vts
usage
    vef2vts INPUT OUTPUT [OPTIONS]

)RAW";
    }
    return false;
}

struct Setup {
    slpk::SpatialReference spatialReference;
    math::Extents2 workExtents;
    geo::SrsDefinition workSrs;
    geo::SrsDefinition dstSrs;
    geo::CsConvertor work2dst;
    vts::Lod maxLod;
    double resolution;

    Setup() : maxLod() {}

    void save(std::ostream &os) const;

    void configuration(const std::string&, po::options_description &od);
    void configure(const std::string&, po::variables_map) {}

    boost::optional<geo::SrsDefinition>
    cutSrs(const geo::SrsDefinition &srs) const;
};

boost::optional<geo::SrsDefinition>
Setup::cutSrs(const geo::SrsDefinition &srs) const
{
    if (geo::areSame(srs, workSrs)) { return boost::none; }
    return workSrs;
}

void Setup::save(std::ostream &os) const
{
    os << std::fixed << std::setprecision(15)
       << "workExtents = " << workExtents
       << "\nworkSrs = " << workSrs
       << "\ndstSrs = " << dstSrs
       << "\nmaxLod = " << maxLod
       << "\nresolution = " << resolution
       << "\n"
       << "\nspatialReference.wkid = " << spatialReference.wkid
       << "\nspatialReference.latestWkid = " << spatialReference.latestWkid
       << "\nspatialReference.vcsWkid = " << spatialReference.vcsWkid
       << "\nspatialReference.latestVcsWkid = "
       << spatialReference.latestVcsWkid
       << "\nspatialReference.wkt = " << spatialReference.wkt
       << "\n"
        ;
}
 
void Setup::configuration(const std::string&, po::options_description &od)
{
    od.add_options()
        ("workExtents", po::value(&workExtents)->required()
         , "workExtents")
        ("workSrs", po::value(&workSrs)->required()
         , "workSrs")
        ("dstSrs", po::value(&dstSrs)->required()
         , "dstSrs")
        ("maxLod", po::value(&maxLod)->required(), "maxLod")
        ("resolution", po::value(&resolution)->required(), "resolution")

        ("spatialReference.wkid"
         , po::value(&spatialReference.wkid)->required()
         , "nspatialReference.wkid")
        ("spatialReference.latestWkid",
         po::value(&spatialReference.latestWkid)->required()
         , "nspatialReference.latestWkid")
        ("spatialReference.vcsWkid"
         , po::value(&spatialReference.vcsWkid)->required()
         , "nspatialReference.vcsWkid")
        ("spatialReference.latestVcsWkid"
         , po::value(&spatialReference.latestVcsWkid)->required()
         , "nspatialReference.latestVcsWkid")
        ("spatialReference.wkt"
         , po::value(&spatialReference.wkt)->required()
         , "nspatialReference.wkt")
        ;
}

void writeSetup(const fs::path &file, const Setup &setup)
{
    LOG(info2) << "Saving setup to " << file;

    const auto tmp(utility::addExtension(file, ".tmp"));

    std::ofstream f;
    f.exceptions(std::ios::badbit | std::ios::failbit);
    f.open(tmp.native(), std::ios_base::out | std::ios_base::trunc);

    setup.save(f);
    f.close();
    fs::rename(tmp, file);
}

vef::Tiling makeTiling(const Config &config
                       , const vef::Archive::list &archives)
{
    const auto &inputSrs(archives.front().manifest().srs.value());

    switch (config.workSrs.source) {
    case SrsOverride::Source::default_:
        return vef::Tiling(archives, config.optimalTextureSize);

    case SrsOverride::Source::input:
        return vef::Tiling
            (archives, config.optimalTextureSize
             , false, boost::none
             , vef::World(inputSrs, math::Extents3(math::InvalidExtents{})));

    case SrsOverride::Source::custom:
        return vef::Tiling
            (archives, config.optimalTextureSize
             , false, boost::none
             , vef::World(inputSrs, math::Extents3(math::InvalidExtents{})));
    }

    throw; // never reached
}

Setup makeSetup(Config &config, const vef::Archive::list &archives)
{
    auto tiling(makeTiling(config, archives));

    const auto &inputSrs(archives.front().manifest().srs.value());

    Setup setup;
    setup.workExtents = math::extents2(tiling.workExtents);
    setup.workSrs = tiling.workSrs;
    setup.maxLod = tiling.maxLod;
    setup.resolution = tiling.resolution;

    // set output SRS
    switch (config.dstSrs.source) {
    case SrsOverride::Source::default_:
        setup.spatialReference = {};
        break;

    case SrsOverride::Source::input:
        setup.spatialReference.setSrs(inputSrs);
        break;

    case SrsOverride::Source::custom:
        setup.spatialReference.setSrs(config.dstSrs.srs.value());
        break;

    }

    setup.dstSrs = setup.spatialReference.srs();

    setup.work2dst = geo::CsConvertor(setup.workSrs, setup.dstSrs);
    return setup;
}

Setup readSetup(const fs::path &file)
{
    Setup setup;
    utility::readConfig(file, setup);

    setup.work2dst = geo::CsConvertor(setup.workSrs, setup.dstSrs);
    return setup;
}

struct NodeId {
    std::vector<int> path;

    NodeId() {}

    NodeId child(int which) const {
        NodeId nodeId(*this);
        nodeId.path.push_back(which);
        return nodeId;
    }

    int level() const { return path.size() + 1; }
};

typedef boost::optional<slpk::NodeReference> OptNodeReference;

template<typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits> &os, const NodeId &nodeId)
{
    return os << utility::join(nodeId.path, "-", "root");
}

inline std::string asString(const NodeId &nodeId) {
    return boost::lexical_cast<std::string>(nodeId);
}

void updateMbs(slpk::MinimumBoundingSphere &dst
               , slpk::MinimumBoundingSphere &src)
{
    if (!dst.r) {
        dst = src;
        return;
    }

    const auto cDist(norm_2(dst.center - src.center));
    // dst inside src: replace dst with src
    if ((cDist + dst.r) < src.r) {
        dst = src;
        return;
    }

    // src inside dst, keep
    if ((cDist + src.r) < dst.r) { return; }

    double r((dst.r + src.r + cDist) / 2.0);
    math::Point3 c(dst.center + (src.center - dst.center)
                   * (r - dst.r) / cDist);
    dst.r = r;
    dst.center = c;
}

class NodeHolder {
public:
    typedef std::shared_ptr<NodeHolder> pointer;
    typedef std::vector<NodeHolder::pointer> list;

    slpk::NodeReference reference;
    slpk::Node node;
    slpk::MinimumBoundingSphere srcMbs;
    slpk::SharedResource sharedResource;

    NodeHolder(const vts::TileId &tileId, const NodeId &nodeId)
        : tileId_(tileId)
    {
        node.id = asString(nodeId);
    }

    void child(const NodeHolder::pointer &child) {
        if (child) { children_.push_back(child); }
    }

    void finish(const Setup &setup)
    {
        // finish children first
        const bool emptyNode(!node.mbs.r);

        for (auto &child : children_) {
            child->finish(setup);
            if (emptyNode) {
                updateMbs(srcMbs, child->srcMbs);
            }
        }

        if (emptyNode) {
            auto mbs(srcMbs);
            mbs.center = setup.work2dst(mbs.center);
            reference.mbs = node.mbs = mbs;
        }

        // LOD selection
        {
            node.lodSelection.emplace_back();
            auto &ls(node.lodSelection.back());
            ls.metricType = slpk::MetricType::maxScreenThreshold;
            ls.maxError = ((2.0 * reference.mbs.r)
                           / (scale(setup) * setup.resolution));
        }

        // cross-reference children (N^2)
        for (auto &child : children_) {
            for (auto &other : children_) {
                if (child != other) {
                    child->node.neighbors.push_back(other->reference);
                }
            }
            child->node.parentNode = reference;
            node.children.push_back(child->reference);
        }
    }

    void write(slpk::Writer &writer) {
        if (node.geometryData.empty()) {
            writer.write(node);
        } else {
            writer.write(node, &sharedResource);
        }

        for (auto &child : children_) { child->write(writer); }
    }

private:
    const vts::TileId tileId_;

    bool bottom(const Setup &setup) const {
        return (tileId_.lod >= setup.maxLod);
    }

    double scale(const Setup &setup) const {
        if (bottom(setup)) { return 1.0; }
        return 1 << (setup.maxLod - tileId_.lod);
    }

    list children_;
};

std::string generateUuid() {
    // generate random uuid
    boost::mt19937 ran;
    // maybe use better initialization
    ran.seed(std::time(nullptr));
    return to_string(boost::uuids::basic_random_generator
                     <boost::mt19937>(&ran)());
}

slpk::SceneLayerInfo makeSceneLayerInfo(const Config &config
                                        , const Setup &setup)
{
    slpk::SceneLayerInfo sli;

    sli.id = 0;
    sli.href = "layers/0";
    sli.layerType = slpk::LayerType::integratedMesh;
    sli.spatialReference = setup.spatialReference;

    // spatialReference -> sli.heightModelInfo
    sli.heightModelInfo = slpk::HeightModelInfo(sli.spatialReference.srs());

    // TODO: generate VERSION
    sli.name = config.layerName;
    sli.alias = config.alias;
    sli.copyrightText = config.copyrightText;

    sli.capabilities.insert(slpk::Capability::view);
    sli.capabilities.insert(slpk::Capability::query);

    // store
    auto &store(*sli.store);

    store.id = generateUuid();
    store.profile = slpk::Profile::meshpyramids;
    store.resourcePattern = { slpk::ResourcePattern::nodeIndexDocument
                              , slpk::ResourcePattern::sharedResource
                              , slpk::ResourcePattern::geometry
                              , slpk::ResourcePattern::texture };
    store.rootNode = "./nodes/" + asString(NodeId());

    store.textureEncoding.emplace_back("image/jpeg");

    {
        auto &idx(store.indexingScheme);
        idx.name = slpk::IndexSchemeName::quadTree; // ???
        // we do not calculate accumulated extents -> false
        idx.inclusive = false;
        idx.dimensionality = 3;
        idx.childrenCardinality.max = 4;
        idx.neighborCardinality.max = 4;
    }

    // default geometry schema
    auto &dgs(*(store.defaultGeometrySchema = boost::in_place()));
    dgs.geometryType = slpk::GeometryType::triangles;
    dgs.topology = slpk::Topology::perAttributeArray;

    /* vertexAttributes */ {
        dgs.header.emplace_back("vertexCount", slpk::DataType::uint32);

        /* position */ {
            auto &position(utility::append(dgs.vertexAttributes, "position"));
            position.valueType = slpk::DataType::float32;
            position.valuesPerElement = 3;
        }

        /* uv0 */ {
            auto &uv0(utility::append(dgs.vertexAttributes, "uv0"));
            uv0.valueType = slpk::DataType::float32;
            uv0.valuesPerElement = 2;
        }

    }

    /* featureAttributes */ {
        dgs.header.emplace_back("featureCount", slpk::DataType::uint32);

        /* id */ {
            auto &id(utility::append(dgs.featureAttributes, "id"));
            id.valueType = slpk::DataType::uint64;
            id.valuesPerElement = 1;
        }

        /* faceRange */ {
            auto &faceRange
                (utility::append(dgs.featureAttributes, "faceRange"));
            faceRange.valueType = slpk::DataType::uint32;
            faceRange.valuesPerElement = 2;
        }
    }

    return sli;
}

class Generator {
public:
    Generator(slpk::Writer &writer, const tools::TmpTileset &ts
              , const Config &config, const Setup &setup)
        : writer_(writer), ts_(ts), config_(config), setup_(setup)
        , ti_(ts.tileIndex()), fullTree_(ti_)
        , sceneExtents_(math::InvalidExtents{})
        , generated_(), total_(ti_.count())
    {
        fullTree_.makeAbsolute().complete();
    }

    void operator()(/**vt::ExternalProgress &progress*/);

private:
    NodeHolder::pointer process(const vts::TileId &tileId, NodeId nodeId);

    slpk::Writer &writer_;

    const tools::TmpTileset &ts_;
    const Config &config_;
    const Setup &setup_;

    vts::TileIndex ti_;
    vts::TileIndex fullTree_;

    math::Extents2 sceneExtents_;

    std::atomic<std::size_t> generated_;
    std::size_t total_;
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

struct MeshSaver : slpk::MeshSaver {
    MeshSaver(const slpk::Node &node, const vts::SubMesh &sm)
        : node(node), sm(sm)
    {}

    virtual Properties properties() const {
        Properties p;
        p.faceCount = sm.faces.size();
        return p;
    }

    virtual math::Triangle3d face(std::size_t index) const {
        auto &f(sm.faces[index]);
        return {{ sm.vertices[f(0)], sm.vertices[f(1)], sm.vertices[f(2)] }};
    }

    virtual math::Triangle2d faceTc(std::size_t index) const {
        auto &f(sm.facesTc[index]);
        return {{ normalize(sm.tc[f(0)])
                  , normalize(sm.tc[f(1)])
                  , normalize(sm.tc[f(2)]) }};
    }

    math::Point2 normalize(const math::Point2 &p) const {
        return { p(0), 1.0 - p(1) };
    }

    const slpk::Node &node;
    const vts::SubMesh &sm;
};

struct TextureSaver : slpk::TextureSaver {
    TextureSaver(const vts::Atlas &atlas, std::size_t index)
        : atlas(atlas), index(index)
    {}

    virtual math::Size2 imageSize() const {
        return atlas.imageSize(index);
    }

    virtual void save(std::ostream &os, const std::string &mimeType) const {
        // TODO: maybe store in different formats
        (void) mimeType;
        atlas.write(os, index);
    }

    const vts::Atlas &atlas;
    std::size_t index;
};

void write(slpk::Writer &writer
           , slpk::Node &node, slpk::SharedResource &sharedResource
           , const vts::Mesh &mesh, const vts::Atlas &atlas)
{
    int smi(0);
    for (const auto &sm : mesh) {
        writer.write(node, sharedResource
                     , MeshSaver(node, sm), TextureSaver(atlas, smi));
        ++smi;
    }
}

inline void warpInPlace(vts::SubMesh &mesh, const geo::CsConvertor &conv)
{
    for (auto &v : mesh.vertices) { v = conv(v); }
}

inline void warpInPlace(vts::Mesh &mesh, const geo::CsConvertor &conv)
{
    for (auto &sm : mesh) { warpInPlace(sm, conv); }
}

math::Extents2 measureMesh(const vts::Mesh &mesh)
{
    math::Extents2 extents(math::InvalidExtents{});
    for (const auto &sm : mesh) {
        for (const auto &v : sm.vertices) {
            math::update(extents, v(0), v(1));
        }
    }
    return extents;
}


struct Done {
    Done(std::size_t count, std::size_t total)
        : count(count), total(total)
    {}

    std::size_t count;
    std::size_t total;
};

template<typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits> &os, const Done &d)
{
    if (d.total) {
        double percentage((100.0 * d.count) / d.total);
        boost::io::ios_precision_saver ps(os);
        return os << '#' << d.count << " of " << d.total << " ("
                  << std::fixed << std::setprecision(2)
                  << std::setw(6) << percentage
                  << " % done)";
    }
    return os << '#' << d.count;
}

NodeHolder::pointer
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

    if (!fullTree_.get(tileId)) { return {}; }

    TIDGuard tg(str(boost::format("tile:%s") % tileId));

    auto node(std::make_shared<NodeHolder>(tileId, nodeId));
    auto &nodeReference(node->reference);

    if (ti_.get(tileId)) {
        LOG(info2)
            << "Generating node <" << nodeId << "> from tile " << tileId
            << ".";

        // create new node
        auto &n(node->node);
        auto &sr(node->sharedResource);

        {
            auto &material
                (utility::append(sr.materialDefinitions, "TexturedMaterial"));
            material.name = "StandardMaterial";
        }

        n.level = nodeId.level();

        // TODO: build node version

        // build node reference
        nodeReference = n.reference();
        nodeReference.href = "../" + nodeReference.id;

        vts::Mesh::pointer mesh;
        vts::Atlas::pointer atlas;
        {
            const auto loaded(ts_.load(tileId, config_.textureQuality));
            if (config_.fuseSubmeshes) {
                // merge submeshes
                std::tie(mesh, atlas)
                    = vts::mergeSubmeshes
                    (tileId, std::get<0>(loaded), std::get<1>(loaded)
                     , config_.textureQuality, config_.smMergeOptions);
            } else {
                mesh = std::get<0>(loaded);
                atlas = std::get<1>(loaded);
            }
        }

        // measure mesh
        {
            auto mbs(miniball::minimumBoundingSphere(MeshVertices(*mesh)));
            node->srcMbs.center = mbs.center;
            node->srcMbs.r = mbs.radius;

            n.mbs.center = setup_.work2dst(mbs.center);
            n.mbs.r = mbs.radius;
        }
        nodeReference.mbs = n.mbs;

        // convert mesh vertices to output SRS
        warpInPlace(*mesh, setup_.work2dst);

        // measure extents
        const auto meshExtents(measureMesh(*mesh));
        UTILITY_OMP(critical(vef2slpk_process_2))
            math::update(sceneExtents_, meshExtents);

        // write mesh and atlas
        write(writer_, n, sr, *mesh, *atlas);

        Done done(++generated_, total_);
        LOG(info3)
            << "Generated node " << done << " <" << nodeId
            << "> from tile " << tileId << ".";
    } else {
        // non-geometry node, fill in
        nodeReference.id = asString(nodeId);
        nodeReference.href = "../" + nodeReference.id;
    }

    // proces children -> go down
    int childIndex(0);
    for (auto child : vts::children(tileId)) {
        UTILITY_OMP(task)
        {
            auto childNode(process(child, nodeId.child(childIndex)));

            UTILITY_OMP(critical(vef2slpk_process_1))
                node->child(childNode);
        }
        ++childIndex;
    }

    // done
    return node;
}

void Generator::operator()(/**vt::ExternalProgress &progress*/)
{
    {
        NodeHolder::pointer root;

        UTILITY_OMP(parallel shared(root))
            UTILITY_OMP(single)
            {
                root = process({}, {});
            }

        root->finish(setup_);
        root->write(writer_);
    }

    // finish archive
    writer_.flush([&](slpk::SceneLayerInfo &sli) -> void
    {
        sli.store->extents = sceneExtents_;
    });
}

int Vef2Slpk::run()
{
    // First, open output zip file to be sure the precondition is met
    utility::zip::Writer zip(output_, overwrite_ || config_.resume);

    tools::TmpTileset ts(tmptsPath_, !config_.resume, config_.binaryOrder);
    ts.keep(config_.keepTmpset);

    // measure mesh extents
    Setup setup;
    if (!config_.resume) {
        // load input manifests
        vef::Archive::list input;
        input.reserve(input_.size());

        // load input manifests
        for (const auto &path : input_) {
            input.emplace_back(path);
            if (!input.back().manifest().srs) {
                LOG(fatal)
                    << "VEF archive " << path
                    << " doesn't have assigned an SRS, cannot proceed.";
                return EXIT_FAILURE;
            }
        }

        LOG(info3) << "Analyzing input...";
        // not resuming, cut tiles
        setup = makeSetup(config_, input);
        LOG(info3) << "Analyzing input... done.";

        for (const auto &archive : input) {
            // cut to tiles
            vef::cutToTiles(ts, archive, setup.workExtents
                            , setup.cutSrs(archive.manifest().srs.value())
                            , setup.maxLod, config_.clipMargin);
        }

        // save setup
        writeSetup(tmptsPath_ / "setup.conf", setup);
    } else {
        // (try to) resume
        LOG(info3) << "Resuming from " << tmptsPath_ << ".";
        setup = readSetup(tmptsPath_ / "setup.conf");
        // TODO: check for the same SRS etc.
    }

    // output file
    slpk::Writer writer(std::move(zip)
                        , {}, makeSceneLayerInfo(config_, setup));

    Generator(writer, ts, config_, setup)(/* progress */);

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
