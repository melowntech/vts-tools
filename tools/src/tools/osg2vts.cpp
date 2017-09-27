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

#include <queue>

#include <boost/utility/in_place_factory.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <osgDB/ReadFile>
#include <osg/Node>
#include <osg/NodeVisitor>
#include <osg/PagedLOD>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/GraphicsContext>

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

#if 0
class DummyTextureExtensions : public osg::Texture::Extensions
{
public:
    DummyTextureExtensions()
        : osg::Texture::Extensions(0)
    {
        _glCompressedTexImage2D = nullptr;
        _glCompressedTexSubImage2D = nullptr;
        _glGetCompressedTexImage = nullptr;
        _glTexImage2DMultisample = nullptr;
        _glTexParameterIiv = nullptr;
        _glTexParameterIuiv = nullptr;
        _glBindImageTexture = nullptr;

        _isMultiTexturingSupported = false;
        _isTextureFilterAnisotropicSupported = false;
        _isTextureSwizzleSupported = false;
        _isTextureCompressionARBSupported = false;
        _isTextureCompressionS3TCSupported = false;
        _isTextureCompressionPVRTC2BPPSupported = false;
        _isTextureCompressionPVRTC4BPPSupported = false;
        _isTextureCompressionETCSupported = false;
        _isTextureCompressionRGTCSupported = false;
        _isTextureCompressionPVRTCSupported = false;
        _isTextureMirroredRepeatSupported = false;
        _isTextureEdgeClampSupported = false;
        _isTextureBorderClampSupported = false;
        _isGenerateMipMapSupported = false;
        _preferGenerateMipmapSGISForPowerOfTwo = false;
        _isTextureMultisampledSupported = false;
        _isShadowSupported = false;
        _isShadowAmbientSupported = false;
        _isClientStorageSupported = false;
        _isNonPowerOfTwoTextureMipMappedSupported = false;
        _isNonPowerOfTwoTextureNonMipMappedSupported = true;
        _isTextureIntegerEXTSupported = false;
        _isTextureMaxLevelSupported = false;
        _maxTextureSize = 1 << 18;
        _numTextureUnits = 1;
    }
};


class DummyGC : public osg::GraphicsContext {
public:
    DummyGC()
        : osg::GraphicsContext()
    {
        osg::Texture::setExtensions(0, new DummyTextureExtensions());
    }

private:
    virtual void bindPBufferToTextureImplementation(GLenum) {}
    virtual const char* className() const { return "DummyGC"; }
    virtual void clear() {}
    virtual Object* clone(const osg::CopyOp&) const { return new DummyGC(); }
    virtual Object* cloneType() const { return new DummyGC(); }
    virtual void closeImplementation() {}
    virtual bool isRealizedImplementation()	const { return true; }
    virtual bool isSameKindAs(const osg::Object *object) const {
        return typeid(*this) == typeid(*object);
    }
    virtual const char* libraryName() const { return "main"; }
    virtual bool makeContextCurrentImplementation(osg::GraphicsContext*) {
        return true;
    }
    virtual bool makeCurrentImplementation() { return true; }
    virtual bool realizeImplementation() {return true; }
    virtual bool releaseContextImplementation() { return true; }
    virtual void resizedImplementation(int, int, int, int) {}
    virtual void runOperations() {}
    virtual void swapBuffersImplementation() {}
    virtual bool valid() const { return true; }
};
#endif

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

class Osg2Vts : public service::Cmdline
{
public:
    Osg2Vts()
        : service::Cmdline("osg2vts", BUILD_TARGET_VERSION)
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

void Osg2Vts::configuration(po::options_description &cmdline
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
         , "Path to input OSG(B) archive.")
        ("overwrite", "Existing tile set gets overwritten if set.")
        ;

    vt::progressConfiguration(config);

    pd
        .add("input", 1)
        .add("output", 1);

    (void) config;
}

void Osg2Vts::configure(const po::variables_map &vars)
{
    vr::registryConfigure(vars);

    config_.configure(vars);

    createMode_ = (vars.count("overwrite")
                   ? vts::CreateMode::overwrite
                   : vts::CreateMode::failIfExists);

    epConfig_ = vt::configureProgress(vars);
}

bool Osg2Vts::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(osg2vts
usage
    osg2vts INPUT OUTPUT [OPTIONS]

)RAW";
    }
    return false;
}

struct TreeElement {
    fs::path path;
    int depth;

    TreeElement(const fs::path &path, int depth)
        : path(path), depth(depth)
    {}

    typedef std::queue<TreeElement> queue;
};


typedef math::Point3_<unsigned int> Face;

class MeshLoader {
public:
    virtual ~MeshLoader() {}
    virtual void addVertex(const math::Point3d&) = 0;
    virtual void addTexture(const math::Point2d&) = 0;
    virtual void addFace(const Face &mesh, const Face &tc = Face()) = 0;

};

class MeasureMesh : public MeshLoader {
public:
    virtual void addVertex(const math::Point3d &p) {
        vertices_.push_back(p);
    }

    virtual void addTexture(const math::Point2d &t) {
        tc_.push_back(t);
    }

    virtual void addFace(const Face &mesh, const Face &tc = Face()) {
        (void) mesh;
        (void) tc;
    }

private:
    math::Points3 vertices_;
    math::Points2 tc_;
};

template <typename T>
void loadVertices(MeshLoader &loader, const osg::MixinVector<T> &vector)
{
    for (const auto &p : vector) {
        loader.addVertex(math::Point3(p.x(), p.y(), p.z()));
    }
}

template <typename T>
void loadTexCoords(MeshLoader &loader, const osg::MixinVector<T> &vector)
{
    for (const auto &p : vector) {
        loader.addTexture(math::Point2(p.x(), p.y()));
    }
}

void loadMesh(const osg::Geometry &geometry, MeshLoader &loader)
{
    const auto &primitiveSets(geometry.getPrimitiveSetList());
    for (const auto &primitiveSet : primitiveSets) {
        LOG(info4) << "primitiveSet.getMode(): " << primitiveSet->getMode()
                   << ", primitiveSet.getType(): " << primitiveSet->getType()
                   << ", primitiveSet.getNumIndices(): "
                   << primitiveSet->getNumIndices()
            ;
        // auto pc(primitiveSet->getNumPrimitives
        // const auto &dw(primitiveSet->getDrawElements());
        // for getNumPrimitives
    }

    if (const auto *a = geometry.getVertexArray()) {
        LOG(info4)
            << "\n        vertices: " << a->getNumElements()
            << "\n            type: " << a->getType()
            << "\n            dataType: "
            << std::hex << a->getDataType()
            << "\n            class: " << a->className()
            ;
        switch (a->getType()) {
        case osg::Array::Vec3ArrayType:
            loadVertices(loader, dynamic_cast<const osg::Vec3Array&>(*a));
            break;

        case osg::Array::Vec3dArrayType:
            loadVertices(loader, dynamic_cast<const osg::Vec3Array&>(*a));
            break;

        default:
            LOGTHROW(err2, std::runtime_error)
                << "Unsupported vertex array type <" << a->className()
                << ">.";
        }
    }

    if (const auto *a = geometry.getTexCoordArray(0)) {
        LOG(info4)
            << "\n        texCoords: " << a->getNumElements()
            << "\n            class: " << a->className()
            ;
        switch (a->getType()) {
        case osg::Array::Vec2ArrayType:
            loadTexCoords(loader, dynamic_cast<const osg::Vec2Array&>(*a));
            break;

        case osg::Array::Vec2dArrayType:
            loadTexCoords(loader, dynamic_cast<const osg::Vec2Array&>(*a));
            break;

        default:
            LOGTHROW(err2, std::runtime_error)
                << "Unsupported texCoord array type <" << a->className()
                << ">.";
        }
    }
}

double bestLod(const Config &config
               , const vts::NodeInfo &rfNode
               , const vts::SubMeshArea &area)
{
    const double texelArea(area.mesh / area.internalTexture);

    LOG(info3) << "<" << rfNode.srs() << ">: texel size: "
               << std::sqrt(texelArea) << ".";

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

    void update(const vts::SubMesh &mesh, const math::Size2 &txSize)
    {
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

class Measure : public osg::NodeVisitor {
public:
    Measure(TreeElement::queue &teq, int depth)
        : osg::NodeVisitor(VisitorType::NODE_VISITOR
                           , TraversalMode::TRAVERSE_ALL_CHILDREN)
        , teq_(teq), depth_(depth)
    {}

    virtual void apply(osg::Geode &node) {
        LOG(info4) << "Geode[" << depth_ << "]:";

        const int count(node.getNumDrawables());
        for (int i(0); i < count; ++i) {
            osg::ref_ptr<const osg::Texture2D> texture;

            const auto *drawable(node.getDrawable(i));

            osg::ref_ptr<const osg::Geometry>
                geometry(drawable->asGeometry());

            if (!geometry) {
                LOG(info4) << "    unknown drawable";
                continue;
            }

            MeasureMesh loader;
            loadMesh(*geometry, loader);

            osg::ref_ptr<const osg::StateSet>
                stateSet(drawable->getStateSet());

            if (const auto *attr = stateSet->getTextureAttribute
                (0, osg::StateAttribute::Type::TEXTURE))
            {
                texture = dynamic_cast<const osg::Texture2D*>
                    (attr->asTexture());
            }

            if (!texture) {
                LOGTHROW(err2, std::runtime_error)
                    << "Untextured mesh.";
            }

            const auto *image(texture->getImage());

            LOG(info4)
                << "        texture <" << texture->className() << ">: size="
                << image->s() << "x" << image->t()
                << ", bytes=" << image->getTotalDataSize()
                << ", format=" << std::hex << image->getPixelFormat()
                << ", filename=\"" << image->getFileName() << "\""
                ;
        }
    }

    virtual void apply(osg::Group &node) {
        const int count(node.getNumChildren());
        LOG(info4) << "Group: " << count;

        for (int i(0); i < count; ++i) {
            auto *child(node.getChild(i));

            LOG(info4) << "    child: library: <" << child->libraryName()
                       << ">, class <" << child->className() << ">.";
            child->accept(*this);
        }
    }

    virtual void apply(osg::LOD &node) {
        (void) node;
        LOG(info4) << "LOD!";
    }

    virtual void apply(osg::PagedLOD &node) {
        int count(node.getNumFileNames());
        LOG(info4) << "PagedLOD: " << count << ".";

        for (int i(0); i < count; ++i) {
            const auto &fname(node.getFileName(i));
            if (!fname.empty()) {
                teq_.emplace
                    (fs::path(node.getDatabasePath()) / fname, depth_ + 1);
            } else {
                auto *child(node.getChild(i));
                LOG(info4) << "    child: library: <" << child->libraryName()
                           << ">, class <" << child->className() << ">.";
                child->accept(*this);
            }
        }
    }

private:
    TreeElement::queue &teq_;
    int depth_;
};

int Osg2Vts::run()
{
    TreeElement::queue teq;
    teq.emplace(input_, 0);

    while (!teq.empty()) {
        const auto &te(teq.front());
        osg::ref_ptr<osg::Node> node(osgDB::readNodeFile(te.path.string()));
        LOG(info4) << "Loaded node from " << te.path
                   << " [library: <" << node->libraryName()
                   << ">, class <" << node->className() << ">].";
        Measure measure(teq, te.depth);
        node->accept(measure);
        teq.pop();
    }

#if 0
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

    (void) properties;
#endif

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[])
{
    utility::unlimitedCoredump();
    return Osg2Vts()(argc, argv);
}
