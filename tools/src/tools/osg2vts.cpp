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

typedef std::queue<fs::path> PathQueue;

class Measure : public osg::NodeVisitor {
public:
    Measure(PathQueue &pathQueue)
        : osg::NodeVisitor(VisitorType::NODE_VISITOR
                           , TraversalMode::TRAVERSE_ALL_CHILDREN)
        , pathQueue_(pathQueue)
    {}

    virtual void apply(osg::Geode &node) {
        LOG(info4) << "Geode:";

        const int count(node.getNumDrawables());
        for (int i(0); i < count; ++i) {
            const auto *drawable(node.getDrawable(i));
            if (const auto *geometry = drawable->asGeometry()) {
                LOG(info4) << "    geometry: " << geometry;
                if (const auto *va = geometry->getVertexArray()) {
                    LOG(info4)
                        << "        vertices: " << va->getNumElements();
                }

                if (const auto ta = geometry->getTexCoordArray(0)) {
                    LOG(info4)
                        << "        texCoords: " << ta->getNumElements();
                }
            } else {
                LOG(info4) << "    unknown drawable";
            }

            const auto *stateSet(drawable->getStateSet());
            if (const auto *attr = stateSet->getTextureAttribute
                (0, osg::StateAttribute::Type::TEXTURE))
            {
                if (const auto *texture = dynamic_cast<const osg::Texture2D*>
                    (attr->asTexture()))
                {
                    const auto *image(texture->getImage());

                    const auto *buf
                        (static_cast<const char*>(image->getDataPointer()));
                    std::cerr.write(buf, 32);

                    LOG(info4)
                        << "        texture <" << texture->className() << ">: "
                        << texture->getTextureWidth() << "x"
                        << texture->getTextureHeight()
                        << ", " << image->getTotalDataSize()
                        ;
                }
            }
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
                pathQueue_.push
                    (fs::path(node.getDatabasePath()) / fname);
            } else {
                auto *child(node.getChild(i));
                LOG(info4) << "    child: library: <" << child->libraryName()
                           << ">, class <" << child->className() << ">.";
                child->accept(*this);
            }
        }
    }

private:
    PathQueue &pathQueue_;
};

int Osg2Vts::run()
{
    PathQueue pathQueue;
    Measure measure(pathQueue);

    pathQueue.push(input_);

    while (!pathQueue.empty()) {
        const auto &path(pathQueue.front());
        osg::ref_ptr<osg::Node> node(osgDB::readNodeFile(path.string()));

        LOG(info4) << "Loaded node from " << path
                   << " [library: <" << node->libraryName()
                   << ">, class <" << node->className() << ">].";

        pathQueue.pop();

        node->accept(measure);
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
