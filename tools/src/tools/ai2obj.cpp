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

#include <boost/optional.hpp>
#include <boost/utility/in_place_factory.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/limits.hpp"
#include "utility/path.hpp"
#include "utility/openmp.hpp"
#include "utility/streams.hpp"

#include "service/cmdline.hpp"

#include "geometry/meshop.hpp"

#include "imgproc/readimage.cpp"
#include "imgproc/texturing.cpp"

#include "geo/csconvertor.hpp"

#include "vts-libs/vts/mesh.hpp"
#include "vts-libs/vts/meshop.hpp"
#include "vts-libs/vts/opencv/atlas.hpp"

#include "vts-libs/tools-support/assimp.hpp"

namespace po = boost::program_options;
namespace bio = boost::iostreams;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;
namespace vts = vtslibs::vts;
namespace tools = vtslibs::vts::tools;

namespace {

class Ai2Obj : public service::Cmdline
{
public:
    Ai2Obj()
        : service::Cmdline("ai2obj", BUILD_TARGET_VERSION)
        , overwrite_(false)
    {}

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
    bool overwrite_;
};

void Ai2Obj::configuration(po::options_description &cmdline
                             , po::options_description &config
                             , po::positional_options_description &pd)
{
    cmdline.add_options()
        ("output", po::value(&output_)->required()
         , "Path to output converted input.")
        ("input", po::value(&input_)->required()
         , "Path to input AI archive.")
        ("overwrite", "Generate output even if output directory exists.")
        ;

    pd
        .add("input", 1)
        .add("output", 1);

    (void) config;
}

void Ai2Obj::configure(const po::variables_map &vars)
{
    overwrite_ = vars.count("overwrite");
}

bool Ai2Obj::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(ai2obj

    Converts AI file into textured meshes in OBJ format.

usage
    ai2obj INPUT OUTPUT [OPTIONS]
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

int Ai2Obj::run()
{
    LOG(info4) << "Opening Assimp-supported file at " << input_ << ".";

    Assimp::Importer imp;
    const auto scene(tools::loadAssimpScene(imp, input_));
    const auto &mesh(std::get<0>(scene));
    const auto &atlas(std::get<1>(scene));

    if (!fs::create_directories(output_) && !overwrite_) {
        std::cerr << "Output path " << output_ << " already exists."
                  << std::endl;
        return EXIT_FAILURE;
    }

    for (int i(0), e(mesh.submeshes.size()); i != e; ++i) {
        const auto meshPath(output_ / str(boost::format("%d.obj") % i));
        const auto texPath(output_ / str(boost::format("%d.jpg") % i));
        const auto mtlPath(output_ / str(boost::format("%d.mtl") % i));

        // save mesh
        saveSubMeshAsObj(meshPath, mesh.submeshes[i], i, atlas.get()
                         , mtlPath.filename().string());

        // save texture
        atlas->write(texPath, i);

        // save material file
        writeMtl(mtlPath, texPath.filename().string());
    }

    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[])
{
    utility::unlimitedCoredump();
    return Ai2Obj()(argc, argv);
}
