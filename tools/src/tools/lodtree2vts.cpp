#include <cstdlib>
#include <string>

#include "dbglog/dbglog.hpp"

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"

#include "service/cmdline.hpp"

#include "../vts-libs/vts.hpp"
#include "../vts-libs/registry/po.hpp"

#include "../tinyxml2/tinyxml2.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

namespace vs = vadstena::storage;
namespace vr = vadstena::registry;
namespace vts = vadstena::vts;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace xml = tinyxml2;

namespace {

class LodTree2Vts : public service::Cmdline
{
public:
    LodTree2Vts()
        : service::Cmdline("lodtree2vts", BUILD_TARGET_VERSION)
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
};

void LodTree2Vts::configuration(po::options_description &cmdline
                               , po::options_description &config
                               , po::positional_options_description &pd)
{
    vr::registryConfiguration(cmdline, vr::defaultPath());

    cmdline.add_options()
        ("input", po::value(&input_)->required()
         , "Path to input (vts0) tile set.")
        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")
        ("overwrite", "Existing tile set gets overwritten if set.")
        ;

    pd.add("input", 1);
    pd.add("output", 1);

    (void) config;
}

void LodTree2Vts::configure(const po::variables_map &vars)
{
    vr::registryConfigure(vars);

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

/*void testAssImp(const fs::path &path)
{
    Assimp::Importer imp;

    const aiScene *scene = imp.ReadFile(path.native(), 0);

    if (!scene) {
        LOGTHROW(err3, std::runtime_error)
            << "Error loading " << path << ": " << imp.GetErrorString();
    }

    LOG(info2) << "# vertices: " << scene->mMeshes[0]->mNumVertices;
    LOG(info2) << "# faces: " << scene->mMeshes[0]->mNumFaces;
}*/


int LodTree2Vts::run()
{
    //testAssImp("/mnt/media/vadstena/cowi/2016-03-18/Production_2/Data/Tile_+000_+006/Tile_+000_+006_L14_0.dae");

    xml::XMLDocument doc;
    doc.LoadFile("/mnt/media/vadstena/cowi/2016-03-18/Production_2/Data/LODTreeExport.xml");

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}


} // namespace


int main(int argc, char *argv[])
{
    return LodTree2Vts()(argc, argv);
}