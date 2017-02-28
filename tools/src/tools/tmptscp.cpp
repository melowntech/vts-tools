#include <cstdlib>
#include <string>
#include <iostream>
#include <algorithm>
#include <iterator>

#include <boost/algorithm/string/split.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "dbglog/dbglog.hpp"
#include "utility/streams.hpp"

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/progress.hpp"
#include "utility/streams.hpp"
#include "utility/openmp.hpp"
#include "utility/progress.hpp"
#include "utility/openmp.hpp"
#include "utility/limits.hpp"
#include "utility/filesystem.hpp"

#include "service/cmdline.hpp"

#include "math/transform.hpp"
#include "math/filters.hpp"

#include "imgproc/scanconversion.hpp"
#include "imgproc/jpeg.hpp"

#include "geometry/parse-obj.hpp"

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
#include "vts-libs/vts/math.hpp"
#include "vts-libs/vts/opencv/atlas.hpp"


#include "./tmptileset.hpp"
#include "./repackatlas.hpp"


namespace po = boost::program_options;
namespace bio = boost::iostreams;
namespace ba = boost::algorithm;
namespace fs = boost::filesystem;
namespace ublas = boost::numeric::ublas;
namespace vs = vtslibs::storage;
namespace vr = vtslibs::registry;
namespace vts = vtslibs::vts;
namespace tools = vtslibs::vts::tools;

namespace {

class TmpTsCherryPick : public service::Cmdline
{
public:
    TmpTsCherryPick()
        : service::Cmdline("tmptscp", BUILD_TARGET_VERSION)
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
    std::vector<vts::TileId> tileIds_;
};

void TmpTsCherryPick::configuration(po::options_description &cmdline
                             , po::options_description &config
                             , po::positional_options_description &pd)
{
    vr::registryConfiguration(cmdline, vr::defaultPath());

    cmdline.add_options()
        ("input", po::value(&input_)->required()
         , "Path to input tmp tileset.")
        ("output", po::value(&output_)->required()
         , "Path to output tmp tileset.")
        ("tileId", po::value(&tileIds_)->required()
         , "One (or more) tiles to cherry pick from input tmp tileset.")
        ;

    pd
        .add("input", 1)
        .add("output", 1);

    (void) config;
}

void TmpTsCherryPick::configure(const po::variables_map &vars)
{
    (void) vars;
}

bool TmpTsCherryPick::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(tmptscp
usage
    tmptscp INPUT OUTPUT

)RAW";
    }
    return false;
}

int TmpTsCherryPick::run()
{
    // open source and destination, keep them both
    tools::TmpTileset src(input_, false);
    src.keep(true);
    tools::TmpTileset dst(output_);
    dst.keep(true);

    utility::copy_file(input_ / "navtile.info"
                       , output_ / "navtile.info", true);

    for (const auto &tileId : tileIds_) {
        const auto tile(src.load(tileId, 0));
        dst.store(tileId, *std::get<0>(tile), *std::get<1>(tile));
    }

    dst.flush();

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[])
{
    utility::unlimitedCoredump();
    return TmpTsCherryPick()(argc, argv);
}
