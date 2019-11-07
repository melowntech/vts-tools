#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/cppversion.hpp"
#include "utility/limits.hpp"
#include "utility/path.hpp"
#include "utility/openmp.hpp"

#include "service/cmdline.hpp"

#include "vts-libs/registry/po.hpp"
#include "vts-libs/vts/io.hpp"
#include "vts-libs/vts.hpp"
#include "vts-libs/vts/csconvertor.hpp"

#include "3dtiles/3dtiles.hpp"
#include "3dtiles/encoder.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace vts = vtslibs::vts;
namespace vr = vtslibs::registry;
namespace tdt = threedtiles;

namespace {

class Vts23DTiles : public service::Cmdline
{
public:
    Vts23DTiles()
        : service::Cmdline("vts23dtiles", BUILD_TARGET_VERSION)
        , createMode_(vts::CreateMode::failIfExists)
    {}

    virtual void configuration(po::options_description &cmdline
                               , po::options_description &config
                               , po::positional_options_description &pd)
        override;

    virtual void configure(const po::variables_map &vars) override;

    virtual bool help(std::ostream &out, const std::string &what) const
        override;

    virtual int run() override;

private:
    fs::path output_;
    fs::path input_;
    boost::optional<vts::LodTileRange> range_;

    vts::CreateMode createMode_;

    tdt::Encoder::Config gConfig_;
};

void Vts23DTiles::configuration(po::options_description &cmdline
                                , po::options_description &config
                                , po::positional_options_description &pd)
{
    vr::registryConfiguration(cmdline, vr::defaultPath());

    gConfig_.configuration(cmdline);

    cmdline.add_options()
        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")
        ("input", po::value(&input_)->required()
         , "Path to input 3D Tileset archive.")
        ("overwrite", "Existing 3D tileset gets overwritten if set.")

        ("range", po::value<vts::LodTileRange>()
         , "Limit output to given LOD/tile range. Optional.")
        ;

    pd
        .add("input", 1)
        .add("output", 1);

    (void) config;
}

void Vts23DTiles::configure(const po::variables_map &vars)
{
    vr::registryConfigure(vars);

    gConfig_.configure(vars);

    createMode_ = (vars.count("overwrite")
                   ? vts::CreateMode::overwrite
                   : vts::CreateMode::failIfExists);

    if (vars.count("range")) {
        range_ = vars["range"].as<vts::LodTileRange>();
    }
}

bool Vts23DTiles::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(vts23dtiles: converts VTS tileset into 3D Tileset
usage
    vts23dtiles INPUT OUTPUT [OPTIONS]

Both tilesets have 1:1 tile mapping, no mesh cutting nor atlas repacking
is performed.
)RAW";
        return true;
    }

    return false;
}

using TF = vts::TileIndex::Flag;

class Generator : public tdt::Encoder {
public:
    Generator(const tdt::Encoder::Config &config
              , const vts::TileSet &its, const fs::path &root
              , const boost::optional<vts::LodTileRange> &range)
        : tdt::Encoder(config, root, makeValidTiles(its.tileIndex(), range))
        , its_(its)
        , conv_(its.referenceFrame().model.physicalSrs, config.srs)
    {}

private:
    static vts::TileIndex
    makeValidTiles(vts::TileIndex ti
                   , const boost::optional<vts::LodTileRange> &range)
    {
        // chose only textured meshes
        ti.simplify(TF::mesh | TF::atlas, TF::mesh | TF::atlas).makeAbsolute();

        if (!range) { return ti; }

        // clip to ranges
        const vts::LodRange lr(range->lod, ti.lodRange().max);

        // create clipping tile index
        vts::TileIndex clipTi;
        for (const auto &r : vts::Ranges(lr, range->range).ranges()) {
            clipTi.set(r.lod, r.range, TF::mesh);
        }

        const auto &combiner([&](TF::value_type o, TF::value_type n)
                             -> TF::value_type
        {
            // intersect
            return o & n;
        });

        return ti.combine(clipTi, combiner, ti.lodRange());
    }

    virtual TexturedMesh generate(const vts::TileId &tileId)
        UTILITY_OVERRIDE;

    const vts::TileSet &its_;
    const vts::CsConvertor conv_;
};

inline void warpInPlace(vts::SubMesh &mesh, const vts::CsConvertor &conv)
{
    for (auto &v : mesh.vertices) { v = conv(v); }
}

inline void warpInPlace(vts::Mesh &mesh, const vts::CsConvertor &conv)
{
    for (auto &sm : mesh) { warpInPlace(sm, conv); }
}

tdt::Encoder::TexturedMesh Generator::generate(const vts::TileId &tileId)
{
    LOG(info2) << "Cloning tile " << tileId << ".";

    TexturedMesh content;
    auto &atlas(content.initialize<vts::RawAtlas>());

    UTILITY_OMP(critical(vts23dtiles_generate_1))
    {
        *content.mesh = its_.getMesh(tileId);
        its_.getAtlas(tileId, atlas);
    }

    warpInPlace(*content.mesh, conv_);

    return content;
}

int Vts23DTiles::run()
{
    const auto its(vts::openTileSet(input_));

    Generator generator(gConfig_, its, output_, range_);
    generator.run();

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[])
{
    utility::unlimitedCoredump();
    return Vts23DTiles()(argc, argv);
}
