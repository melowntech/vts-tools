#include <boost/filesystem.hpp>

#include "vts-libs/vts/tileset/driver.hpp"

#include "./tmptileset.hpp"

namespace fs = boost::filesystem;

namespace vadstena { namespace vts { namespace tools {

class TmpTileset::Slice {
public:
    typedef std::shared_ptr<Slice> pointer;

    Slice(const fs::path &root)
        : driver_(vts::Driver::create
                  (root, vts::driver::PlainOptions(5), {}))
    {}

private:
    vts::TileIndex index_;
    vts::Driver::pointer driver_;
};

TmpTileset::TmpTileset(const boost::filesystem::path &root)
    : root_(root)
{
    fs::create_directories(root_);
}

} } } // namespace vadstena::vts::tools
