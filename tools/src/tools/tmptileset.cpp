#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "dbglog/dbglog.hpp"

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

    bool hasTile(const vts::TileId &tileId) const {
        return index_.get(tileId);
    }

    void setTile(const vts::TileId &tileId) {
        index_.set(tileId, 1);
    }

    vts::Driver::pointer driver() { return driver_; }

    void flush() { driver_->flush(); }

private:
    vts::TileIndex index_;
    vts::Driver::pointer driver_;
};

TmpTileset::TmpTileset(const boost::filesystem::path &root)
    : root_(root)
{
    // make room for tilesets
    fs::remove_all(root_);
    // create root for tilesets
    fs::create_directories(root_);
}

TmpTileset::~TmpTileset()
{
    // cleanup
    fs::remove_all(root_);
}

void TmpTileset::store(const vts::TileId &tileId, const vts::Mesh &mesh
                       , const vts::opencv::Atlas &atlas)
{
    // get driver for tile
    auto driver([&]() -> vts::Driver::pointer
    {
        std::unique_lock<std::mutex> lock(mutex_);
        for (auto &slice : slices_) {
            // TODO: make get/set in one pass
            if (!slice->hasTile(tileId)) {
                slice->setTile(tileId);
                return slice->driver();
            }
        }

        // no available slice for this tile, create new
        // path
        auto path(root_ / boost::lexical_cast<std::string>(slices_.size()));
        LOG(info3) << "Creating temporary tileset at " << path << ".";
        slices_.push_back(std::make_shared<Slice>(path));
        auto &slice(slices_.back());
        slice->setTile(tileId);
        return slice->driver();
    }());


    // store mesh (under lock)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        auto os(driver->output(tileId, storage::TileFile::mesh));
        saveMesh(os, mesh);
        os->close();
    }

    // store atlas (under lock)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        auto os(driver->output(tileId, storage::TileFile::atlas));
        atlas.serialize(os->get());
        os->close();
    }
}

void TmpTileset::flush()
{
    // unlocked!
    for (const auto &slice : slices_) {
        slice->flush();
    }
}

} } } // namespace vadstena::vts::tools
