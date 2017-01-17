#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "dbglog/dbglog.hpp"

#include "vts-libs/vts/tileset/driver.hpp"
#include "vts-libs/vts/opencv/atlas.hpp"

#include "./tmptileset.hpp"

namespace fs = boost::filesystem;

namespace vadstena { namespace vts { namespace tools {

class TmpTileset::Slice {
public:
    typedef std::shared_ptr<Slice> pointer;

    Slice(const fs::path &root)
        : driver_(Driver::create(root, driver::PlainOptions(5), {}))
    {}

    bool hasTile(const TileId &tileId) const {
        return index_.get(tileId);
    }

    void setTile(const TileId &tileId) {
        index_.set(tileId, 1);
    }

    Driver::pointer driver() { return driver_; }

    Driver::pointer driver() const { return driver_; }

    const TileIndex& index() { return index_; }

    void flush() { driver_->flush(); }

private:
    TileIndex index_;
    Driver::pointer driver_;
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

void TmpTileset::store(const TileId &tileId, const Mesh &mesh
                       , const Atlas &atlas)
{
    LOG(debug)
        << tileId << " Storing mesh with "
        << std::accumulate(mesh.begin(), mesh.end(), std::size_t(0)
                           , [](std::size_t v, const SubMesh &sm) {
                               return v + sm.faces.size();
                           })
        << " faces.";

    // get driver for tile
    auto driver([&]() -> Driver::pointer
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

std::tuple<Mesh::pointer, opencv::HybridAtlas::pointer>
TmpTileset::load(const vts::TileId &tileId, int quality)
{
    auto input([&](const Driver::pointer &driver, TileFile type)
               -> IStream::pointer
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return driver->input(tileId, type);
    });

    std::tuple<Mesh::pointer, opencv::HybridAtlas::pointer> tile;
    auto &mesh(std::get<0>(tile));
    auto &atlas(std::get<1>(tile));

    for (const auto &slice : slices_) {
        if (!slice->hasTile(tileId)) { continue; }

        auto driver(slice->driver());

        auto is(input(driver, storage::TileFile::mesh));
        Mesh m(loadMesh(is->get(), is->name()));

        opencv::HybridAtlas a(quality);
        {
            auto is(input(driver, storage::TileFile::atlas));
            a.deserialize(is->get(), is->name());
        }

        if (!mesh) {
            mesh = std::make_shared<Mesh>(m);
        } else {
            mesh->submeshes.insert(mesh->submeshes.end(), m.submeshes.begin()
                                   , m.submeshes.end());
        }

        if (!atlas) {
            atlas = std::make_shared<opencv::HybridAtlas>(a);
        } else {
            atlas->append(a);
        }
    }

    return tile;
}

void TmpTileset::flush()
{
    // unlocked!
    for (const auto &slice : slices_) {
        slice->flush();
    }
}

TileIndex TmpTileset::tileIndex() const
{
    TileIndex ti;
    for (const auto &slice : slices_) {
        ti = unite(ti, slice->index());
    }
    return ti;
}

} } } // namespace vadstena::vts::tools
