#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "dbglog/dbglog.hpp"

#include "utility/binaryio.hpp"
#include "utility/streams.hpp"

#include "math/math.hpp"

#include "vts-libs/vts/tileset/driver.hpp"
#include "vts-libs/vts/opencv/atlas.hpp"
#include "vts-libs/vts/tileset/config.hpp"

#include "./tmptileset.hpp"

namespace fs = boost::filesystem;
namespace bin = utility::binaryio;

namespace vadstena { namespace vts { namespace tools {

namespace {

// mesh proper
const char MAGIC[2] = { 'S', 'M' };
const std::uint16_t VERSION = 0;

void saveSimpleMesh(std::ostream &out, const Mesh &mesh)
{
    // helper functions
    auto saveVertexComponent([&out](double v, double o, double s) -> void
    {
        bin::write
            (out, std::uint32_t
             (std::round
              (((v - o) * std::numeric_limits<std::uint32_t>::max()) / s)));
    });

    auto saveTexCoord([&out](double v)
    {
        v = std::round(math::clamp(v, 0.0, 1.0)
                       * std::numeric_limits<std::uint32_t>::max());
        bin::write(out, std::uint32_t(v));
    });

    // write header
    bin::write(out, MAGIC);
    bin::write(out, std::uint16_t(0));

    bin::write(out, std::uint16_t(mesh.submeshes.size()));

    // write submeshes
    for (const auto &sm : mesh) {
        // compute extents
        const auto bbox(extents(sm));
        const math::Point3d bbsize(bbox.ur - bbox.ll);

        // write extents
        bin::write(out, bbox.ll(0));
        bin::write(out, bbox.ll(1));
        bin::write(out, bbox.ll(2));
        bin::write(out, bbox.ur(0));
        bin::write(out, bbox.ur(1));
        bin::write(out, bbox.ur(2));

        // write vertices
        bin::write(out, std::uint16_t(sm.vertices.size()));
        for (const auto &vertex : sm.vertices) {
            saveVertexComponent(vertex(0), bbox.ll(0), bbsize(0));
            saveVertexComponent(vertex(1), bbox.ll(1), bbsize(1));
            saveVertexComponent(vertex(2), bbox.ll(2), bbsize(2));
        }

        // write tc
        bin::write(out, std::uint16_t(sm.tc.size()));
        for (const auto &tc : sm.tc) {
            saveTexCoord(tc(0));
            saveTexCoord(tc(1));
        }

        // save faces
        bin::write(out, std::uint16_t(sm.faces.size()));

        auto ifacesTc(sm.facesTc.begin());
        for (auto &face : sm.faces) {
            // face
            bin::write(out, std::uint16_t(face(0)));
            bin::write(out, std::uint16_t(face(1)));
            bin::write(out, std::uint16_t(face(2)));

            // tc face
            bin::write(out, std::uint16_t((*ifacesTc)(0)));
            bin::write(out, std::uint16_t((*ifacesTc)(1)));
            bin::write(out, std::uint16_t((*ifacesTc)(2)));
            ++ifacesTc;
        }
    }
}

Mesh loadSimpleMesh(std::istream &in, const fs::path &path)
{
    // helper functions
    auto loadVertexComponent([&in](double o, double s) -> double
    {
        std::uint32_t v;
        bin::read(in, v);
        return o + ((v * s) / std::numeric_limits<std::uint32_t>::max());
    });

    auto loadTexCoord([&in]() -> double
    {
        std::uint32_t v;
        bin::read(in, v);
        return (double(v) / std::numeric_limits<std::uint32_t>::max());
    });

    // Load mesh headers first
    char magic[sizeof(MAGIC)];
    std::uint16_t version;

    bin::read(in, magic);
    bin::read(in, version);

    LOG(info1) << "Mesh version: " << version;

    if (std::memcmp(magic, MAGIC, sizeof(MAGIC))) {
        LOGTHROW(err1, storage::BadFileFormat)
            << "File " << path << " is not a VTS simplemesh file.";
    }
    if (version > VERSION) {
        LOGTHROW(err1, storage::VersionError)
            << "File " << path
            << " has unsupported version (" << version << ").";
    }

    std::uint16_t subMeshCount;
    bin::read(in, subMeshCount);

    Mesh mesh;

    mesh.submeshes.resize(subMeshCount);
    for (auto &sm : mesh) {
        // load sub-mesh bounding box
        math::Extents3 bbox;
        bin::read(in, bbox.ll(0));
        bin::read(in, bbox.ll(1));
        bin::read(in, bbox.ll(2));
        bin::read(in, bbox.ur(0));
        bin::read(in, bbox.ur(1));
        bin::read(in, bbox.ur(2));

        const math::Point3d bbsize(bbox.ur - bbox.ll);

        // load vertices
        std::uint16_t vertexCount;
        bin::read(in, vertexCount);
        sm.vertices.resize(vertexCount);
        for (auto &vertex : sm.vertices) {
            vertex(0) = loadVertexComponent(bbox.ll(0), bbsize(0));
            vertex(1) = loadVertexComponent(bbox.ll(1), bbsize(1));
            vertex(2) = loadVertexComponent(bbox.ll(2), bbsize(2));
        }

        // load tc
        std::uint16_t tcCount;
        bin::read(in, tcCount);
        sm.tc.resize(tcCount);
        for (auto &tc : sm.tc) {
            tc(0) = loadTexCoord();
            tc(1) = loadTexCoord();
        }

        // load faces
        std::uint16_t faceCount;
        bin::read(in, faceCount);
        sm.faces.resize(faceCount);
        sm.facesTc.resize(faceCount);
        auto ifacesTc(sm.facesTc.begin());

        for (auto &face : sm.faces) {
            std::uint16_t index;
            bin::read(in, index); face(0) = index;
            bin::read(in, index); face(1) = index;
            bin::read(in, index); face(2) = index;

            bin::read(in, index); (*ifacesTc)(0) = index;
            bin::read(in, index); (*ifacesTc)(1) = index;
            bin::read(in, index); (*ifacesTc)(2) = index;
            ++ifacesTc;
        }
    }

    return mesh;
}

} // namespace

class TmpTileset::Slice {
public:
    typedef std::shared_ptr<Slice> pointer;
    struct OpenTag {};

    Slice(const fs::path &root)
        : driver_(Driver::create(root, driver::PlainOptions(5), {}))
    {}

    Slice(const fs::path &root, const OpenTag&)
        : driver_(Driver::open(root, Driver::BareConfigTag{}, {}))
    {
        index_.load(driver_->input(File::tileIndex)->get());
    }

    bool hasTile(const TileId &tileId) const {
        return index_.get(tileId);
    }

    void setTile(const TileId &tileId) {
        index_.set(tileId, 1);
    }

    Driver::pointer driver() { return driver_; }

    Driver::pointer driver() const { return driver_; }

    const TileIndex& index() { return index_; }

    void flush() {
        {
            auto f(driver_->output(File::tileIndex));
            index_.save(f->get());
            f->close();
        }
        {
            auto f(driver_->output(File::config));
            vts::tileset::saveDriver(f->get(), driver_->options());
            f->close();
        }
        driver_->flush();
    }

    void saveMesh(const TileId &tileId, const Mesh &mesh);
    void saveAtlas(const TileId &tileId, const Atlas &atlas);

    IStream::pointer input(const TileId &tileId, TileFile type) {
        std::unique_lock<std::mutex> lock(mutex_);
        return driver_->input(tileId, type);
    }

private:
    std::mutex mutex_;
    TileIndex index_;
    Driver::pointer driver_;
};

void TmpTileset::Slice::saveMesh(const TileId &tileId, const Mesh &mesh)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto os(driver_->output(tileId, storage::TileFile::mesh));
    saveSimpleMesh(os->get(), mesh);
    os->close();
}

void TmpTileset::Slice::saveAtlas(const TileId &tileId, const Atlas &atlas)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto os(driver_->output(tileId, storage::TileFile::atlas));
    atlas.serialize(os->get());
    os->close();
}

TmpTileset::TmpTileset(const boost::filesystem::path &root
                       , bool create)
    : root_(root), keep_(false)
{
    if (create) {
        // make room for tilesets
        fs::remove_all(root_);
        // create root for tilesets
        fs::create_directories(root_);
        return;
    }

    for (int i(0);; ++i) {
        auto path(root_ / boost::lexical_cast<std::string>(i));
        if (!exists(path)) { break; }
        slices_.push_back(std::make_shared<Slice>(path, Slice::OpenTag{}));
    }

    if (slices_.empty()) {
        LOGTHROW(err1, std::runtime_error)
            << "No tileset slice found in temporary tileset " << root_ << ".";
    }
}

TmpTileset::~TmpTileset()
{
    // cleanup
    if (!keep_) {
        fs::remove_all(root_);
    }
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
    auto slice([&]() -> Slice::pointer
    {
        std::unique_lock<std::mutex> lock(mutex_);
        for (auto &slice : slices_) {
            // TODO: make get/set in one pass
            if (!slice->hasTile(tileId)) {
                slice->setTile(tileId);
                return slice;
            }
        }

        // no available slice for this tile, create new
        // path
        auto path(root_ / boost::lexical_cast<std::string>(slices_.size()));
        LOG(info3) << "Creating temporary tileset at " << path << ".";
        slices_.push_back(std::make_shared<Slice>(path));
        auto &slice(slices_.back());
        slice->setTile(tileId);
        return slice;
    }());

    slice->saveMesh(tileId, mesh);
    slice->saveAtlas(tileId, atlas);
}

std::tuple<Mesh::pointer, opencv::HybridAtlas::pointer>
TmpTileset::load(const vts::TileId &tileId, int quality)
{
    std::tuple<Mesh::pointer, opencv::HybridAtlas::pointer> tile;
    auto &mesh(std::get<0>(tile));
    auto &atlas(std::get<1>(tile));

    for (const auto &slice : slices_) {
        if (!slice->hasTile(tileId)) { continue; }

        auto driver(slice->driver());

        auto is(slice->input(tileId, storage::TileFile::mesh));
        Mesh m(loadSimpleMesh(is->get(), is->name()));

        opencv::HybridAtlas a(quality);
        {
            auto is(slice->input(tileId, storage::TileFile::atlas));
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
