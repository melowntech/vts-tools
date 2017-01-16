/**
 * Temporary tileset. Used during encoding.
 * \file tmptileset.hpp
 * \author Vaclav Blazek <vaclav.blazek@melown.com>
 */

#ifndef vts_tools_tmptileset_hpp_included
#define vts_tools_tmptileset_hpp_included

#include <mutex>

#include <boost/filesystem/path.hpp>

#include "vts-libs/vts.hpp"
#include "vts-libs/vts/types.hpp"
#include "vts-libs/vts/mesh.hpp"
#include "vts-libs/vts/opencv/atlas.hpp"

namespace vadstena { namespace vts { namespace tools {

class TmpTileset {
public:
    TmpTileset(const boost::filesystem::path &root);
    ~TmpTileset();

    void store(const vts::TileId &tileId, const vts::Mesh &mesh
               , const Atlas &atlas);

    std::tuple<Mesh::pointer, opencv::HybridAtlas::pointer>
    load(const vts::TileId &tileId);

    /** Flushes data to disk.
     */
    void flush();

    /** Returns dataset tileindex (union of all tileindices).
     */
    TileIndex tileIndex() const;

private:
    class Slice;

    boost::filesystem::path root_;

    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<Slice>> slices_;
};

} } } // namespace vadstena::vts::tools

#endif // vts_tools_tmptileset_hpp_included


