/**
 * Tile mapping helper class.
 * \file tilemapping.hpp
 * \author Jakub Cerveny <jakub.cerveny@melown.com>
 * \author Vaclav Blazek <vaclav.blazek@melown.com>
 */

#ifndef vts_tools_tilemapping_hpp_included
#define vts_tools_tilemapping_hpp_included

#include "utility/openmp.hpp"
#include "utility/progress.hpp"

#include "vts-libs/vts.hpp"
#include "vts-libs/vts/csconvertor.hpp"
#include "vts-libs/tools/progress.hpp"

namespace vtslibs { namespace vts { namespace tools {

struct InputTile
{
    int id;     // tile unique identifier in input
    int depth;  // depth in input tree
    int dstLod; // output (vts) LOD

    virtual math::Points2 projectCorners(
        const vtslibs::registry::ReferenceFrame::Division::Node &node) const = 0;

    InputTile(int id, int depth, int dstLod = 0)
        : id(id), depth(depth), dstLod(dstLod) {}

    virtual ~InputTile() {}
};


/** Provides a mapping between an arbitrary list of tiles and the tiles of a
 *  destination reference frame. The constructor projects the input tiles
 *  to the destination RF and stores a list of input tiles for each destination
 *  tile. These lists can then be queried with 'source()'.
 */
class TileMapping : boost::noncopyable
{
public:
    template<typename input_tile>
    TileMapping(const std::vector<input_tile> inputTiles
                , const vtslibs::registry::ReferenceFrame &dstRf
                , double margin
                , vtslibs::tools::ExternalProgress &extProgress)
    {
        extProgress.expect(inputTiles.size());

        utility::Progress progress(inputTiles.size());

        UTILITY_OMP(parallel for)
        for (unsigned i = 0; i < inputTiles.size(); i++)
        {
            rasterizeTile(inputTiles[i], dstRf, margin);

            UTILITY_OMP(critical)
            {
                (++progress).report
                    (utility::Progress::ratio_t(5, 1000)
                     , "Building tile mapping, ");
                ++extProgress;
            }
        }

        // clone dst tile index to valid tree and make it complete
        validTree_ = vts::TileIndex(vts::LodRange(0, dstTi_.maxLod()), &dstTi_);
        validTree_.complete();
    }

    const vts::TileIndex* validTree() const { return &validTree_; }

    /// Get a list of source tiles that project into destination tileId.
    const std::vector<int>& source(const vts::TileId &dstTileId) const
    {
        auto fsourceInfo(sourceInfo_.find(dstTileId));
        if (fsourceInfo == sourceInfo_.end()) { return emptySource_; }
        return fsourceInfo->second;
    }

    std::size_t size() const { return sourceInfo_.size(); }

    std::size_t expectedCount() const { return dstTi_.count(); }

    bool expected(const vts::TileId &dstTileId) const {
        return dstTi_.get(dstTileId);
    }

private:
    std::map<vts::TileId, std::vector<int>> sourceInfo_;
    vts::TileIndex dstTi_;
    vts::TileIndex validTree_;

    static const std::vector<int> emptySource_;

    void rasterizeTile(const InputTile &tile
                       , const vtslibs::registry::ReferenceFrame &dstRf
                       , double margin);
};


} } } // namespace vtslibs::vts::tools

#endif // vts_tools_tilemapping_hpp_included
