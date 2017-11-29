#include "dbglog/dbglog.hpp"

#include "tilemapping.hpp"

namespace vs = vtslibs::storage;
namespace vr = vtslibs::registry;
namespace vts = vtslibs::vts;

namespace vtslibs { namespace vts { namespace tools {

vts::TileRange::point_type
tiled(const math::Size2f &ts, const math::Point2 &origin
      , const math::Point2 &p)
{
    math::Point2 local(p - origin);
    return vts::TileRange::point_type(local(0) / ts.width
                                      , -local(1) / ts.height);
}

vts::TileRange tileRange(const vr::ReferenceFrame::Division::Node &node
                         , vts::Lod localLod, const math::Points2 &points
                         , double margin)
{
    const auto ts(vts::tileSize(node.extents, localLod));
    // NB: origin is in upper-left corner and Y grows down
    const auto origin(math::ul(node.extents));

    math::Size2f isize(ts.width * margin, ts.height * margin);
    std::array<math::Point2, 4> inflates{{
            { -isize.width, +isize.height }
            , { +isize.width, +isize.height }
            , { +isize.width, -isize.height }
            , { -isize.width, -isize.height }
        }};

    vts::TileRange r(math::InvalidExtents{});

    for (const auto &p : points) {
        for (const auto &inflate : inflates) {
            update(r, tiled(ts, origin, p + inflate));
        }
    }

    return vts::global(node.id, localLod, r);
}

template <typename Op>
void forEachTile(const vr::ReferenceFrame &referenceFrame
                 , vts::Lod lod, const vts::TileRange &tileRange
                 , Op op)
{
    typedef vts::TileRange::value_type Index;
    for (Index j(tileRange.ll(1)), je(tileRange.ur(1)); j <= je; ++j) {
        for (Index i(tileRange.ll(0)), ie(tileRange.ur(0)); i <= ie; ++i) {
            op(vts::NodeInfo(referenceFrame, vts::TileId(lod, i, j)));
        }
    }
}

template <typename Op>
void rasterizeTiles(const vr::ReferenceFrame &referenceFrame
                    , const vr::ReferenceFrame::Division::Node &rootNode
                    , vts::Lod lod, const vts::TileRange &tileRange
                    , Op op)
{
    // process tile range
    forEachTile(referenceFrame, lod, tileRange
                , [&](const vts::NodeInfo &ni)
    {
        LOG(info1)
            << std::fixed << "dst tile: "
            << ni.nodeId() << ", " << ni.extents();

        // TODO: check for incidence with Q; NB: clip margin must be taken into
        // account

        // check for root
        if (ni.subtree().root().id == rootNode.id) {
            op(vts::tileId(ni.nodeId()));
        }
    });
}

void TileMapping::rasterizeTile(const InputTile &tile,
                                const registry::ReferenceFrame &dstRf,
                                double margin)
{
    // for each destination division node
    for (const auto &item : dstRf.division.nodes) {
        const auto &node(item.second);
        if (!node.real()) { continue; }

        auto dstCorners(tile.projectCorners(node));

        // ignore tiles that cannot be transformed
        if (dstCorners.empty()) { continue; }

        vts::Lod dstLocalLod(tile.dstLod - node.id.lod);

        // generate tile range from bounding box of projected corners
        auto range(tileRange(node, dstLocalLod, dstCorners, margin));
        LOG(info1) << "tile range: " << range;

        // TODO: add margin
        rasterizeTiles(dstRf, node, tile.dstLod, range
                       , [&](const vts::TileId &id)
        {
            UTILITY_OMP(critical)
            {
                sourceInfo_[id].push_back(tile.id);
                dstTi_.set(id, 1);
            }
        });
    }
}

// keep empty, used as placeholder!
const std::vector<int> TileMapping::emptySource_;

} } } // namespace vtslibs::vts::tools
