/**
 * Utility functions for import tools
 * \file importutil.hpp
 * \author Jakub Cerveny <jakub.cerveny@melown.com>
 * \author Vaclav Blazek <vaclav.blazek@melown.com>
 */

#ifndef vts_tools_importutil_hpp_included
#define vts_tools_importutil_hpp_included

#include "vts-libs/vts.hpp"
#include "vts-libs/vts/csconvertor.hpp"
#include "vts-libs/vts/heightmap.hpp"

#include "tilemapping.hpp"

namespace vtslibs { namespace vts { namespace tools {

/// Find a SDS node of a RF, where 'point' lies ('point' is defined in 'srs').
vtslibs::registry::ReferenceFrame::Division::Node
findSpatialDivisionNode(const vtslibs::registry::ReferenceFrame &rf,
                        const geo::SrsDefinition &srs,
                        const math::Point3 &point);


/// Remove duplicate vertices and texture coords.
void optimizeMesh(vts::SubMesh &mesh);

/// Rasterize mesh into 'hma'.
void generateHeightMap(vts::HeightMap::Accumulator &hma
                       , const vts::TileId &tileId
                       , const vts::SubMesh &submesh
                       , const math::Extents2 &extents);

/// Apply coordinate conversion to sm.vertices.
void warpInPlace(const vts::CsConvertor &conv, vts::SubMesh &sm);

/// Apply coordinate conversion to all submeshes
void warpInPlace(const vts::CsConvertor &conv, vts::Mesh &mesh);

/// Shift height in all sm.vertices.
void shiftInPlace(vts::SubMesh &sm, double zShift);


struct NavTileParams
{
    vtslibs::storage::LodRange lodRange;
    int sourceLod;
    double sourceLodPixelSize;
};

/** Determine suitable LOD for each tile based on texture resolution,
 *  calculate parameters for navtile (heightmap) generation.
 */
template<typename input_tile>
inline NavTileParams
assignTileLods(std::vector<input_tile> &inputTiles,
               const vtslibs::registry::ReferenceFrame::Division::Node &sdsNode,
               unsigned int ntLodPixelSize)
{
    // calculate texel size of each tree level
    std::vector<double> texelArea;
    {
        std::vector<std::pair<double, double> > areas;
        for (const auto &tile : inputTiles)
        {
            while (size_t(tile.depth) >= areas.size()) {
                areas.emplace_back(0.0, 0.0);
            }
            auto &pair(areas[tile.depth]);
            pair.first += tile.sdsArea;
            pair.second += tile.texArea;
        }
        for (const auto &pair : areas) {
            texelArea.push_back(pair.first / pair.second);
        }
    }

    // print resolutions and warnings
    {
        int level(0);
        for (double ta : texelArea) {
            if (!level) {
                LOG(info3) << "Tree level " << level
                           << ": avg texel area = " << ta;
            }
            else {
                double factor(texelArea[level-1] / ta);
                LOG(info3) << "Tree level " << level
                           << ": avg texel area = " << ta
                           << " (resolution " << sqrt(factor)
                           << " times previous)";

                if (level && factor < 1.0) {
                    LOG(warn3)
                        << "Warning: level " << level << " has smaller "
                           "resolution than previous level. This level should "
                           "be removed (see also --maxLevel).";
                }
                else if (level && factor < 3.9) {
                    LOG(warn3)
                        << "Warning: level " << level << " does not have "
                           "double the resolution of previous level.";
                }
            }
            ++level;
        }
    }

    // LOD assignment
    NavTileParams nt;
    nt.sourceLod = -1;
    nt.sourceLodPixelSize = 1.0;

    int level(0), count(0);
    double avgRootLod(0.);
    for (double txa : texelArea)
    {
        // calculate VTS lod assuming 256^2 optimal texture tiles
        double tileArea = 256*256*txa;
        double tileLod = 0.5*log2(sdsNode.extents.area() / tileArea);
        tileLod += sdsNode.id.lod;

        LOG(info3) << "Tree level " << level << " ~ VTS LOD " << tileLod;

        if (!level || (texelArea[level-1] / txa > 3.0)) // skip bad levels
        {
            avgRootLod += tileLod - level;
            ++count;
        }
        ++level;
    }
    avgRootLod /= count;
    LOG(info2) << "avgRootLod = " << avgRootLod;

    int rootLod(round(avgRootLod));
    LOG(info3) << "Placing tree level 0 at VTS LOD " << rootLod << ".";

    for (auto &tile : inputTiles) {
        tile.dstLod = rootLod + tile.depth;
        nt.sourceLod = std::max(tile.dstLod, nt.sourceLod);
    }

    // determine LOD for heightmap extraction
    level = 0;
    for (double txa : texelArea) {
        if (std::sqrt(txa) <= ntLodPixelSize) { // TODO: correct?
            nt.sourceLod = rootLod + level;
            nt.sourceLodPixelSize = std::sqrt(txa); // FIXME
            break;
        }
        ++level;
    }
    LOG(info3) << "Navtile data will be extracted at LOD " << nt.sourceLod;

    // TODO: is this correct?
    nt.lodRange.min = rootLod;
    nt.lodRange.max = nt.sourceLod;
    LOG(info3) << "Navtile data will be generated in LOD range: "
               << nt.lodRange << ".";

    return nt;
}


} } } // namespace vtslibs::vts::tools

#endif // vts_tools_importutil_hpp_included


