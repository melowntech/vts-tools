/**
 * Copyright (c) 2017 Melown Technologies SE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * *  Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * *  Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef vts_tools_analyze_hpp_
#define vts_tools_analyze_hpp_

#include <limits>
#include <cmath>

#include <boost/optional.hpp>

#include "math/geometry_core.hpp"

#include "vts-libs/vts/nodeinfo.hpp"
#include "vts-libs/vts/mesh.hpp"
#include "vts-libs/vts/ntgenerator.hpp"

namespace vtslibs { namespace vts { namespace tools {

struct LodParams {
    math::Extents2 meshExtents;
    vts::Lod lod;

    operator vts::Lod() const { return lod; }

    LodParams(const math::Extents2 &meshExtents = math::Extents2()
              , vts::Lod lod = 0)
        : meshExtents(meshExtents), lod(lod)
    {}
};

struct LodInfo {
    /** Rf subtree root to bottom lod mapping.
     */
    std::map<const vts::NodeInfo*, LodParams> localLods;

    /** Common min tree depth.
     *  This is depth where there are data available in all nodes.
     */
    int topDepth;

    /** Common bottome -- depth where there are all data available.
     */
    int commonBottom;

    /** Max tree depth.
     */
    int bottomDepth;

    LodInfo() : topDepth(), commonBottom(), bottomDepth() {}

    int levelDiff() const { return bottomDepth - topDepth; }
};

double bestLod(const vts::NodeInfo &rfNode
               , const vts::SubMeshArea &area
               , const math::Size2 &optimalTextureSize);

vts::TileRange computeTileRange(const math::Extents2 &nodeExtents
                                , vts::Lod localLod
                                , const math::Extents2 &meshExtents);

void computeNavtileInfo(const vts::NodeInfo &node
                        , const tools::LodParams lodParams
                        , const tools::LodInfo &lodInfo
                        , vts::NtGenerator &ntg
                        , const boost::optional<vts::LodTileRange> &tileExtents
                        , double ntLodPixelSize);

void updateExtents(math::Extents2 &extents, const vts::SubMesh &sm);

void updateExtents(math::Extents2 &extents, const vts::Mesh &mesh);

math::Extents2 computeExtents(const vts::Mesh &mesh);

struct MeshInfo {
    vts::SubMeshArea area;
    math::Extents2 extents;

    typedef std::map<const vts::NodeInfo*, MeshInfo> map;

    MeshInfo() : extents(math::InvalidExtents{}) {}

    operator bool() const { return area.internalTexture; }

    void update(const vts::SubMesh &mesh, const math::Size2 &txSize);

    MeshInfo& operator+=(const MeshInfo &o);
};

// inline implementation

inline double bestLod(const vts::NodeInfo &rfNode
                      , const vts::SubMeshArea &area
                      , const math::Size2 &optimalTextureSize)
{
    const double texelArea(area.mesh / area.internalTexture);

    const auto optimalTileArea
        (math::area(optimalTextureSize) * texelArea);
    const auto optimalTileCount(rfNode.extents().area()
                                / optimalTileArea);
    return (0.5 * std::log2(optimalTileCount));
}

inline void updateExtents(math::Extents2 &extents, const vts::SubMesh &sm)
{
    for (const auto &p : sm.vertices) {
        update(extents, p(0), p(1));
    }
}

inline void updateExtents(math::Extents2 &extents, const vts::Mesh &mesh)
{
    for (const auto &sm : mesh) {
        updateExtents(extents, sm);
    }
}

inline math::Extents2 computeExtents(const vts::Mesh &mesh)
{
    math::Extents2 extents(math::InvalidExtents{});
    updateExtents(extents, mesh);
    return extents;
}

inline void MeshInfo::update(const vts::SubMesh &mesh
                             , const math::Size2 &txSize)
{
    const auto a(vts::area(mesh));
    area.internalTexture += (a.internalTexture * math::area(txSize));
    area.mesh += a.mesh;
    updateExtents(extents, mesh);
}

inline MeshInfo& MeshInfo::operator+=(const MeshInfo &o)
{
    if (!o) { return *this; }

    area.mesh += o.area.mesh;
    area.internalTexture += o.area.internalTexture;
    extents = unite(extents, o.extents);
    return *this;
}

} } } // namespace vtslibs::vts::tools

#endif // vts_tools_analyze_hpp_
