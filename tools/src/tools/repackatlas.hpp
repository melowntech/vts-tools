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

#include <vector>

#include "math/geometry_core.hpp"

#include "vts-libs/vts/types.hpp"
#include "vts-libs/vts/mesh.hpp"
#include "vts-libs/vts/opencv/atlas.hpp"

namespace vtslibs { namespace vts { namespace tools {

struct TextureRegionInfo {
    typedef std::vector<int> Faces;

    struct Region {
        math::Extents2 region;
        math::Size2f size;

        typedef std::vector<Region> list;

        Region(const math::Extents2 &region)
            : region(region), size(math::size(region))
        {}
    };

    Region::list regions;
    Faces faces;

    TextureRegionInfo() = default;
    TextureRegionInfo(const Region::list &regions)
        : regions(regions)
    {}

    typedef std::vector<TextureRegionInfo> list;
};

void repack(const TileId &tileId, Mesh &mesh, opencv::Atlas &atlas);

void repack(const TileId &tileId, Mesh &mesh, opencv::Atlas &atlas
            , const TextureRegionInfo::list &textureRegions);

} } } // namespace vtslibs::vts::tools
