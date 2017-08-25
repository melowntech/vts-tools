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

#include <set>
#include <vector>

#include "utility/expect.hpp"

#include "imgproc/uvpack.hpp"

#include "./repackatlas.hpp"

namespace vtslibs { namespace vts { namespace tools {

inline math::Point2d denormalize(const math::Point2d &p
                                 , const cv::Size &texSize)
{
    return { (p(0) * texSize.width), ((1.0 - p(1)) * texSize.height) };
}

inline math::Points2d denormalize(const math::Points2d &ps
                                  , const cv::Size &texSize)
{
    math::Points2d out;
    for (const auto &p : ps) {
        out.push_back(denormalize(p, texSize));
    }
    return out;
}

inline math::Points2d denormalize(Faces faces
                                  , const math::Points2d &tc
                                  , const cv::Size &texSize
                                  , const TextureRegionInfo &regionInfo)
{
    if (regionInfo.regions.empty()) { return denormalize(tc, texSize); }

    math::Points2d out(tc.size());
    std::vector<char> seen(tc.size(), false);

    const auto &remap([&](int index, const TextureRegionInfo::Region
                          &region) -> void
    {
        auto &iseen(seen[index]);
        if (iseen) { return; }

        // remap from region space to texture space
        auto &itc(tc[index]);
        auto &otc(out[index]);
        otc(0) = itc(0) * region.size.width * texSize.width;
        otc(1) = (1.0 - itc(1)) * region.size.height * texSize.height;

        iseen = true;
    });

    auto ifaceRegion(regionInfo.faces.begin());
    for (const auto &face : faces) {
        const auto &region(regionInfo.regions[*ifaceRegion++]);
        for (const auto &index : face) { remap(index, region); }
    }

    return out;
}

inline math::Point2d normalize(const imgproc::UVCoord &uv
                               , const math::Size2 &texSize)
{
    return { uv.x / texSize.width
            , 1.0 - uv.y / texSize.height };
}

class TextureInfo {
public:
    typedef std::vector<TextureInfo> list;

    TextureInfo(const SubMesh &sm, const cv::Mat &texture)
        : tc_(denormalize(sm.tc, texture.size()))
        , faces_(sm.facesTc), texture_(texture)
    {}

    TextureInfo(const SubMesh &sm, const cv::Mat &texture
                , const TextureRegionInfo &regionInfo)
        : tc_(denormalize(sm.facesTc, sm.tc, texture.size(), regionInfo))
        , faces_(sm.facesTc), texture_(texture)
        , regionInfo_(regionInfo)
    {
        // prepare uv rectangles for regions
        for (const auto &region : regionInfo_.regions) {
            regionRects_.emplace_back();
            auto &r(regionRects_.back());
            const auto &rr(region.region);
            r.update(rr.ll(0) * texture.cols, rr.ll(1) * texture.rows);
            r.update(rr.ur(0) * texture.cols, rr.ur(1) * texture.rows);
        }
    }

    const math::Points2d& tc() const { return tc_; }
    const Faces& faces() const { return faces_; }
    const cv::Mat& texture() const { return texture_; }

    const math::Point2d& uv(const Face &face, int index) const {
        return tc_[face(index)];
    }

    imgproc::UVCoord uvCoord(const Face &face, int index) const {
        const auto &p(uv(face, index));
        return imgproc::UVCoord(p(0), p(1));
    }

    const math::Point2d& uv(int index) const {
        return tc_[index];
    }

    const Face& face(int faceIndex) const {
        return (faces_)[faceIndex];
    }

    int faceRegion(int faceIndex) const {
        return (regionInfo_.regions.empty()
                ? 0 : regionInfo_.faces[faceIndex]);
    }

    const imgproc::UVRect* regionRect(int regionId) const {
        if (regionRects_.empty()) { return nullptr; }
        return &regionRects_[regionId];
    }

private:
    math::Points2d tc_;
    Faces faces_;
    cv::Mat texture_;
    const TextureRegionInfo regionInfo_;

    std::vector<imgproc::UVRect> regionRects_;
};

namespace {

typedef Face::value_type VertexIndex;
typedef std::set<VertexIndex> VertexIndices;

/** Continuous mesh component.
 */
struct Component {
    /** Set of components's indices to faces
     */
    std::set<int> faces;

    /** Set of components's indices to texture coordinates
     */
    VertexIndices indices;

    /** UV rectangle.
     */
    imgproc::UVRect rect;

    /** Region this texturing component belongs to. Defaults to 0 for
     *  non-regioned textures.
     */
    int regionId;

    typedef std::shared_ptr<Component> pointer;
    typedef std::vector<pointer> list;
    typedef std::set<pointer> set;

    Component() {}

    Component(int findex, const Face &face, const TextureInfo &tx
              , int regionId = 0)
        : faces{findex}, indices{face(0), face(1), face(2)}
        , regionId(regionId)
    {
        rect.update(tx.uvCoord(face, 0));
        rect.update(tx.uvCoord(face, 1));
        rect.update(tx.uvCoord(face, 2));
    }

    void add(int findex, const Face &face, const TextureInfo &tx)  {
        faces.insert(findex);
        indices.insert({ face(0), face(1), face(2) });

        rect.update(tx.uvCoord(face, 0));
        rect.update(tx.uvCoord(face, 1));
        rect.update(tx.uvCoord(face, 2));
    }

    void add(const Component &other) {
        faces.insert(other.faces.begin(), other.faces.end());
        indices.insert(other.indices.begin(), other.indices.end());
        rect.update(other.rect.min);
        rect.update(other.rect.max);
    }

    void copy(cv::Mat &tex, const cv::Mat &texture, const TextureInfo &tx)
        const;

    imgproc::UVCoord adjustUV(const math::Point2 &p) const {
        imgproc::UVCoord c(p(0), p(1));
        rect.adjustUV(c);
        return c;
    }
};

namespace {

void clipToLowerBound(int &spos, int &ssize, int &dpos, int &dsize
                      , const char*)
{
    if (spos >= 0) { return; }

    const auto diff(-spos);
    spos = 0;
    ssize -= diff;
    dpos += diff;
    dsize -= diff;
}

void clipToUpperBound(int limit, int &spos, int &ssize, int &dsize
                      , const char*)
{
    const auto send(spos + ssize);
    if (send <= limit) { return; }

    const auto diff(send - limit);
    ssize -= diff;
    dsize -= diff;
}

void copyFromRegion(const imgproc::UVRect &regionRect
                    , const imgproc::UVRect &rect
                    , cv::Mat &tex, const cv::Mat &texture)
{
    (void) regionRect;
    (void) rect;
    (void) tex;
    (void) texture;

    LOG(info4)
        << "About to copy regional patch: src: "
        << rect.width() << "x" << rect.height()
        << " " << rect.x() << " " << rect.y()
        << "; dst: "
        << rect.width() << "x" << rect.height()
        << " " << rect.packX << " " << rect.packY
        << "; region: "
        << regionRect.width() << "x" << regionRect.height()
        << " " << regionRect.x() << " " << regionRect.y()
        ;


    const auto wrap([](int pos, int origin, int size) -> int
    {
        auto mod(pos % size);
        if (mod < 0) { mod += size; }
        return origin + mod;
    });

    const math::Point2i regionOrigin(regionRect.x(), regionRect.y());
    const math::Size2 regionSize(regionRect.width(), regionRect.height());
    const math::Size2 size(rect.width(), rect.height());

    const math::Point2i diff(regionOrigin(0) - rect.x()
                             , regionOrigin(1) - rect.y());

    // copy data
    for (int j(0), je(size.height); j != je; ++j) {
        const auto jsrc(wrap(j - diff(1), regionOrigin(1), regionSize.height));

        if ((jsrc < 0) || (jsrc >= texture.rows)) { continue; }

        for (int i(0), ie(size.width); i != ie; ++i) {
            const auto isrc
                (wrap(i - diff(0), regionOrigin(0), regionSize.width));

            if ((isrc < 0) || (isrc >= texture.cols)) { continue; }

            tex.at<cv::Vec3b>(rect.packY + j, rect.packX + i)
                = texture.at<cv::Vec3b>(jsrc, isrc);
        }
    }
}

} // namespace

void Component::copy(cv::Mat &tex, const cv::Mat &texture
                     , const TextureInfo &tx) const
{
    if (const auto *regionRect = tx.regionRect(regionId)) {
        copyFromRegion(*regionRect, rect, tex, texture);
        return;
    }

    cv::Rect src(rect.x(), rect.y(), rect.width(), rect.height());
    cv::Rect dst(rect.packX, rect.packY, rect.width(), rect.height());

    // clip if source is out of bounds
    clipToLowerBound(src.x, src.width, dst.x, dst.width, "left");
    clipToLowerBound(src.y, src.height, dst.y, dst.height, "top");

    clipToUpperBound(texture.cols, src.x, src.width, dst.width, "right");
    clipToUpperBound(texture.rows, src.y, src.height, dst.height, "bottom");

    if ((src.width <= 0) || (src.height <= 0)) { return; }

    cv::Mat dstPatch(tex, dst);
    cv::Mat(texture, src).copyTo(dstPatch);
}

struct ComponentInfo {
    /** Mesh broken to continuous components.
     */
    Component::set components;

    /** Mapping between texture coordinate index to owning component.
     */
    Component::list tcMap;

    /** Texturing information, i.e. the context of the problem.
     */
    TextureInfo *tx;

    typedef std::vector<ComponentInfo> list;

    ComponentInfo(const TileId &tileId, int id, TextureInfo &tx);

    cv::Mat composeTexture(const math::Size2 &ts) const {
        const auto &texture(tx->texture());
        cv::Mat tex(ts.height, ts.width, texture.type());
        tex = cv::Scalar(0, 0, 0);
        for (const auto &c : components) {
            c->copy(tex, texture, *tx);
        }
        return tex;
    }

    math::Points2d composeTc(const math::Size2 &ts) const {
        math::Points2d tc;
        auto itcMap(tcMap.begin());
        for (const auto &oldUv : tx->tc()) {
            tc.push_back(normalize((*itcMap++)->adjustUV(oldUv), ts));
        }
        return tc;
    }

private:
    TileId tileId_;
    int id_;
};

ComponentInfo::ComponentInfo(const TileId &tileId, int id, TextureInfo &tx)
    : tcMap(tx.tc().size()), tx(&tx), tileId_(tileId), id_(id)
{
    const auto &faces(tx.faces());
    Component::list fMap(tx.faces().size());
    Component::list tMap(tx.tc().size());

    typedef std::array<Component::pointer*, 3> FaceComponents;

    // sorts face per-vertex components by number of faces
    // no components are considered empty
    auto sortBySize([](FaceComponents &fc)
    {
        std::sort(fc.begin(), fc.end()
                  , [](const Component::pointer *l, Component::pointer *r)
                  -> bool
        {
            // prefer non-null
            if (!*l) { return !*r; }
            if (!*r) { return true; }

            // both are non-null; prefer longer
            return ((**l).faces.size() > (**r).faces.size());
        });
    });

    for (std::size_t i(0), e(faces.size()); i != e; ++i) {
        const auto &face(faces[i]);
        const auto regionId(tx.faceRegion(i));

        FaceComponents fc =
            { { &tMap[face(0)], &tMap[face(1)], &tMap[face(2)] } };
        sortBySize(fc);

        auto assign([&](Component::pointer &owner, Component::pointer &owned)
                    -> void
        {
            // no-op
            if (owned == owner) { return; }

            if (!owned) {
                // no component assigned yet
                owned = owner;
                return;
            }

            // forget this component beforehand
            components.erase(owned);

            // grab owned by value to prevent overwrite
            const auto old(*owned);

            // merge components
            owner->add(old);

            // move everything to owner
            for (auto index : old.faces) { fMap[index] = owner; }
            for (auto index : old.indices) { tMap[index] = owner; }
        });

        if (*fc[0] || *fc[1] || *fc[2]) {
            auto &owner(*fc[0]);
            fMap[i] = owner;
            owner->add(i, face, tx);
            assign(owner, *fc[1]);
            assign(owner, *fc[2]);
        } else {
            // create new component
            components.insert
                (fMap[i] = *fc[0] = *fc[1] = *fc[2]
                 = std::make_shared<Component>(i, face, tx, regionId));
        }
    }

    // map tc
    for (auto &c : components) {
        for (const auto &index : c->indices) {
            tcMap[index] = c;
        }

        c->rect.inflate(1.0);

        std::set<int> regions;
        for (const auto &face : c->faces) {
            regions.insert(tx.faceRegion(face));
        }

        if (regions.size() != 1) {
            LOGTHROW(info4, std::runtime_error)
                << "Multiple regions in single component: "
                << utility::join(regions, ",", "-");
        }
    }
}

} // namespace

void repack(const TileId &tileId, vts::Mesh &mesh, opencv::Atlas &atlas)
{
    utility::expect(mesh.submeshes.size() == atlas.size()
                    , "Tile %s: Number of submeshes (%d) is different from "
                    "texture count (%d).", tileId, mesh.submeshes.size()
                    , atlas.size());
    int idx(0);
    for (auto &sm : mesh) {
        // prepare texturing stufff
        TextureInfo tx(sm, atlas.get(idx));
        ComponentInfo cinfo(tileId, idx, tx);

        // pack patches into new atlas
        auto packedSize([&]() -> math::Size2
        {
            // pack the patches
            imgproc::RectPacker packer;
            for (auto &c : cinfo.components) {
                packer.addRect(&c->rect);
            }
            packer.pack();
            return math::Size2(packer.width(), packer.height());
        }());

        atlas.set(idx, cinfo.composeTexture(packedSize));
        sm.tc = cinfo.composeTc(packedSize);

        ++idx;
    }
}

// TODO: implement me
void repack(const TileId &tileId, Mesh &mesh, opencv::Atlas &atlas
            , const TextureRegionInfo::list &textureRegions)
{
    utility::expect(mesh.submeshes.size() == atlas.size()
                    , "Tile %s: Number of submeshes (%d) is different from "
                    "texture count (%d).", tileId, mesh.submeshes.size()
                    , atlas.size());
    utility::expect(mesh.submeshes.size() == atlas.size()
                    , "Tile %s: Number of submeshes (%d) is different from "
                    "texture region info count (%d).", tileId
                    , mesh.submeshes.size(), textureRegions.size());

    int idx(0);
    auto iregions(textureRegions.begin());
    for (auto &sm : mesh) {
        const auto &textureRegion(*iregions++);

        if (!textureRegion.regions.empty()) {
            std::vector<int> tcRegions(sm.tc.size(), -1);
            auto itxFaces(textureRegion.faces.begin());
            for (const auto &f : sm.facesTc) {
                const auto txFace(*itxFaces++);
                for (const auto tc : f) {
                    auto &tcRegion(tcRegions[tc]);
                    if (tcRegion < 0) {
                        tcRegion = txFace;
                    } else if (tcRegion != txFace) {
                        LOG(warn4)
                            << "Texture coordinate " << tc
                            << " belongs to two components: "
                            << tcRegion << " and " << txFace << ".";
                    }
                }
            }
        }

        // prepare texturing stufff
        TextureInfo tx(sm, atlas.get(idx), textureRegion);
        ComponentInfo cinfo(tileId, idx, tx);

        // pack patches into new atlas
        auto packedSize([&]() -> math::Size2
        {
            // pack the patches
            imgproc::RectPacker packer;
            for (auto &c : cinfo.components) {
                packer.addRect(&c->rect);
            }
            packer.pack();
            return math::Size2(packer.width(), packer.height());
        }());

        atlas.set(idx, cinfo.composeTexture(packedSize));
        sm.tc = cinfo.composeTc(packedSize);

        ++idx;
    }
}

} } } // namespace vtslibs::vts::tools
