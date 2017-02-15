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

inline math::Point2d normalize(const imgproc::UVCoord &uv
                               , const math::Size2 &texSize)
{
    return { uv.x / texSize.width
            , 1.0 - uv.y / texSize.height };
}

class TextureInfo {
public:
    TextureInfo(const SubMesh &sm, const cv::Mat &texture)
        : tc_(denormalize(sm.tc, texture.size()))
        , faces_(sm.facesTc), texture_(texture)
    {}

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

    typedef std::vector<TextureInfo> list;

    const Face& face(int faceIndex) const {
        return (faces_)[faceIndex];
    }

    Face& ncface(int faceIndex) {
        return (faces_)[faceIndex];
    }

private:
    math::Points2d tc_;
    Faces faces_;
    cv::Mat texture_;
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

    typedef std::shared_ptr<Component> pointer;
    typedef std::vector<pointer> list;
    typedef std::set<pointer> set;

    Component() {}

    Component(int findex, const Face &face, const TextureInfo &tx)
        : faces{findex}, indices{face(0), face(1), face(2)}
    {
        rect.update(tx.uvCoord(face, 0));
        rect.update(tx.uvCoord(face, 1));
        rect.update(tx.uvCoord(face, 2));
    }

    void add(int findex, const Face &face, const TextureInfo &tx) {
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

    void copy(cv::Mat &tex, const cv::Mat &texture) const;

    imgproc::UVCoord adjustUV(const math::Point2 &p) const {
        imgproc::UVCoord c(p(0), p(1));
        rect.adjustUV(c);
        return c;
    }
};

void Component::copy(cv::Mat &tex, const cv::Mat &texture) const
{
    cv::Mat dst(tex, cv::Rect(rect.packX, rect.packY
                              , rect.width(), rect.height()));
    cv::Mat(texture, cv::Rect(rect.x(), rect.y(), rect.width(), rect.height()))
        .copyTo(dst);
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
            c->copy(tex, texture);
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
                 = std::make_shared<Component>(i, face, tx));
        }
    }

    // map tc
    for (const auto &c : components) {
        for (const auto &index : c->indices) {
            tcMap[index] = c;
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

} } } // namespace vtslibs::vts::tools
