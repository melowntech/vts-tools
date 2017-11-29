#include <unordered_map>

#include "math/transform.hpp"

#include "imgproc/scanconversion.hpp"

#include "importutil.hpp"

namespace vr = vtslibs::registry;
namespace vts = vtslibs::vts;

namespace vtslibs { namespace vts { namespace tools {

vr::ReferenceFrame::Division::Node
findSpatialDivisionNode(const vr::ReferenceFrame &rf,
                        const geo::SrsDefinition &srs,
                        const math::Point3 &point)
{
    vr::ReferenceFrame::Division::Node sdsNode;
    int maxLod(-1);
    for (const auto &pair : rf.division.nodes) {
        const auto &node(pair.second);
        if (!node.real()) { continue; }

        const vts::CsConvertor csconv(srs, node.srs);
        if (math::inside(node.extents, csconv(point))
            && node.id.lod > maxLod)
        {
            sdsNode = node;
            maxLod = node.id.lod;
        }
    }
    if (maxLod < 0) {
        LOGTHROW(err3, std::runtime_error)
            << "Couldn't find reference frame node for " << point;
    }
    return sdsNode;
}


void optimizeMesh(vts::SubMesh &mesh)
{
    auto hash2 = [](const math::Point2 &p) -> std::size_t {
        return p(0)*218943212 + p(1)*168875421;
    };
    auto hash3 = [](const math::Point3 &p) -> std::size_t {
        return p(0)*218943212 + p(1)*168875421 + p(2)*385120205;
    };

    std::unordered_map<math::Point2, int, decltype(hash2)> map2(1024, hash2);
    std::unordered_map<math::Point3, int, decltype(hash3)> map3(1024, hash3);

    // assign unique indices to vertices and texcoords
    for (const auto &pt : mesh.vertices) {
        int &idx(map3[pt]);
        if (!idx) { idx = map3.size(); }
    }
    for (const auto &pt : mesh.tc) {
        int &idx(map2[pt]);
        if (!idx) { idx = map2.size(); }
    }

    // change face indices
    for (auto &f : mesh.faces) {
        for (int i = 0; i < 3; i++) {
            f(i) = map3[mesh.vertices[f(i)]] - 1;
        }
    }
    for (auto &f : mesh.facesTc) {
        for (int i = 0; i < 3; i++) {
            f(i) = map2[mesh.tc[f(i)]] - 1;
        }
    }

    // update vertices, tc
    mesh.vertices.resize(map3.size());
    for (const auto &item : map3) {
        mesh.vertices[item.second - 1] = item.first;
    }
    mesh.tc.resize(map2.size());
    for (const auto &item : map2) {
        mesh.tc[item.second - 1] = item.first;
    }
}


/** Constructs transformation matrix that maps everything in extents into a grid
 *  of defined size so that grid (0, 0) matches the upper-left extents corner
 *  and grid(gridSize.width - 1, gridSize.width - 1) matches the lower-right
 *  extents corner.
 */
inline math::Matrix4 mesh2grid(const math::Extents2 &extents
                              , const math::Size2 &gridSize)
{
    math::Matrix4 trafo(ublas::identity_matrix<double>(4));

    auto es(size(extents));

    // scale to grid
    trafo(0, 0) =  (gridSize.width - 1) / es.width;
    trafo(1, 1) = -(gridSize.height - 1) / es.height;

    // place zero to upper-left corner
    trafo(0, 3) = -trafo(0,0)*extents.ll(0);
    trafo(1, 3) = -trafo(1,1)*extents.ll(1) + (gridSize.height - 1);

    return trafo;
}

template <typename Op>
void rasterizeMesh(const vts::SubMesh &submesh, const math::Matrix4 &trafo
                   , const math::Size2 &rasterSize, Op op)
{
    std::vector<imgproc::Scanline> scanlines;
    cv::Point3f tri[3];
    for (const auto &face : submesh.faces) {
        for (int i : { 0, 1, 2 }) {
            auto p(transform(trafo, submesh.vertices[face(i)]));
            tri[i].x = p(0); tri[i].y = p(1); tri[i].z = p(2);
        }

        scanlines.clear();
        imgproc::scanConvertTriangle(tri, 0, rasterSize.height, scanlines);

        for (const auto &sl : scanlines) {
            imgproc::processScanline(sl, 0, rasterSize.width, op);
        }
    }
}

void generateHeightMap(vts::HeightMap::Accumulator &hma
                       , const vts::TileId &tileId
                       , const vts::SubMesh &submesh
                       , const math::Extents2 &extents)
{
    cv::Mat *hm;
    UTILITY_OMP(critical)
    hm = &hma.tile(tileId);

    // invalid heightmap value (i.e. initial value) is +oo and we take minimum
    // of all rasterized heights in given place
    rasterizeMesh(submesh, mesh2grid(extents, hma.tileSize())
                  , hma.tileSize()
                  , [&](int x, int y, float z)
    {
        auto &value(hm->at<float>(y, x));
        if (z > value) { value = z; }
    });
}


void warpInPlace(const vts::CsConvertor &conv, vts::SubMesh &sm)
{
    // just convert vertices
    for (auto &v : sm.vertices) {
        // convert vertex in-place
        v = conv(v);
    }
}

void warpInPlace(const vts::CsConvertor &conv, vts::Mesh &mesh)
{
    // convert all submeshes
    for (auto &sm : mesh) { warpInPlace(conv, sm); }
}

void shiftInPlace(vts::SubMesh &sm, double zShift)
{
    // just convert vertices
    for (auto &v : sm.vertices) {
        v(2) += zShift;
    }
}

} } } // namespace vtslibs::vts::tools
