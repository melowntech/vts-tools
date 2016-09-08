#include <cstdlib>
#include <string>

#include "dbglog/dbglog.hpp"

#include "geometry/mesh.hpp"
#include "geometry/meshop.hpp"

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"

#include "service/cmdline.hpp"

#include "../vts-libs/vts.hpp"
#include "../vts-libs/registry/po.hpp"

#include "../tinyxml2/tinyxml2.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

#include <opencv2/highgui/highgui.hpp>

namespace vs = vadstena::storage;
namespace vr = vadstena::registry;
namespace vts = vadstena::vts;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace xml = tinyxml2;

namespace {

//// LodTreeExport.xml parse ///////////////////////////////////////////////////

struct LodTreeNode
{
    double radius, minRange;
    math::Point3 center;
    fs::path modelPath;
    std::vector<LodTreeNode> children;

    LodTreeNode(xml::XMLElement *elem, const fs::path &dir);
};

struct LodTreeExport
{
    std::string srs;
    math::Point3 origin;
    std::vector<LodTreeNode> blocks;

    LodTreeExport(const fs::path &xmlPath);
};


xml::XMLElement* getElement(xml::XMLNode *node, const char* elemName)
{
    xml::XMLElement* elem = node->FirstChildElement(elemName);
    if (!elem) {
        LOGTHROW(err3, std::runtime_error)
            << "XML element \"" << elemName << "\" not found.";
    }
    return elem;
}

void errorAttrNotFound(xml::XMLElement *elem, const char* attrName)
{
    LOGTHROW(err3, std::runtime_error)
        << "XML attribute \"" << attrName
        << "\" not found in element \"" << elem->Name() << "\".";
}

const char* getTextAttr(xml::XMLElement *elem, const char* attrName)
{
    const char* text = elem->Attribute(attrName);
    if (!text) {
        errorAttrNotFound(elem, attrName);
    }
    return text;
}

double getDoubleAttr(xml::XMLElement *elem, const char* attrName)
{
    double a;
    if (elem->QueryDoubleAttribute(attrName, &a) == xml::XML_NO_ATTRIBUTE) {
        errorAttrNotFound(elem, attrName);
    }
    return a;
}

xml::XMLElement* loadLodTreeXml(const fs::path &fname, xml::XMLDocument &doc)
{
    auto err = doc.LoadFile(fname.native().c_str());
    if (err != xml::XML_SUCCESS) {
        LOGTHROW(err3, std::runtime_error)
            << "Error loading " << fname << ": " << doc.ErrorName();
    }

    auto *root = getElement(&doc, "LODTreeExport");

    double version = getDoubleAttr(root, "version");
    if (version > 1.1 + 1e-12) {
        LOGTHROW(err3, std::runtime_error)
            << fname << ": unsupported format version (" << version << ").";
    }

    return root;
}


LodTreeNode::LodTreeNode(tinyxml2::XMLElement *node, const fs::path &dir)
{
    int ok = xml::XML_SUCCESS;
    if (getElement(node, "Radius")->QueryDoubleText(&radius) != ok ||
        getElement(node, "MinRange")->QueryDoubleText(&minRange) != ok)
    {
        LOGTHROW(err3, std::runtime_error) << "Error reading node data";
    }

    auto *ctr = getElement(node, "Center");
    center(0) = getDoubleAttr(ctr, "x");
    center(1) = getDoubleAttr(ctr, "y");
    center(2) = getDoubleAttr(ctr, "z");

    modelPath = dir / getElement(node, "ModelPath")->GetText();

    // load all children
    std::string strNode("Node");
    for (auto *elem = node->FirstChildElement();
         elem;
         elem = elem->NextSiblingElement())
    {
        if (strNode == elem->Name())
        {
            children.emplace_back(elem, dir);
        }
    }
}


LodTreeExport::LodTreeExport(const fs::path &xmlPath)
{
    xml::XMLDocument doc;
    auto *root = loadLodTreeXml(xmlPath, doc);

    srs = getElement(root, "SRS")->GetText();

    auto *local = getElement(root, "Local");
    origin(0) = getDoubleAttr(local, "x");
    origin(1) = getDoubleAttr(local, "y");
    origin(2) = getDoubleAttr(local, "z");

    // load all blocks ("Tiles")
    std::string strTile("Tile");
    for (auto *elem = root->FirstChildElement();
         elem;
         elem = elem->NextSiblingElement())
    {
        if (strTile == elem->Name())
        {
            fs::path path(getTextAttr(elem, "path"));
            if (path.is_relative()) {
                path = xmlPath.parent_path() / path;
            }
            LOG(info3) << "Parsing block " << path << ".";

            xml::XMLDocument tileDoc;
            auto *tileRoot = loadLodTreeXml(path, tileDoc);
            auto *rootNode = getElement(tileRoot, "Tile");

            blocks.emplace_back(rootNode, path.parent_path());
        }
    }
}


//// utility main //////////////////////////////////////////////////////////////

class LodTree2Vts : public service::Cmdline
{
public:
    LodTree2Vts()
        : service::Cmdline("lodtree2vts", BUILD_TARGET_VERSION)
    {
    }

private:
    virtual void configuration(po::options_description &cmdline
                               , po::options_description &config
                               , po::positional_options_description &pd)
        UTILITY_OVERRIDE;

    virtual void configure(const po::variables_map &vars)
        UTILITY_OVERRIDE;

    virtual bool help(std::ostream &out, const std::string &what) const
        UTILITY_OVERRIDE;

    virtual int run() UTILITY_OVERRIDE;

    fs::path input_;
    fs::path output_;
};

void LodTree2Vts::configuration(po::options_description &cmdline
                               , po::options_description &config
                               , po::positional_options_description &pd)
{
    vr::registryConfiguration(cmdline, vr::defaultPath());

    cmdline.add_options()
        ("input", po::value(&input_)->required()
         , "Path to LODTreeExport.xml input file.")
        ("output", po::value(&output_)->required()
         , "Path to output (vts) tile set.")
        ("overwrite", "Existing tile set gets overwritten if set.")
        ;

    pd.add("input", 1);
    pd.add("output", 1);

    (void) config;
}

void LodTree2Vts::configure(const po::variables_map &vars)
{
    vr::registryConfigure(vars);

}

bool LodTree2Vts::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(lodtree2vts
usage
    lodtree2vts INPUT OUTPUT [OPTIONS]

)RAW";
    }
    return false;
}


// LodTreeNodes stratified into levels (by tree depth)
typedef std::vector<std::vector<const LodTreeNode*> > Levels;

void getLevelsRecursive(const LodTreeNode &node, Levels &levels, unsigned depth)
{
    while (levels.size() <= depth) {
        levels.emplace_back();
    }
    levels[depth].push_back(&node);

    for (const auto &ch : node.children) {
        getLevelsRecursive(ch, levels, depth+1);
    }
}

Levels getLevels(const LodTreeExport &lte)
{
    Levels levels;
    for (const auto& block : lte.blocks) {
        getLevelsRecursive(block, levels, 0);
    }
    return levels;
}

math::Point3 point3(const aiVector3D &vec)
{
    return {vec.x, vec.y, vec.z};
}

double calcTexArea(const aiMesh* mesh)
{
    double area = 0.0;
    if (mesh->GetNumUVChannels())
    {
        for (unsigned f = 0; f < mesh->mNumFaces; f++)
        {
            aiFace &face = mesh->mFaces[f];
            assert(face.mNumIndices == 3);

            math::Point3 a(point3(mesh->mTextureCoords[0][face.mIndices[0]]));
            math::Point3 b(point3(mesh->mTextureCoords[0][face.mIndices[1]]));
            math::Point3 c(point3(mesh->mTextureCoords[0][face.mIndices[2]]));

            area += 0.5*norm_2(math::crossProduct(b - a, c - a));
        }
    }
    return area;
}

int LodTree2Vts::run()
{
    LodTreeExport lte(input_);

    Levels levels(getLevels(lte));

    /*for (unsigned i = 0; i < levels.size(); i++)
    {
        double texArea = 0.0;
        for (const LodTreeNode* node : levels[i])
        {
            LOG(info2) << "Loading model " << node->modelPath;

            Assimp::Importer imp;
            const aiScene *scene = imp.ReadFile(node->modelPath.native(), 0);

            for (unsigned m = 0; m < scene->mNumMeshes; m++)
            {
                aiMesh *mesh = scene->mMeshes[m];
                aiMaterial *mat = scene->mMaterials[mesh->mMaterialIndex];

                aiString texFile;
                mat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), texFile);
                fs::path texPath(node->modelPath.parent_path() / texFile.C_Str());

                cv::Mat image(cv::imread(texPath.native()));
                double texSize = image.rows * image.cols;

                //LOG(info3) << texPath << " " << image.cols << "x" << image.rows;

                texArea += calcTexArea(mesh) * texSize;
            }
        }

        LOG(info3) << "Level " << i+13 << ": area " << texArea;
    }*/


    int l = 0;
    for (const auto &level : levels)
    {
        geometry::Mesh levelMesh;
        std::vector<int> faceValue;

        int modelCount = 0;
        for (const LodTreeNode* node : level)
        {
            LOG(info2) << "Loading model " << node->modelPath;

            Assimp::Importer imp;
            const aiScene *scene = imp.ReadFile(node->modelPath.native(), 0);

            for (unsigned m = 0; m < scene->mNumMeshes; m++)
            {
                aiMesh *mesh = scene->mMeshes[m];

                unsigned base = levelMesh.vertices.size();
                for (unsigned i = 0; i < mesh->mNumVertices; i++) {
                    auto vert(point3(mesh->mVertices[i]));
                    levelMesh.vertices.push_back(vert + node->center);
                }

                for (unsigned i = 0; i < mesh->mNumFaces; i++) {
                    aiFace *f = &(mesh->mFaces[i]);
                    assert(f->mNumIndices == 3);
                    levelMesh.faces.emplace_back(
                        base + f->mIndices[0],
                        base + f->mIndices[1],
                        base + f->mIndices[2]);

                    faceValue.push_back(modelCount);
                }
            }

            modelCount++;
        }

        char name[100];
        sprintf(name, "level%02d.ply", (l++) + 13);

#if 1
        LOG(info3) << "Writing " << name;

        std::ofstream out(name);
        out.setf(std::ios::scientific, std::ios::floatfield);

        out << "ply\n"
            << "format ascii 1.0\n"
            << "element vertex " << levelMesh.vertices.size() << '\n'
            << "property float x\n"
            << "property float y\n"
            << "property float z\n"
            << "element face " << levelMesh.faces.size() << '\n'
            << "property list uchar int vertex_indices\n"
            << "property uchar red\n"
            << "property uchar green\n"
            << "property uchar blue\n"
            << "end_header\n";

        for (const auto &vertex : levelMesh.vertices)
        {
            out << vertex(0) << ' ' << vertex(1) << ' '  << vertex(2) << '\n';
        }

        unsigned fidx = 0;
        for (const auto &face : levelMesh.faces)
        {
            out << "3 " << face.a << ' ' << face.b << ' ' << face.c << ' ';
            srand(faceValue[fidx++]+1);
            out << rand()%256 << ' ' << rand()%256 << ' ' << rand()%256 << '\n';
        }

#else
        geometry::saveAsPly(levelMesh, name);
#endif
    }

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}


} // namespace


int main(int argc, char *argv[])
{
    return LodTree2Vts()(argc, argv);
}
