#include <cstdlib>
#include <string>

#include "dbglog/dbglog.hpp"

#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"

#include "service/cmdline.hpp"

#include "../vts-libs/vts.hpp"
#include "../vts-libs/registry/po.hpp"

#include "../tinyxml2/tinyxml2.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

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
    LOG(info3) << modelPath;

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
            LOG(info3) << "Loading block " << path << ".";

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
         , "Path to input (vts0) tile set.")
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

/*void testAssImp(const fs::path &path)
{
    Assimp::Importer imp;

    const aiScene *scene = imp.ReadFile(path.native(), 0);

    if (!scene) {
        LOGTHROW(err3, std::runtime_error)
            << "Error loading " << path << ": " << imp.GetErrorString();
    }

    LOG(info2) << "# vertices: " << scene->mMeshes[0]->mNumVertices;
    LOG(info2) << "# faces: " << scene->mMeshes[0]->mNumFaces;
}*/


int LodTree2Vts::run()
{
    //testAssImp("/mnt/media/vadstena/cowi/2016-03-18/Production_2/Data/Tile_+000_+006/Tile_+000_+006_L14_0.dae");

    // all done
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}


} // namespace


int main(int argc, char *argv[])
{
    LodTreeExport lte("/mnt/media/vadstena/cowi/2016-03-18/Production_2/Data/LODTreeExport.xml");

    return LodTree2Vts()(argc, argv);
}
