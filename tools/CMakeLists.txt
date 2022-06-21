# bootstrap build system
cmake_minimum_required(VERSION 3.3)
project(vts-tools)
# by default, this is OSS build (can be overrided by customer machinery)
set(BUILDSYS_DEFAULT_CUSTOMER_NAME oss)
set(BUILDSYS_CPP_STANDARD 14)

include(buildsys/cmake/buildsys.cmake)

enable_OpenMP()

find_package(Boost 1.46 REQUIRED
  COMPONENTS thread program_options filesystem system date_time
             serialization regex chrono iostreams)
link_directories(${Boost_LIBRARY_DIRS})
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

if(${Boost_VERSION_MAJOR}.${Boost_VERSION_MINOR} VERSION_GREATER_EQUAL 1.73)
  # Since Boost.Python and Boost.Iostreams in Boost >=1.73 still include
  # deprecated headers we need to silence compiler
  add_definitions(-DBOOST_BIND_GLOBAL_PLACEHOLDERS
    -DBOOST_ALLOW_DEPRECATED_HEADERS)
endif()

find_package(OpenCV REQUIRED)

find_package(GDAL REQUIRED)
include_directories(SYSTEM ${GDAL_INCLUDE_DIR})
find_package(Proj REQUIRED)
include_directories(${PROJ_INCLUDE_DIR})
find_package(GeographicLib REQUIRED)
include_directories(${GeographicLib_INCLUDE_DIR})

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(CURL REQUIRED)
find_package(magic REQUIRED)
find_package(JsonCPP REQUIRED)
include_directories(${JSONCPP_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# Asset import library for loading mesh files in lodtree2vts
# (get it with 'apt-get install libassimp-dev')
find_package(Assimp REQUIRED)

find_package(LibProc REQUIRED)
include_directories(${LIBPROC_INCLUDE_DIR})

find_package(TinyXML2 REQUIRED)
include_directories(${TINYXML2_INCLUDE_DIR})

# Protobuf: needed by gdal drivers
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

find_package(OpenMesh 2.3.1 REQUIRED)
include_directories(${OPENMESH_INCLUDE_DIR})
link_directories(${OPENMESH_LIBRARY_DIR})

# dependencies
add_subdirectory(src/dbglog)
add_subdirectory(src/utility)
add_subdirectory(src/service)
add_subdirectory(src/math)
add_subdirectory(src/geometry)
add_subdirectory(src/imgproc)
add_subdirectory(src/jsoncpp)
add_subdirectory(src/geo)
add_subdirectory(src/gdal-drivers)
add_subdirectory(src/half)
add_subdirectory(src/http)
add_subdirectory(src/roarchive)
add_subdirectory(src/miniball)

# vts libraries: force build and install of vts-tools
set(vts-libs_vts-install_component vts-tools)
add_subdirectory(src/vts-libs)

# format decoding libraries
add_subdirectory(src/vef)
add_subdirectory(src/slpk)
add_subdirectory(src/lodtree)


if (NOT BUILDSYS_LIMITED_CPP11)
  # these are supported only when building on modern systems
  add_subdirectory(src/gltf)
  add_subdirectory(src/3dtiles)
endif()

# tools
add_subdirectory(src/tools)

# for testing
add_subdirectory(src/utility/tools EXCLUDE_FROM_ALL)
