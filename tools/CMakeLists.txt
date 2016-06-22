# bootstrap build system
cmake_minimum_required(VERSION 2.8.11)
include(buildsys/cmake/buildsys.cmake)

# project
setup_project(vts-tools "${BUILDSYS_PACKAGE_VERSION}")

enable_OpenMP()

find_package(Boost 1.46 REQUIRED
  COMPONENTS thread program_options filesystem system date_time
             serialization regex chrono iostreams)
link_directories(${Boost_LIBRARY_DIRS})
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(OpenCV 2.4 REQUIRED)

find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIR})
find_package(Proj REQUIRED)
include_directories(${PROJ_INCLUDE_DIR})
find_package(GeographicLib REQUIRED)
include_directories(${GEOGRAPHICLIB_INCLUDE_DIR})

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(CURL REQUIRED)
find_package(magic REQUIRED)

find_package(LibProc REQUIRED)
include_directories(${LIBPROC_INCLUDE_DIR})

# dependencies
add_subdirectory(src/dbglog)
add_subdirectory(src/utility)
add_subdirectory(src/math)
add_subdirectory(src/geometry)
add_subdirectory(src/imgproc)
add_subdirectory(src/geo)
add_subdirectory(src/service)
add_subdirectory(src/jsoncpp)
add_subdirectory(src/half)

# vts libraries: force build and install of vts-tools
set(vts-libs_vts-install_component vts-tools)
add_subdirectory(src/vts-libs)
