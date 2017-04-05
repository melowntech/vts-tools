set(common_DEPENDS
  dbglog>=1.4 vts-libs>=2.3 service>=1.6
  roarchive>=1.1
  )

# support library for VTS tools
define_module(LIBRARY vtstools
  DEPENDS ${common_DEPENDS}
  )

set(vtstools_SOURCES
  tmptileset.hpp tmptileset.cpp
  repackatlas.hpp repackatlas.cpp
  )

add_library(vtstools STATIC ${vtstools_SOURCES})
buildsys_library(vtstools)
target_link_libraries(vtstools ${MODULE_LIBRARIES})
buildsys_target_compile_definitions(vtstools ${MODULE_DEFINITIONS})

# ------------------------------------------------------------------------





# ------------------------------------------------------------------------



# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
# vef2vts tool
define_module(BINARY vef2vts
  DEPENDS vtstools ${common_DEPENDS} vef>=1.2)
set(vef2vts_SOURCES
  vef2vts.cpp)

add_executable(vef2vts ${vef2vts_SOURCES})
target_link_libraries(vef2vts ${MODULE_LIBRARIES})
buildsys_target_compile_definitions(vef2vts ${MODULE_DEFINITIONS})
buildsys_binary(vef2vts)

add_executable(tmptscp tmptscp.cpp)
target_link_libraries(tmptscp ${MODULE_LIBRARIES})
buildsys_target_compile_definitions(tmptscp ${MODULE_DEFINITIONS})
buildsys_binary(tmptscp)

# ------------------------------------------------------------------------
# installation

install(TARGETS  vef2vts
  RUNTIME DESTINATION bin COMPONENT vts-tools)
