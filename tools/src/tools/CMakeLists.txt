set(common_DEPENDS
  dbglog>=1.4 vts-libs>=2.10 service>=1.7
  vts-libs-tools-support>=2.10
  )

# ------------------------------------------------------------------------
# vef2vts tool
define_module(BINARY vef2vts
  DEPENDS ${common_DEPENDS} vef>=1.2)
set(vef2vts_SOURCES
  vef2vts.cpp)

add_executable(tmptscp tmptscp.cpp)
target_link_libraries(tmptscp ${MODULE_LIBRARIES})
buildsys_target_compile_definitions(tmptscp ${MODULE_DEFINITIONS})
buildsys_binary(tmptscp)

add_executable(vef2vts ${vef2vts_SOURCES})
target_link_libraries(vef2vts ${MODULE_LIBRARIES})
buildsys_target_compile_definitions(vef2vts ${MODULE_DEFINITIONS})
buildsys_binary(vef2vts)

# ------------------------------------------------------------------------
# lodtree2vts tool
define_module(BINARY lodtree2vts
  DEPENDS ${common_DEPENDS}
  lodtree>=1.1
  TINYXML2
  )

set(lodtree2vts_SOURCES
  lodtree2vts.cpp
)

add_executable(lodtree2vts ${lodtree2vts_SOURCES})
target_link_libraries(lodtree2vts ${MODULE_LIBRARIES})
buildsys_target_compile_definitions(lodtree2vts ${MODULE_DEFINITIONS})
buildsys_binary(lodtree2vts)

# ------------------------------------------------------------------------
# slpk2vts tool
define_module(BINARY slpk2vts
  DEPENDS ${common_DEPENDS} slpk>=1.2)
set(slpk2vts_SOURCES
  slpk2vts.cpp)

add_executable(slpk2vts slpk2vts.cpp)
target_link_libraries(slpk2vts ${MODULE_LIBRARIES})
buildsys_target_compile_definitions(slpk2vts ${MODULE_DEFINITIONS})
buildsys_binary(slpk2vts)

# ------------------------------------------------------------------------
# vef2slpk tool
define_module(BINARY vef2slpk
  DEPENDS ${common_DEPENDS} vef>=1.2 slpk>=1.2)
set(vef2slpk_SOURCES
  vef2slpk.cpp)

add_executable(vef2slpk ${vef2slpk_SOURCES})
target_link_libraries(vef2slpk ${MODULE_LIBRARIES})
buildsys_target_compile_definitions(vef2slpk ${MODULE_DEFINITIONS})
buildsys_binary(vef2slpk)

# ------------------------------------------------------------------------
# installation
install(TARGETS vef2vts lodtree2vts slpk2vts vef2slpk
  RUNTIME DESTINATION bin COMPONENT vts-tools)
