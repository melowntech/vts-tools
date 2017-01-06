# support library for VTS tools
define_module(LIBRARY vtstools
  DEPENDS dbglog vts-libs
  )

set(vtstools_SOURCES
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
  DEPENDS vtstools vts-libs vef dbglog service)
set(vef2vts_SOURCES
    vef2vts.cpp
    )

add_executable(vef2vts ${vef2vts_SOURCES})
target_link_libraries(vef2vts ${MODULE_LIBRARIES})
buildsys_target_compile_definitions(vef2vts ${MODULE_DEFINITIONS})
buildsys_binary(vef2vts)

# ------------------------------------------------------------------------
# installation
install(TARGETS  
        RUNTIME DESTINATION bin COMPONENT main)
