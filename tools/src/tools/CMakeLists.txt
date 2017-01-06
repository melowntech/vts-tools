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

