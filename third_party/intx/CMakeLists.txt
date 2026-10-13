# intx: extended precision integer library.
# Copyright 2019 Pawel Bylica.
# Licensed under the Apache License, Version 2.0.

# Minimal build wiring for the vendored, header-only intx snapshot.
# Mirrors the upstream `intx::intx` INTERFACE target so consumers link
# against it exactly as they did with the git submodule / Hunter package.

include(GNUInstallDirs)

set(INTX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(intx INTERFACE)
add_library(intx::intx ALIAS intx)
target_compile_features(intx INTERFACE cxx_std_20)
target_sources(intx INTERFACE $<BUILD_INTERFACE:${INTX_INCLUDE_DIR}/intx/intx.hpp>)
target_include_directories(
  intx INTERFACE $<BUILD_INTERFACE:${INTX_INCLUDE_DIR}>$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# INTX_INSTALL is set by the parent project (e.g. zilkworm) when it wants the
# headers staged into the install tree.
option(INTX_INSTALL "Enable intx install target" OFF)
if(INTX_INSTALL)
  install(DIRECTORY ${INTX_INCLUDE_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
