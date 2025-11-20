# evmone: Fast Ethereum Virtual Machine implementation
# Copyright 2019 The evmone Authors.
# SPDX-License-Identifier: Apache-2.0

add_library(evmone.testutils STATIC)
add_library(evmone::testutils ALIAS evmone.testutils)
target_link_libraries(evmone.testutils PUBLIC evmone::state evmc::evmc_cpp nlohmann_json::nlohmann_json)

add_library(testutils STATIC)
add_library(evmone::testutils ALIAS testutils)
target_link_libraries(testutils PUBLIC evmc::evmc_cpp)
target_include_directories(testutils INTERFACE ${PROJECT_SOURCE_DIR} ${evmone_private_include_dir})

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    target_sources(
        testutils PRIVATE
        blob_schedule.hpp
        blob_schedule.cpp
        bytecode.hpp
        stdx/utility.hpp
        utils.hpp
    )
endif()
target_sources(testutils PRIVATE utils.cpp)
