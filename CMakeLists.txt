cmake_minimum_required(VERSION 3.20)
project(metalflat LANGUAGES CXX OBJCXX)

# Standalone, self-contained library — no duhthusam, no third-party deps.
# Just the system C++ runtime + Apple's Metal / MetalPerformanceShaders /
# Foundation frameworks.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- library --------------------------------------------------------
add_library(metalflat STATIC
    src/FlatIndex.mm
    src/GemmDistance.mm
    src/IvfIndex.mm)
target_include_directories(metalflat PUBLIC include)
target_include_directories(metalflat PRIVATE src)
target_compile_options(metalflat PRIVATE -fobjc-arc)
# PUBLIC so the frameworks propagate to anything linking the static lib
# (the bench, and later duhthusam).
target_link_libraries(metalflat PUBLIC
    "-framework Metal"
    "-framework MetalPerformanceShaders"
    "-framework Foundation")

# ---- bench / correctness harness ------------------------------------
add_executable(metalflat_bench bench/bench.mm)
target_link_libraries(metalflat_bench PRIVATE metalflat)
target_compile_options(metalflat_bench PRIVATE -fobjc-arc)
