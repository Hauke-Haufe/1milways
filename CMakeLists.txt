cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA)
project(1MWays)

include(FetchContent)

FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad
    GIT_TAG        v2.0.6
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)

FetchContent_GetProperties(glad)
if(NOT glad_POPULATED)
    message("Fetching glad")
    FetchContent_MakeAvailable(glad)

    add_subdirectory("${glad_SOURCE_DIR}/cmake" glad_cmake)
    glad_add_library(glad REPRODUCIBLE EXCLUDE_FROM_ALL LOADER API gl:core=4.6 EXTENSIONS GL_ARB_bindless_texture GL_EXT_texture_compression_s3tc)
endif()

FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
)
FetchContent_MakeAvailable(glm)

find_package(glfw3 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)


