cmake_minimum_required(VERSION 3.20)
project(rhythm_map_c_example LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(RHYTHM_MAP_LIBRARY_DIR "" CACHE PATH "Directory containing the built Rhythm Map native library")

find_library(
    RHYTHM_MAP_LIBRARY
    NAMES rhythm_map_ffi
    PATHS "${RHYTHM_MAP_LIBRARY_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

add_executable(rhythm_map_c_example main.c)
target_include_directories(rhythm_map_c_example PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../../include")
target_link_libraries(rhythm_map_c_example PRIVATE "${RHYTHM_MAP_LIBRARY}")

if(MSVC)
    target_compile_options(rhythm_map_c_example PRIVATE /W4 /WX)
else()
    target_compile_options(rhythm_map_c_example PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
