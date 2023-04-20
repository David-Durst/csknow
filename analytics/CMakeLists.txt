cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,leak,undefined -fno-omit-frame-pointer -fsanitize-address-use-after-scope")
#set(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-all")
#set(CMAKE_VERBOSE_MAKEFILE ON)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

project (csknow VERSION "0.1")
include(GNUInstallDirs)
include(FetchContent)
find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(glm INTERFACE)
target_include_directories(glm SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/external/glm")

set(HIGHFIVE_USE_BOOST OFF)
set(HIGHFIVE_UNIT_TESTS OFF)
set(HIGHFIVE_UNIT_TESTS OFF CACHE STRING "")
set(HIGHFIVE_BUILD_DOCS OFF)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/HighFive" EXCLUDE_FROM_ALL)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/libtorch")
foreach(path ${CMAKE_PREFIX_PATH})
    message("Path: " ${path})
endforeach(path)
find_package(Torch REQUIRED)

find_program(GZIP_LOCATION gzip DOC "Path to gzip program required.")
if(GZIP_LOCATION)
    set(UNZIP_COMMAND ${GZIP_LOCATION} -dc)
else()
    #cause the searches to be repeated
    unset(GZIP_LOCATION CACHE)
    if(WIN32)
        message(SEND_ERROR "gzip is required, but was not found.  Please install gzip and set GZIP_LOCATION to the gzip.")
    else()
        message(SEND_ERROR "gzip is required, but was not found.  Please install gzip.")
    endif()
endif()

find_program(TAR_LOCATION tar DOC "Path to tar program required.")
if(NOT TAR_LOCATION)
    #cause the searches to be repeated
    unset(TAR_LOCATION CACHE)
    if(WIN32)
        message(SEND_ERROR "tar is required, but was not found.  Please install tar and set TAR_LOCATION to the gzip.")
    else()
        message(SEND_ERROR "tar is required, but was not found.  Please install tar.")
    endif()
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-function -Wno-unknown-pragmas)

include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE CSKNOW_SOURCES ${PROJECT_SOURCE_DIR}/src/lib/*.cpp)
list(FILTER CSKNOW_SOURCES EXCLUDE REGEX "\\.#.*")
file(GLOB_RECURSE CSKNOW_HEADERS ${PROJECT_SOURCE_DIR}/include/*.h)


# from OpenCV via https://medium.com/@onur.dundar1/cmake-tutorial-585dd180109b
# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    message(STATUS "CMAKE_SOURCE_DIR is ${CMAKE_SOURCE_DIR}")
    message(STATUS "CMAKE_BINARY_DIR is ${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

add_library(csknow_lib SHARED ${CSKNOW_SOURCES} ${CSKNOW_HEADERS})
add_executable(csknow ${PROJECT_SOURCE_DIR}/src/bin/csknow_main.cpp)
add_executable(csknow_test ${PROJECT_SOURCE_DIR}/src/bin/csknow_test.cpp)
#add_executable(csknow_bot ${PROJECT_SOURCE_DIR}/src/bin/csknow_bot.cpp)
add_executable(csknow_bt_bot ${PROJECT_SOURCE_DIR}/src/bin/csknow_bt_bot.cpp)
add_executable(csknow_test_bt_bot ${PROJECT_SOURCE_DIR}/src/bin/csknow_test_bt_bot.cpp)
add_executable(csknow_bt_io ${PROJECT_SOURCE_DIR}/src/bin/csknow_bt_io.cpp)
add_executable(csknow_compute_vis_points ${PROJECT_SOURCE_DIR}/src/bin/csknow_compute_vis_points.cpp)
add_executable(csknow_create_train_datasets ${PROJECT_SOURCE_DIR}/src/bin/csknow_create_train_datasets.cpp)
#add_executable(csknow_train ${PROJECT_SOURCE_DIR}/src/bin/csknow_train.cpp)
#add_executable(csknow_intersection ${PROJECT_SOURCE_DIR}/src/bin/csknow_intersection.cpp)
target_link_libraries(csknow_lib OpenMP::OpenMP_CXX)
target_link_libraries(csknow_lib OpenSSL::SSL)
target_link_libraries(csknow_lib "${TORCH_LIBRARIES}")
target_link_libraries(csknow_lib ${OpenCV_LIBS})
target_link_libraries(csknow_lib glm)
target_link_libraries(csknow_lib HighFive)
target_link_libraries(csknow csknow_lib)
target_link_libraries(csknow_test csknow_lib)
#target_link_libraries(csknow_bot csknow_lib)
target_link_libraries(csknow_bt_bot csknow_lib)
target_link_libraries(csknow_test_bt_bot csknow_lib)
target_link_libraries(csknow_bt_io csknow_lib)
target_link_libraries(csknow_compute_vis_points csknow_lib)
target_link_libraries(csknow_create_train_datasets csknow_lib)
#target_link_libraries(csknow_train csknow_lib)
#target_link_libraries(csknow_intersection csknow_lib)

# add this to enable debug printing statements

set_target_properties(csknow_lib PROPERTIES VERSION ${PROJECT_VERSION})

# add the install targets
install (TARGETS csknow DESTINATION bin)
#install (TARGETS csknow_intersection DESTINATION bin)
install (TARGETS csknow_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDERDIR})

