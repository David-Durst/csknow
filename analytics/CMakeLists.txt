cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fno-omit-frame-pointer -g")
#set(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-all")
#set(CMAKE_VERBOSE_MAKEFILE ON)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

project (csknow VERSION "0.1")
include(GNUInstallDirs)
find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)

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

add_library(csknow_lib STATIC ${CSKNOW_SOURCES} ${CSKNOW_HEADERS})
add_executable(csknow ${PROJECT_SOURCE_DIR}/src/bin/csknow_main.cpp)
add_executable(csknow_bot ${PROJECT_SOURCE_DIR}/src/bin/csknow_bot.cpp)
add_executable(csknow_intersection ${PROJECT_SOURCE_DIR}/src/bin/csknow_intersection.cpp)
target_link_libraries(csknow_lib OpenMP::OpenMP_CXX)
target_link_libraries(csknow_lib OpenSSL::SSL)
target_link_libraries(csknow_lib stdc++fs)
target_link_libraries(csknow csknow_lib)
target_link_libraries(csknow_bot csknow_lib)
target_link_libraries(csknow_intersection csknow_lib)

# add this to enable debug printing statements

set_target_properties(csknow_lib PROPERTIES VERSION ${PROJECT_VERSION})

# add the install targets
install (TARGETS csknow DESTINATION bin)
install (TARGETS csknow_intersection DESTINATION bin)
install (TARGETS csknow_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDERDIR})

