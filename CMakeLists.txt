cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  tensor
  VERSION 0.0.1
  DESCRIPTION "A fast, zero-dependency tensor library in C++"
  HOMEPAGE_URL "https://github.com/oniani/tensor"
  LANGUAGES C CXX
)

option(BUILD "Enable building" OFF)
if(BUILD)
  message(STATUS "Enabling building...")
  add_executable(${PROJECT_NAME} main.cpp)
endif()

option(TESTING "Enable testing" ON)
if(TESTING)
  message(STATUS "Enabling testing...")
  add_subdirectory(test)
endif()

file(GLOB_RECURSE CXX_SOURCE_FILES ${PROJECT_SOURCE_DIR}/include/*.hpp)

find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
  message(STATUS "clang-format found!")
  add_custom_target(
    format COMMAND ${CLANG_FORMAT}
    -i
    -style='{BasedOnStyle: Google, ColumnLimit: 100, Language: Cpp}'
    ${CXX_SOURCE_FILES}
  )
endif()

find_program(CLANG_TIDY clang-tidy)
if(CLANG_TIDY)
  message(STATUS "clang-tidy found!")
  add_custom_target(
    tidy COMMAND ${CLANG_TIDY}
    -checks=-*,readability-*
    ${CXX_SOURCE_FILES}
  )
endif()

find_program(CPP_CHECK cppcheck)
if(CPP_CHECK)
  message(STATUS "cppcheck found!")
  add_custom_target(
    check COMMAND ${CPP_CHECK}
    --enable=all
    --std=c++20
    --template="[{severity}][{id}] {message} {callstack} \(at {file}:{line}\)"
    --verbose
    ${CXX_SOURCE_FILES}
  )
endif()
