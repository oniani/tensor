cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(tensor VERSION 1.0 LANGUAGES CXX)

add_executable(${PROJECT_NAME} main.cxx)

file(GLOB_RECURSE FILES *.cxx *.hxx)

find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
  add_custom_target(
    clang-format
    COMMAND ${CLANG_FORMAT} -i -style='{BasedOnStyle: Google, IndentWidth: 4}' ${FILES}
  )
endif()

find_program(CLANG_TIDY clang-tidy)
if(CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,readability-*)
  add_custom_target(clang-tidy COMMAND ${CMAKE_CXX_CLANG_TIDY} ${FILES})
endif()

find_program(CPP_CHECK cppcheck)
if(CPP_CHECK)
  add_custom_target(
    cppcheck
    COMMAND ${CPP_CHECK}
    --enable=information,missingInclude,performance,portability,warning
    --std=c++20
    --template="[{severity}][{id}] {message} {callstack} \(at {file}:{line}\)"
    --verbose
    ${FILES}
  )
endif()
