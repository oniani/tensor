add_library(Catch2 INTERFACE)
target_include_directories(Catch2 INTERFACE ../third_party)

add_executable(tests test.cpp)
target_link_libraries(tests Catch2)
