cmake_minimum_required(VERSION 3.16)
project(so101_fk CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(so101_fk so101_fk.cpp)
target_link_libraries(so101_fk PUBLIC Eigen3::Eigen)
target_compile_options(so101_fk PRIVATE -Wall -Wextra)

add_executable(fk_main fk_main.cpp)
target_link_libraries(fk_main PRIVATE so101_fk)

enable_testing()
add_executable(test_fk test_fk.cpp)
target_link_libraries(test_fk PRIVATE so101_fk)
add_test(NAME so101_fk_test COMMAND test_fk)
