cmake_minimum_required(VERSION 3.10)
project(so101_nodes_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(joint_wave src/joint_wave.cpp)
ament_target_dependencies(joint_wave rclcpp sensor_msgs)

# ros2 run looks for executables in lib/<package>/
install(TARGETS joint_wave
  DESTINATION lib/${PROJECT_NAME})

ament_package()
