cmake_minimum_required(VERSION 3.10)
project(so101_description)

find_package(ament_cmake REQUIRED)

# Data-only package: copy these folders to install/share/so101_description/
install(DIRECTORY urdf meshes launch rviz
  DESTINATION share/${PROJECT_NAME})

ament_package()
