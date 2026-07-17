cmake_minimum_required(VERSION 3.8)
project(exo_openarm_bridge)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(exo_to_openarm_can0 src/exo_to_openarm_can0.cpp)
ament_target_dependencies(exo_to_openarm_can0 rclcpp sensor_msgs)
target_link_libraries(exo_to_openarm_can0 openarm_can)

install(TARGETS exo_to_openarm_can0 DESTINATION lib/${PROJECT_NAME})
ament_package()

add_executable(poke_can0 src/poke_can0.cpp)
target_link_libraries(poke_can0 openarm_can)
install(TARGETS poke_can0 DESTINATION lib/${PROJECT_NAME})
