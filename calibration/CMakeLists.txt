cmake_minimum_required(VERSION 2.8.3)
project(calibration)

include(../env.cmake)
set(EXECUTABLE_OUTPUT_PATH ../bin)

file(GLOB exec_src RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
foreach(cppfile ${exec_src})
  string(REGEX REPLACE ".cpp\$" "" out ${cppfile})
  message(STATUS "add: ${out} ${cppfile}")
  add_executable(${out} ${cppfile})
  target_link_libraries(${out} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} freenect2 NiTE2)
endforeach(cppfile)
