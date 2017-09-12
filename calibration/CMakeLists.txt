cmake_minimum_required(VERSION 2.8.3)
project(calibration_app)

add_definitions(-std=c++1y)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV QUIET)
find_package(Boost REQUIRED filesystem system)
find_package(Eigen3 REQUIRED)

include_directories(include)
include_directories("${EIGEN3_INCLUDE_DIR}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

###########
## Build ##
###########

add_executable(calibration_app
    src/fisheye_calib.cpp
    src/fisheye.cpp
)
target_link_libraries(calibration_app
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
)

add_executable(video_convert_image src/video_convert_image.cpp)
target_link_libraries(video_convert_image ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(movie_capture src/movie_capture.cpp)
target_link_libraries(movie_capture ${OpenCV_LIBS})

add_executable(estimate_external_matrix_one_camera src/estimate_external_matrix.cpp)
target_link_libraries(estimate_external_matrix_one_camera ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(calibration
	src/calibration.cpp
	src/fisheye_calibration.cpp
	src/fisheye.cpp)

target_link_libraries(calibration ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(estimate_external_matrix src/estimate_external_matrix.cpp)
target_link_libraries(estimate_external_matrix ${Boost_LIBRARIES} ${OpenCV_LIBS})