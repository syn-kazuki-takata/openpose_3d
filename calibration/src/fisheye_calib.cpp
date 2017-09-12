#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "fisheye.hpp"

// reads images in a directory
std::vector<cv::Mat> read_images(const std::string& directory_path) {
    std::vector<cv::Mat> images;

    boost::filesystem::directory_iterator dir(directory_path);
    boost::filesystem::directory_iterator end;

    for(dir; dir != end; dir++) {
        std::cout << dir->path().string() << std::endl;
        cv::Mat image = cv::imread(dir->path().string());
        if(image.data) {
            images.push_back(image);
        } else {
            std::cerr << "couldn't read the image file!!" << dir->path().string() << std::endl;
        }
    }

    return images;
}

// finds chessboard corners on an image
std::vector<cv::Point2f> find_chessboard_corners(const cv::Mat& image, int rows, int cols) {
    std::vector<cv::Point2f> corners;

    cv::Mat gray;
    cv::cvtColor(image, gray, CV_BGR2GRAY);

    bool found = cv::findChessboardCorners(gray, cv::Size(cols, rows), corners);
    if(!found || corners.size() != rows * cols) {
        std::cerr << "failed to find the corners!!" << std::endl;
        return std::vector<cv::Point2f>();
    }

    cv::find4QuadCornerSubpix(gray, corners, cv::Size(3, 3));
    cv::Mat canvas = image.clone();
    cv::drawChessboardCorners(canvas, cv::Size(cols, rows), corners, found);
    cv::imshow("image", canvas);
    cv::waitKey(1);

    return corners;
}

// undistorts an fisheye image
// @ref http://stackoverflow.com/questions/38983164/opencv-fisheye-undistort-issues
cv::Mat undistort(const cv::Mat& K, const cv::Mat& D, const cv::Mat& frame) {
    cv::Mat newK, map1, map2;
    cv::Mat rview(frame.size(), frame.type());
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, frame.size(), cv::Matx33d::eye(), newK, 1);
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Matx33d::eye(), newK, frame.size(), CV_16SC2, map1, map2);
    cv::remap(frame, rview, map1, map2, cv::INTER_LINEAR);
    return rview;
}

// main
int main(int argc, char* argv[]) {
    // chessboard parameters
    const int chess_rows = 6;
    const int chess_cols = 9;
    const double chess_size = 21.5 / 1000.0;

    // gradient decent parameter
    // carefully choose a GOOD value between 0.0 and 1.0 (OpenCV default = 0.4)
    const double alpha_smooth = 0.4;

    std::cout << "reading images..." << std::endl;
    const auto& images = read_images(argv[1]);

    std::cout << "detecting the chessboard corners..." << std::endl;
    std::cout << "# of images : " << images.size() << std::endl;
    std::vector<std::vector<cv::Point2f>> image_points;
    for(const auto& image : images) {
        auto corners = find_chessboard_corners(image, chess_rows, chess_cols);
        if(!corners.empty()){
            image_points.push_back(corners);
        }
    }

    std::cout << "calculating the point coordinates in the object space..." << std::endl;
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<cv::Point3f> points;
    for(int i=0; i<chess_rows; i++) {
        for(int j=0; j<chess_cols; j++) {
            points.push_back(cv::Point3f(i*chess_size, j*chess_size, 0.0f));
        }
    }
    object_points.assign(image_points.size(), points);

    std::cout << "calibrating..." << std::endl;
    cv::Mat intrinsic, distortion;
    cv::Mat rvecs, tvecs;

    std::cout << "STEP 1: intrinsic estimation" << std::endl;
    cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1024, 1e-9);
    int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_FIX_SKEW |
                cv::fisheye::CALIB_FIX_K1 | cv::fisheye::CALIB_FIX_K2 |
                cv::fisheye::CALIB_FIX_K3 | cv::fisheye::CALIB_FIX_K4;
    //cv::fisheye::calibrate(object_points, image_points, cv::Size(images[0].cols, images[0].rows), intrinsic, distortion, rvecs, tvecs, flags, criteria);
    cv::fisheye::calibrate_(object_points, image_points, cv::Size(images[0].cols, images[0].rows), intrinsic, distortion, rvecs, tvecs, flags, criteria, alpha_smooth);

    std::cout << "--- intrinsic ---\n" << intrinsic << std::endl;
    std::cout << "--- distortion ---\n" << distortion << std::endl;

    std::cout << "STEP 2: intrinsic & distortion estimation" << std::endl;
    flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_USE_INTRINSIC_GUESS | cv::fisheye::CALIB_FIX_SKEW;
    //cv::fisheye::calibrate(object_points, image_points, cv::Size(images[0].cols, images[0].rows), intrinsic, distortion, rvecs, tvecs, flags, criteria);
    cv::fisheye::calibrate_(object_points, image_points, cv::Size(images[0].cols, images[0].rows), intrinsic, distortion, rvecs, tvecs, flags, criteria, alpha_smooth);

    std::cout << "--- intrinsic ---\n" << intrinsic << std::endl;
    std::cout << "--- distortion ---\n" << distortion << std::endl;

    cv::FileStorage fs(argv[2], cv::FileStorage::WRITE);
    fs << "intrinsic" << intrinsic;
    fs << "distortion" << distortion;
    fs.release();

    for(const cv::Mat& image : images) {

        // 歪みを取り除いた画像の生成
        cv::Mat undistorted = undistort(intrinsic, distortion, image);

        // nを押すたびに次の画像に
        while(1){
            cv::imshow("distorted", image);
            cv::imshow("undistorted", undistorted);
            int key = cv::waitKey(0);
            if(key=='n'){
                break;
            }
        }
    }

    return 0;
}