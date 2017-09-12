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

int main(int argc, char* argv[]) {
	// chessboard parameters
    const int chess_rows = 6;
    const int chess_cols = 9;
    const double chess_size = 21.5 / 1000.0;

    //画像群読み出し
    //const auto& images = read_images(argv[1]);
    cv::Mat image = cv::imread(argv[1]);

    //内部行列、歪みベクトル読み出し
    cv::FileStorage inputfs(argv[2], cv::FileStorage::READ);
    if (!inputfs.isOpened()){
        std::cout << "File can not be opened." << std::endl;
        return -1;
    }

	cv::Mat intrinsic, distortion;
	inputfs["intrinsic"] >> intrinsic;
	inputfs["distortion"] >> distortion;

    inputfs.release();

    // 歪みを取り除いた画像の生成
    cv::Mat undistorted = undistort(intrinsic, distortion, image);

    // 歪み補正前の画像に対してチェッカーボード検出
    auto corners_distorted = find_chessboard_corners(image, chess_rows, chess_cols);

    // 歪み補正後の画像に対してチェッカーボード検出
    auto corners_undistorted = find_chessboard_corners(undistorted, chess_rows, chess_cols);

    // 検出された格子点に緑色の円を描画
    for(int i=0; i<corners_distorted.size();i++){
        cv::circle(image, corners_distorted[i], 5, cv::Scalar(0,255,0), 3, 4);
    }
    for(int i=0; i<corners_undistorted.size();i++){
        cv::circle(undistorted, corners_undistorted[i], 5, cv::Scalar(0,255,0), 3, 4);
    }

    // 三次元座標
    std::vector<cv::Point3f> points;
    for(int i=0; i<chess_rows; i++) {
        for(int j=0; j<chess_cols; j++) {
            points.push_back(cv::Point3f(i*chess_size, j*chess_size, 0.0f));
        }
    }

    cv::Mat rvec_distorted, tvec_distorted, rvec_undistorted, tvec_undistorted;
    // 歪み補正前の外部行列を求める
    solvePnP(points, corners_distorted, intrinsic, distortion, rvec_distorted, tvec_distorted, false);
    solvePnP(points, corners_undistorted, intrinsic, distortion, rvec_undistorted, tvec_undistorted, false);

    std::cout << "--- rvec_distorted ---\n" << rvec_distorted << std::endl;
    std::cout << "--- tvec_distorted ---\n" << tvec_distorted << std::endl;
    std::cout << "--- rvec_undistorted ---\n" << rvec_undistorted << std::endl;
    std::cout << "--- tvec_undistorted ---\n" << tvec_undistorted << std::endl;

    //cv::FileStorage outputfs("external_camera_matrix.xml", cv::FileStorage::WRITE);
    //内部行列、歪みベクトル読み出し
    cv::FileStorage outputfs(argv[2], cv::FileStorage::APPEND);
    if (!outputfs.isOpened()){
        std::cout << "File can not be opened." << std::endl;
        return -1;
    }

    outputfs << "rvec_distorted" << rvec_distorted;
    outputfs << "tvec_distorted" << tvec_distorted;
    outputfs << "rvec_undistorted" << rvec_undistorted;
    outputfs << "tvec_undistorted" << tvec_undistorted;
    outputfs.release();

    std::vector<cv::Point3f> center_points3D;
    for(int i=0; i<chess_rows; i++) {
        for(int j=0; j<chess_cols; j++) {
            center_points3D.push_back(cv::Point3f(i*chess_size, j*chess_size, 0.0f));
        }
    }

    std::vector<cv::Point2f> center_points2D_distorted, center_points2D_undistorted;
    cv::Mat new_intrinsic;
    //cv::fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic, distortion, image.size(), cv::Matx33d::eye(), new_intrinsic, 1);
    cv::projectPoints(center_points3D, rvec_distorted, tvec_distorted, intrinsic, {}, center_points2D_distorted);
    cv::projectPoints(center_points3D, rvec_undistorted, tvec_undistorted, intrinsic, {}, center_points2D_undistorted);
    for(int i=0; i<center_points2D_distorted.size(); i++){
        cv::circle(image, center_points2D_distorted[i], 5, cv::Scalar(255,0,0), 3, 4);
    }
    for(int i=0; i<center_points2D_undistorted.size(); i++){
        cv::circle(undistorted, center_points2D_undistorted[i], 5, cv::Scalar(255,0,0), 3, 4);
    }
    cv::imshow("image",image);
    cv::imshow("undistorted",undistorted);
    int key = cv::waitKey(0);

    /*
	for(const cv::Mat& image : images) {
		// 歪みを取り除いた画像の生成
        cv::Mat undistorted = undistort(intrinsic, distortion, image);
        cv::imshow("undistorted",undistorted);
        // 歪み補正前の画像に対してチェッカーボード検出
        //auto corners_distorted = find_chessboard_corners(image, chess_rows, chess_cols);

        // 歪み補正後の画像に対してチェッカーボード検出
        auto corners_undistorted = find_chessboard_corners(undistorted, chess_rows, chess_cols);

        // 三次元座標
        std::vector<cv::Point3f> points;
        for(int i=0; i<chess_rows; i++) {
            for(int j=0; j<chess_cols; j++) {
                points.push_back(cv::Point3f(i*chess_size, j*chess_size, 0.0f));
            }
        }

        cv::Mat rvec, tvec;
        // 歪み補正前の外部行列を求める
        if(corners_undistorted.size()==0) continue;
        solvePnP(points, corners_undistorted, intrinsic, distortion, rvec, tvec, false);

        std::cout << "--- rvec ---\n" << rvec << std::endl;
        std::cout << "--- tvec ---\n" << tvec << std::endl;
        int key = cv::waitKey(0);
        if(key == 'n'){
            continue;
        }
	}*/
}