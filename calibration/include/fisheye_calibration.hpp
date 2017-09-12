#ifndef FISHEYE_CALIBRATION_INTERNAL_H
#define FISHEYE_CALIBRATION_INTERNAL_H

namespace internal{
std::vector<cv::Mat> read_images(const std::string& directory_path);

std::vector<cv::Point2f> find_chessboard_corners(const cv::Mat& image, int rows, int cols);
}

namespace fisheye_calibration{
	cv::Mat undistort(const cv::Mat& K, const cv::Mat& D, const cv::Mat& frame);
	std::vector<cv::Mat> fisheye_calibration_(std::vector<cv::Mat> images);
}

#endif