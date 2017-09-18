#ifndef PREDICTION_HPP
#define PREDICTION_HPP

namespace prediction{
	std::vector<std::vector<std::vector<cv::Point2d>>> bilateral_prediction(std::vector<std::vector<std::vector<cv::Point2d>>> bodyPoints2D);	
}
#endif