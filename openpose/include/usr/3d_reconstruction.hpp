#ifndef TRIANGULATION_AND_3DRECONSTRUCTION_HPP
#define TRIANGULATION_AND_3DRECONSTRUCTION_HPP

namespace triangulation_and_3dreconstruction{
	void triangulationWithOptimization(cv::Mat& X, const std::vector<cv::Mat>& matrixEachCamera, const std::vector<cv::Point2d>& pointOnEachCamera);
}

#endif