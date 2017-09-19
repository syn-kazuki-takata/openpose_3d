#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <usr/3d_reconstruction.hpp>

double calclateReprojectionError(const cv::Mat& X, const std::vector<cv::Mat>& M, const std::vector<cv::Point2d>& pt2D)
{
    auto averageError = 0.;
    for(unsigned int i = 0 ; i < M.size() ; i++)
    {
        cv::Mat imageX = M[i] * X;
        imageX /= imageX.at<double>(2,0);
        const auto error = std::sqrt(std::pow(imageX.at<double>(0,0) -  pt2D[i].x,2) + std::pow(imageX.at<double>(1,0) - pt2D[i].y,2));
        //log("Error: " + std::to_string(error));
        averageError += error;
    }
    return averageError / M.size();
}

void triangulation(cv::Mat& X, const std::vector<cv::Mat>& matrixEachCamera, const std::vector<cv::Point2d>& pointOnEachCamera)
{
    // Security checks
    if (matrixEachCamera.empty() || matrixEachCamera.size() != pointOnEachCamera.size())
        std::cout<<"error!"<<std::endl;
    // Create and fill A
    const auto numberCameras = (int)matrixEachCamera.size();
    cv::Mat A = cv::Mat::zeros(numberCameras*2, 4, CV_64F);
    for (auto i = 0 ; i < numberCameras ; i++)
    {
        cv::Mat temp = pointOnEachCamera[i].x*matrixEachCamera[i].rowRange(2,3) - matrixEachCamera[i].rowRange(0,1);
        temp.copyTo(A.rowRange(i*2,i*2+1));
        temp = pointOnEachCamera[i].y*matrixEachCamera[i].rowRange(2,3) - matrixEachCamera[i].rowRange(1,2);
        temp.copyTo(A.rowRange(i*2+1,i*2+2));
    }
    // SVD on A
    cv::SVD svd{A};
    svd.solveZ(A,X);
    X /= X.at<double>(3);
}

// TODO: ask Hanbyul for the missing function: TriangulationOptimization
void triangulation_and_3dreconstruction::triangulationWithOptimization(cv::Mat& X, const std::vector<cv::Mat>& matrixEachCamera, const std::vector<cv::Point2d>& pointOnEachCamera)
{
    triangulation(X, matrixEachCamera, pointOnEachCamera);

    // //if (matrixEachCamera.size() >= 3)
    // //double beforeError = calclateReprojectionError(&matrixEachCamera, pointOnEachCamera, X);
    // double change = TriangulationOptimization(&matrixEachCamera, pointOnEachCamera, X);
    // //double afterError = calclateReprojectionError(&matrixEachCamera,pointOnEachCamera,X);
    // //printfLog("!!Mine %.8f , inFunc %.8f \n",beforeError-afterError,change);
    // return change;
}