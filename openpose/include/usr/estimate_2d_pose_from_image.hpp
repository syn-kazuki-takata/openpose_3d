#ifdef USE_CAFFE
#ifndef ESTIMATE_2D_POSE_FROM_IMAGE_HPP
#define ESTIMATE_2D_POSE_FROM_IMAGE_HPP

#include <caffe/blob.hpp>
#include <openpose/core/common.hpp>
#include <openpose/core/net.hpp>
#include <openpose/core/nmsCaffe.hpp>
#include <openpose/core/resizeAndMergeCaffe.hpp>
#include <openpose/pose/bodyPartConnectorCaffe.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseExtractor.hpp>
#include <openpose/pose/poseExtractorCaffe.hpp>

namespace estimate_2d
{
    cv::Mat execOp(cv::Mat inputImage,
                    op::CvMatToOpInput *cvMatToOpInput,
                    op::CvMatToOpOutput *cvMatToOpOutput,
                    op::PoseExtractorCaffe *poseExtractorCaffe,
                    op::PoseRenderer *poseRenderer,
                    op::OpOutputToCvMat *opOutputToCvMat);
    
    std::vector<cv::Point2d> getEstimated2DPoseVec(cv::Mat inputImage,
                                                   op::CvMatToOpInput *cvMatToOpInput,
                                                   op::CvMatToOpOutput *cvMatToOpOutput,
                                                   op::PoseExtractorCaffe *poseExtractorCaffe);

    cv::Mat getEstimated2DPoseMat(cv::Mat inputImage,
                                   op::CvMatToOpInput *cvMatToOpInput,
                                   op::CvMatToOpOutput *cvMatToOpOutput,
                                   op::PoseExtractorCaffe *poseExtractorCaffe);

    std::vector<cv::Point2d> _getEstimated2DPoseVec(cv::Mat inputImage,
                                                   op::CvMatToOpInput& cvMatToOpInput,
                                                   op::CvMatToOpOutput& cvMatToOpOutput,
                                                   op::PoseExtractorCaffe& poseExtractorCaffe);
    
    void get2DPose(cv::VideoCapture& camera,
                    int frameNum,
                    cv::Mat& mapx,
                    cv::Mat& mapy,
                    op::CvMatToOpInput& cvMatToOpInput,
                    op::CvMatToOpOutput& cvMatToOpOutput,
                    op::PoseExtractorCaffe& poseExtractorCaffe,
                    std::vector<std::vector<cv::Point2d>>& bodyPoints);
}
#endif
#endif