// 3rdparty dependencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <usr/estimate_2d_pose_from_image.hpp>

using namespace std;
using namespace cv;

/*
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path,               "examples/media/COCO_val2014_000000000192.jpg",     "Process the desired image.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased,"
                                                        " the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect"
                                                        " ratio possible to the images or videos to be processed. E.g. the default `656x368` is"
                                                        " optimal for 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) videos.");
DEFINE_string(resolution,               "1280x720",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " default images resolution.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                                        " will only display the results on a black background.");
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
*/

//openposeを実行して骨格を描画したMatを返す
cv::Mat estimate_2d::execOp(cv::Mat inputImage,
                op::CvMatToOpInput *cvMatToOpInput,
                op::CvMatToOpOutput *cvMatToOpOutput,
                op::PoseExtractorCaffe *poseExtractorCaffe,
                op::PoseRenderer *poseRenderer,
                op::OpOutputToCvMat *opOutputToCvMat)
{
    // ------------------------- POSE ESTIMATION AND RENDERING -------------------------
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    //cv::Mat inputImage = op::loadImage(FLAGS_image_path, CV_LOAD_IMAGE_COLOR); // Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    //cv::Mat inputImage = cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    if(inputImage.empty())
        //op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
        std::cout<<"openpose error!"<<std::endl;
    // Step 2 - Format input image to OpenPose input and output formats
    op::Array<float> netInputArray;
    std::vector<float> scaleRatios;
    std::tie(netInputArray, scaleRatios) = cvMatToOpInput->format(inputImage);
    double scaleInputToOutput;
    op::Array<float> outputArray;
    std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput->format(inputImage);
    // Step 3 - Estimate poseKeypoints
    poseExtractorCaffe->forwardPass(netInputArray, {inputImage.cols, inputImage.rows}, scaleRatios);
    const auto poseKeypoints = poseExtractorCaffe->getPoseKeypoints();
    // Step 4 - Render poseKeypoints
    poseRenderer->renderPose(outputArray, poseKeypoints);
    // Step 5 - OpenPose output format to cv::Mat
    cv::Mat outputImage = opOutputToCvMat->formatToCvMat(outputArray);
    
    std::vector<cv::Point2d> bodyPoints2D;
    for(int i = 0; i<18 ;i++){
        cv::Point2d _bodyPoint(poseKeypoints[3*i], poseKeypoints[3*i+1]);
        bodyPoints2D.push_back(_bodyPoint);
    }
    for(int i = 0; i<18 ;i++){
        cv::circle(outputImage, bodyPoints2D[i], 3, cv::Scalar(0,0,200), -1);
        //std::cout << "bodyPoints2DVec[" <<  i << "] : " << bodyPoints2DVec[i] << std::endl;
    }
    // ------------------------- SHOWING RESULT AND CLOSING -------------------------
    // Step 1 - Show results
    //frameDisplayer->displayFrameoutputImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
    cv::imshow("outputImage",outputImage);
    return outputImage;
    //cv::waitKey(0);
    // Step 2 - Logging information message
    //op::log("Example 1 successfully finished.", op::Priority::High);
}

// openposeを実行して関節座標をベクトルで返す
std::vector<cv::Point2d> estimate_2d::getEstimated2DPoseVec(cv::Mat inputImage,
                                               op::CvMatToOpInput *cvMatToOpInput,
                                               op::CvMatToOpOutput *cvMatToOpOutput,
                                               op::PoseExtractorCaffe *poseExtractorCaffe)
{
  // ------------------------- POSE ESTIMATION AND RENDERING -------------------------
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    //cv::Mat inputImage = op::loadImage(FLAGS_image_path, CV_LOAD_IMAGE_COLOR); // Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    //cv::Mat inputImage = cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    if(inputImage.empty())
        //op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
        std::cout<<"openpose error!"<<std::endl;
    // Step 2 - Format input image to OpenPose input and output formats
    op::Array<float> netInputArray;
    std::vector<float> scaleRatios;
    std::tie(netInputArray, scaleRatios) = cvMatToOpInput->format(inputImage);
    double scaleInputToOutput;
    op::Array<float> outputArray;
    std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput->format(inputImage);
    // Step 3 - Estimate poseKeypoints
    poseExtractorCaffe->forwardPass(netInputArray, {inputImage.cols, inputImage.rows}, scaleRatios);
    const auto poseKeypoints = poseExtractorCaffe->getPoseKeypoints();

    std::vector<cv::Point2d> bodyPoints2D;
    if(poseKeypoints.empty()!=1){
        for(int i = 0; i<18 ;i++){
            cv::Point2d _bodyPoint(poseKeypoints[3*i], poseKeypoints[3*i+1]);
            bodyPoints2D.push_back(_bodyPoint);
            //cout<<bodyPoints2D[i]<<endl;
        }
    }
    return bodyPoints2D;
}

// openposeを実行して関節座標をベクトルで返す
std::vector<cv::Point2d> estimate_2d::_getEstimated2DPoseVec(cv::Mat inputImage,
                                               op::CvMatToOpInput& cvMatToOpInput,
                                               op::CvMatToOpOutput& cvMatToOpOutput,
                                               op::PoseExtractorCaffe& poseExtractorCaffe)
{
  // ------------------------- POSE ESTIMATION AND RENDERING -------------------------
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    //cv::Mat inputImage = op::loadImage(FLAGS_image_path, CV_LOAD_IMAGE_COLOR); // Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    //cv::Mat inputImage = cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    if(inputImage.empty())
        //op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
        std::cout<<"openpose error!"<<std::endl;
    // Step 2 - Format input image to OpenPose input and output formats
    op::Array<float> netInputArray;
    std::vector<float> scaleRatios;
    std::tie(netInputArray, scaleRatios) = cvMatToOpInput.format(inputImage);
    double scaleInputToOutput;
    op::Array<float> outputArray;
    std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(inputImage);
    // Step 3 - Estimate poseKeypoints
    poseExtractorCaffe.forwardPass(netInputArray, {inputImage.cols, inputImage.rows}, scaleRatios);
    const auto poseKeypoints = poseExtractorCaffe.getPoseKeypoints();

    std::vector<cv::Point2d> bodyPoints2D;
    if(poseKeypoints.empty()!=1){
        for(int i = 0; i<18 ;i++){
            cv::Point2d _bodyPoint(poseKeypoints[3*i], poseKeypoints[3*i+1]);
            bodyPoints2D.push_back(_bodyPoint);
            //cout<<bodyPoints2D[i]<<endl;
        }
    }
    return bodyPoints2D;
}

void estimate_2d::get2DPose(cv::VideoCapture& camera,
                            int frameNum,
                            cv::Mat& mapx,
                            cv::Mat& mapy,
                            op::CvMatToOpInput& cvMatToOpInput,
                            op::CvMatToOpOutput& cvMatToOpOutput,
                            op::PoseExtractorCaffe& poseExtractorCaffe,
                            std::vector<std::vector<cv::Point2d>>& bodyPoints){
    for(int frame_ptr=0; frame_ptr<frameNum; frame_ptr++){
        cv::Mat frame, undistorted_image;
        camera >> frame;
        cv::remap(frame, undistorted_image, mapx, mapy, INTER_LINEAR);
        std::vector<cv::Point2d> joint_vec = _getEstimated2DPoseVec(undistorted_image,
                                                                    cvMatToOpInput,
                                                                    cvMatToOpOutput,
                                                                    poseExtractorCaffe);
        bodyPoints.push_back(joint_vec);
    }
}

// openposeを実行して関節座標をMatで返す
cv::Mat estimate_2d::getEstimated2DPoseMat(cv::Mat inputImage,
                               op::CvMatToOpInput *cvMatToOpInput,
                               op::CvMatToOpOutput *cvMatToOpOutput,
                               op::PoseExtractorCaffe *poseExtractorCaffe)
{
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    //cv::Mat inputImage = op::loadImage(FLAGS_image_path, CV_LOAD_IMAGE_COLOR); // Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    //cv::Mat inputImage = cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    if(inputImage.empty())
        //op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
        std::cout<<"openpose error!"<<std::endl;
    // Step 2 - Format input image to OpenPose input and output formats
    op::Array<float> netInputArray;
    std::vector<float> scaleRatios;
    std::tie(netInputArray, scaleRatios) = cvMatToOpInput->format(inputImage);
    double scaleInputToOutput;
    op::Array<float> outputArray;
    std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput->format(inputImage);
    // Step 3 - Estimate poseKeypoints
    poseExtractorCaffe->forwardPass(netInputArray, {inputImage.cols, inputImage.rows}, scaleRatios);
    const auto poseKeypoints = poseExtractorCaffe->getPoseKeypoints();
    cv::Mat bodyPoints2D = cv::Mat::zeros(2,18,CV_32FC1);
    for (int i = 0; i<18; i++){
        bodyPoints2D.at<float>(0,i) = poseKeypoints[3*i];
        bodyPoints2D.at<float>(1,i) = poseKeypoints[3*i+1];
        //cout << "poseKeyPoints[" << i << "] : " << poseKeyPoints[3*i] << "," << poseKeyPoints[3*i+1] << std::endl;
    }
    return bodyPoints2D;
}