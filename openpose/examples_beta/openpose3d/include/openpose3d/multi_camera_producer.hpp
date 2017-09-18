#ifndef OPENPOSE3D_POINT_GREY_HPP
#define OPENPOSE3D_POINT_GREY_HPP

#include <openpose/headers.hpp>
#include <openpose3d/datum3D.hpp>
#include <opencv2/opencv.hpp>
//#include <Spinnaker.h>

// Following OpenPose `tutorial_wrapper/` examples, we create our own class inherited from WorkerProducer.
// This worker:
// 1. Set hardware trigger and the buffer to get the latest obtained frame.
// 2. Read images from FLIR cameras.
// 3. Turn them into std::vector<cv::Mat>.
// 4. Return the resulting images wrapped into a std::shared_ptr<std::vector<Datum3D>>.
// The HW trigger + reading FLIR camera code is highly based on the Spinnaker SDK examples `AcquisitionMultipleCamera` and specially `Trigger`
// (located in `src/`). See them for more details about the cameras.
// See `examples/tutorial_wrapper/` for more details about inhering the WorkerProducer class.
class WMultiCamera : public op::WorkerProducer<std::shared_ptr<std::vector<Datum3D>>>
{
public:
	
    //WMultiCamera(std::vector<cv::VideoCapture> &_cameras, std::vector<cv::FileStorage> &_camerafs);
    WMultiCamera(std::vector<std::string> &_camera_path, std::vector<cv::FileStorage> &_camerafs);
    
    ~WMultiCamera();
    
    void initializationOnThread();
    
    std::shared_ptr<std::vector<Datum3D>> workProducer();
    
private:
    bool initialized;
    std::vector<cv::VideoCapture> cameras;
    std::vector<std::string> camera_path;
    std::vector<cv::FileStorage> camerafs;
    std::vector<cv::Mat> intrinsics;
    std::vector<cv::Mat> distortions;
    std::vector<cv::Mat> camera_matrixs;
};

#endif // OPENPOSE3D_POINT_GREY_HPP