// 3rdparty dependencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

#include <usr/estimate_2d_pose_from_image.hpp>
#include <usr/prediction.hpp>
#include <usr/3d_reconstruction.hpp>
// Eigen
#include <Eigen/Core>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/viz/viz3d.hpp>

#include <time.h>

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

using namespace std;
using namespace cv;

int undist_on = 1;

int main(int argc, char *argv[])
{
    int camera_value = std::stoi(argv[1]);
    std::vector<cv::VideoCapture> cameras(camera_value);
    std::vector<cv::FileStorage> camerafs(camera_value);
    for(int i=0; i<camera_value; i++){
        cameras[i] = cv::VideoCapture(argv[2+(i*2)]);
        camerafs[i] = cv::FileStorage(argv[3+(i*2)], cv::FileStorage::READ);
    }
    cv::FileStorage output_3d_fs(argv[2*(camera_value+1)], cv::FileStorage::WRITE);
    if(!output_3d_fs.isOpened()){
        std::cout<<"File can not be opened." << std::endl;
        return -1;
    }

    //内部行列、歪みベクトル、回転ベクトル、並進ベクトル読み出し
    std::vector<cv::Mat> intrinsics(camera_value);
    std::vector<cv::Mat> distortions(camera_value);
    std::vector<cv::Mat> rotations(camera_value);
    std::vector<cv::Mat> translations(camera_value);
    std::vector<cv::Mat> externals(camera_value);
    std::vector<cv::Mat> camera_matrixs(camera_value);
    std::vector<cv::Mat> Pp(camera_value);
    std::vector<cv::Size> video_sizes(camera_value);
    std::vector<cv::Mat> new_intrinsics(camera_value);
    std::vector<cv::Mat> mapx(camera_value);
    std::vector<cv::Mat> mapy(camera_value);
    for(int i=0; i<camera_value; i++){
        camerafs[i]["intrinsic"] >> intrinsics[i];
        camerafs[i]["distortion"] >> distortions[i];
        cv::Mat rvec;
        camerafs[i]["rvec_undistorted"] >> rvec;
        cv::Rodrigues(rvec, rotations[i]);    
        camerafs[i]["tvec_undistorted"] >> translations[i];
        cv::hconcat(rotations[i], translations[i], externals[i]);
        camera_matrixs[i] = intrinsics[i] * externals[i];
        Pp[i] = camera_matrixs[i];
        video_sizes[i] = cv::Size(cameras[i].get(CV_CAP_PROP_FRAME_WIDTH),cameras[i].get(CV_CAP_PROP_FRAME_HEIGHT));
        cv::initUndistortRectifyMap(intrinsics[i], distortions[i], cv::Mat(), new_intrinsics[i], video_sizes[i], CV_32FC1, mapx[i], mapy[i]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Initializing google logging (Caffe uses it for logging)
    google::InitGoogleLogging("openPoseTutorialPose1");

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Initializing openpose
    op::log("OpenPose Library Tutorial - Example 1.", op::Priority::High);
    // ------------------------- INITIALIZATION -------------------------
    // Step 1 - Set logging level
        // - 0 will output all the logging messages
        // - 255 will output nothing
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    
    // Step 2 - Read Google flags (user defined configuration)
    //規定値から変更。入力動画のサイズを読み取り、その画像における関節座標を取得する（全ての画像が同じ解像度だと明らかに楽！）
    stringstream ss;
    ss<<(int)cameras[0].get(CV_CAP_PROP_FRAME_WIDTH)<<"x"<<(int)cameras[0].get(CV_CAP_PROP_FRAME_HEIGHT)<<endl;
    const auto outputSize = op::flagsToPoint(ss.str());
    // netInputSize
    const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "656x368");
    // netOutputSize
    const auto netOutputSize = netInputSize;
    // poseModel
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    // Check no contradictory flags enabled
    if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
        op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
    if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
        op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.", __LINE__, __FUNCTION__, __FILE__);
    // Logging
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Step 3 - Initialize all required classes
    op::CvMatToOpInput cvMatToOpInput{netInputSize, FLAGS_scale_number, (float)FLAGS_scale_gap};
    op::CvMatToOpOutput cvMatToOpOutput{outputSize};
    op::PoseExtractorCaffe poseExtractorCaffe{netInputSize, netOutputSize, outputSize, FLAGS_scale_number, poseModel,
                                              FLAGS_model_folder, FLAGS_num_gpu_start};
    op::PoseRenderer poseRenderer{netOutputSize, outputSize, poseModel, nullptr, (float)FLAGS_render_threshold,
                                  !FLAGS_disable_blending, (float)FLAGS_alpha_pose};
    op::OpOutputToCvMat opOutputToCvMat{outputSize};
    const op::Point<int> windowedSize = outputSize;
    op::FrameDisplayer frameDisplayer{windowedSize, "OpenPose Tutorial - Example 1"};
    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseExtractorCaffe.initializationOnThread();
    poseRenderer.initializationOnThread();
    ////////////////////////////////////////////////////////////////////////////////////////////////

    int frameNum = cameras[0].get(CV_CAP_PROP_FRAME_COUNT);
    for(int i=1; i<cameras.size(); i++){
        if(frameNum>cameras[i].get(CV_CAP_PROP_FRAME_COUNT)){
            frameNum = cameras[i].get(CV_CAP_PROP_FRAME_COUNT);
        }
    }
    std::cout<<"frameNum : "<<frameNum<<std::endl;
    
    std::vector<std::vector<std::vector<cv::Point2d>>> bodyPoints2D; //bodyPoints2D[frameNum][cameraNum][bodyPartsNum]

    clock_t start = clock();    // スタート時間
    
    //二次元関節座標の推定
    for(int video_framePtr=0; video_framePtr<frameNum; video_framePtr++){
        // 動画から画像の読み出し
        std::vector<std::vector<cv::Point2d>> bodyPoints2D_frame;
        for(int i=0; i<camera_value; i++){
            cv::Mat frame, undistorted_image;
            cameras[i] >> frame;
            //歪み補正
            cv::remap(frame, undistorted_image, mapx[i], mapy[i], INTER_LINEAR);
            std::vector<cv::Point2d> joint_vec = estimate_2d::_getEstimated2DPoseVec(undistorted_image,
                                                                                    std::ref(cvMatToOpInput),
                                                                                    std::ref(cvMatToOpOutput),
                                                                                    std::ref(poseExtractorCaffe));
            bodyPoints2D_frame.push_back(joint_vec);
        }
        bodyPoints2D.push_back(bodyPoints2D_frame);
    }
    /*
    std::vector<std::vector<std::vector<cv::Point2d>>> pose_2d_vec_each_camera(camera_value); //2d_pose_vec_each_camera[camera_num][frame_num][joint_num]
    std::vector<std::thread> estimate_2d_pose_threads(camera_value);
    for(int i=0; i<camera_value; i++){
        
        estimate_2d_pose_threads[i] = std::thread(estimate_2d::get2DPose,
                                                std::ref(cameras[i]),
                                                frameNum,
                                                std::ref(mapx[i]),
                                                std::ref(mapy[i]),
                                                std::ref(cvMatToOpInput),
                                                std::ref(cvMatToOpOutput),
                                                std::ref(poseExtractorCaffe),
                                                std::ref(pose_2d_vec_each_camera[i]));
        estimate_2d_pose_threads[i].join();
    }
    */
    clock_t end = clock();     // 終了時間
    std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
    cout<<"2d pose estimated!"<<endl;

    //フレーム補間
    std::vector<std::vector<std::vector<cv::Point2d>>> bodyPoints2D_bilateral_interpolated = prediction::bilateral_prediction(bodyPoints2D); //bodyPoints2D_bilateral_interpolated[frameNum][cameraNum][bodyPartsNum]

    cout<<"interpolated!"<<endl;

    //三次元再構成
    for(int i=0; i<frameNum;i++){
        cv::Mat points3d = (cv::Mat_<double>(3,1) << 1,1,1);
        for(int joint_ptr=0; joint_ptr<bodyPoints2D_bilateral_interpolated[i][0].size(); joint_ptr++){
            vector<cv::Point2d> same_joint_vec = {bodyPoints2D_bilateral_interpolated[i][0][joint_ptr], bodyPoints2D_bilateral_interpolated[i][1][joint_ptr]};
            cv::Mat same_joint_3d;
            triangulation_and_3dreconstruction::triangulationWithOptimization(same_joint_3d, Pp, same_joint_vec);
            //std::cout<<"same_joint_3d"<<same_joint_3d(cv::Rect(0,0,1,3))<<std::endl;
            cv::hconcat(points3d, same_joint_3d(cv::Rect(0,0,1,3)), points3d);
        }
        cv::Mat points3d_reshaped = points3d(cv::Rect(1,0,bodyPoints2D_bilateral_interpolated[i][0].size(),3));

        //フレームに紐づいた名前
        std::string frameCount = "frame" + std::to_string(i);
        output_3d_fs << frameCount << points3d_reshaped;
    }
    cout<<"3d pose estimated!"<<endl;
    output_3d_fs << "frameNum" << frameNum;
    output_3d_fs.release();
    return 0;
}
