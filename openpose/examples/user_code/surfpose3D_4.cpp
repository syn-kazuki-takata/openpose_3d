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
    //ビデオ入力獲得
    cv::VideoCapture camera1(argv[1]);
    cv::VideoCapture camera2(argv[2]);

    cv::FileStorage inputfs1(argv[3], cv::FileStorage::READ);
    if (!inputfs1.isOpened()){
        std::cout << "File1 can not be opened." << std::endl;
        return -1;
    }
    cv::FileStorage inputfs2(argv[4], cv::FileStorage::READ);
    if (!inputfs2.isOpened()){
        std::cout << "File2 can not be opened." << std::endl;
        return -1;
    }

    //内部行列、歪みベクトル、回転ベクトル、並進ベクトル読み出し
    cv::Mat intrinsic1, distortion1,  rvec1, translation1, intrinsic2, distortion2, rvec2, translation2;
    inputfs1["intrinsic"] >> intrinsic1;
    inputfs1["distortion"] >> distortion1;
    inputfs1["rvec_undistorted"] >> rvec1;
    inputfs1["tvec_undistorted"] >> translation1;
    inputfs2["intrinsic"] >> intrinsic2;
    inputfs2["distortion"] >> distortion2;
    inputfs2["rvec_undistorted"] >> rvec2;
    inputfs2["tvec_undistorted"] >> translation2;

    // 回転ベクトルを回転行列に変換
    cv::Mat rotation1, rotation2;
    cv::Rodrigues(rvec1, rotation1);
    cv::Rodrigues(rvec2, rotation2);
    
    //水平方向連結
    cv::Mat externalMat1, externalMat2;
    cv::hconcat(rotation1, translation1, externalMat1);
    cv::hconcat(rotation2, translation2, externalMat2);

    //透視投影行列計算
    cv::Mat camera_matrix1 = intrinsic1 * externalMat1;
    cv::Mat camera_matrix2 = intrinsic2 * externalMat2;

	//カメラ行列のベクトル生成
    std::vector<cv::Mat> Pp = {camera_matrix1,camera_matrix2};

    // 歪みマップ行列を求める
    cv::Mat new_camera_matrix1, mapx1, mapy1, new_camera_matrix2, mapx2, mapy2;
    cv::Size video_size1(camera1.get(CV_CAP_PROP_FRAME_WIDTH), camera1.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::Size video_size2(camera2.get(CV_CAP_PROP_FRAME_WIDTH), camera2.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::initUndistortRectifyMap(intrinsic1, distortion1, cv::Mat(), new_camera_matrix1, video_size1, CV_32FC1, mapx1, mapy1);
    cv::initUndistortRectifyMap(intrinsic2, distortion2, cv::Mat(), new_camera_matrix2, video_size2, CV_32FC1, mapx2, mapy2);    

	//書き込み用のXMLファイルを開く
    cv::FileStorage output_2d_fs1(argv[5], cv::FileStorage::WRITE);
    if(!output_2d_fs1.isOpened()){
        std::cout<<"File can not be opened." << std::endl;
        return -1;
    }
    cv::FileStorage output_2d_fs2(argv[6], cv::FileStorage::WRITE);
    if(!output_2d_fs2.isOpened()){
        std::cout<<"File can not be opened." << std::endl;
        return -1;
    }
	cv::FileStorage output_3d_fs(argv[7], cv::FileStorage::WRITE);
	if(!output_3d_fs.isOpened()){
		std::cout<<"File can not be opened." << std::endl;
        return -1;
    }
    cv::FileStorage _output_3d_fs(argv[8], cv::FileStorage::WRITE);
    if(!_output_3d_fs.isOpened()){
        std::cout<<"File can not be opened." << std::endl;
        return -1;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Initializing google logging (Caffe uses it for logging)
    google::InitGoogleLogging("openPoseTutorialPose1");

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseTutorialPose1
    //initializeOp();

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
    // outputSize
    //規定値から変更。入力動画のサイズを読み取り、その画像における関節座標を取得する
    stringstream ss;
    ss<<(int)camera1.get(CV_CAP_PROP_FRAME_WIDTH)<<"x"<<(int)camera1.get(CV_CAP_PROP_FRAME_HEIGHT)<<endl;
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

    int frameNum = std::min(camera1.get(CV_CAP_PROP_FRAME_COUNT), camera2.get(CV_CAP_PROP_FRAME_COUNT));
    std::cout<<"frameNum : "<<frameNum<<std::endl;
    Mat camera1Img, camera2Img;
    std::vector<std::vector<std::vector<cv::Point2d>>> bodyPoints2D; //bodyPoints2D[frameNum][cameraNum][bodyPartsNum]
    
    //二次元関節座標の推定
    for(int video_framePtr=0 ; video_framePtr<frameNum;video_framePtr++){
        // 動画から画像の読み出し
        camera1 >> camera1Img;
        camera2 >> camera2Img;

        //歪み補正を行う
        cv::Mat undistorted_image1, undistorted_image2;
        cv::remap(camera1Img, undistorted_image1, mapx1, mapy1, INTER_LINEAR);
        cv::remap(camera2Img, undistorted_image2, mapx2, mapy2, INTER_LINEAR);

        std::vector<cv::Point2d> camera1JointVec = estimate_2d::getEstimated2DPoseVec(undistorted_image1,
                                                                          &cvMatToOpInput,
                                                                          &cvMatToOpOutput,
                                                                          &poseExtractorCaffe);

        std::vector<cv::Point2d> camera2JointVec = estimate_2d::getEstimated2DPoseVec(undistorted_image2,
                                                                          &cvMatToOpInput,
                                                                          &cvMatToOpOutput,
                                                                          &poseExtractorCaffe);

        std::vector<std::vector<cv::Point2d>> bodyPoints2D_frame;
        bodyPoints2D_frame.push_back(camera1JointVec);
        bodyPoints2D_frame.push_back(camera2JointVec);

        bodyPoints2D.push_back(bodyPoints2D_frame);
    }

    cout<<"2d pose estimated!"<<endl;

    //フレーム補間
    std::vector<std::vector<std::vector<cv::Point2d>>> bodyPoints2D_bilateral_interpolated = prediction::bilateral_prediction(bodyPoints2D);　//bodyPoints2D_bilateral_interpolated[frameNum][cameraNum][bodyPartsNum]

    cout<<"interpolated!"<<endl;

    for(int i=0; i<frameNum;i++){
        cv::Mat points4D, points3D;
        std::vector<cv::Mat> _bodyPoints3D;

        //フレームに紐づいた名前
        std::string frameCount = "frame" + std::to_string(i);

        cv::Mat points1Mat = (cv::Mat_<double>(2,1) << 1, 1);
        cv::Mat points2Mat = (cv::Mat_<double>(2,1) << 1, 1);
        for (int joint_num_1=0; joint_num_1 < bodyPoints2D_bilateral_interpolated[i][0].size(); joint_num_1++) {
            cv::Point2d myPoint1 = bodyPoints2D_bilateral_interpolated[i][0].at(joint_num_1);
            cv::Mat matPoint1 = (cv::Mat_<double>(2,1) << myPoint1.x, myPoint1.y);
            cv::hconcat(points1Mat, matPoint1, points1Mat);
        }
        for (int joint_num_2=0; joint_num_2 < bodyPoints2D_bilateral_interpolated[i][1].size(); joint_num_2++) {
            cv::Point2d myPoint2 = bodyPoints2D_bilateral_interpolated[i][1].at(joint_num_2);
            cv::Mat matPoint2 = (cv::Mat_<double>(2,1) << myPoint2.x, myPoint2.y);
            cv::hconcat(points2Mat, matPoint2, points2Mat);
        }
        cv::Mat points1Mat_reshaped = points1Mat(cv::Rect(1,0,18,2));
        cv::Mat points2Mat_reshaped = points2Mat(cv::Rect(1,0,18,2));
        output_2d_fs1 << frameCount << points1Mat_reshaped;
        output_2d_fs2 << frameCount << points2Mat_reshaped;
        
        vector<cv::Mat> sfmPoints2d;

        //二次元関節座標の行列をベクトル化
        sfmPoints2d.push_back(points1Mat_reshaped);
        sfmPoints2d.push_back(points2Mat_reshaped);

        cv::Mat points3d;
        // ステレオ視による三次元骨格再構成
        cv::sfm::triangulatePoints(sfmPoints2d,Pp,points3d);

        //三次元関節座標を書き込み
    	output_3d_fs << frameCount << points3d;
        cv::Mat _points3d = (cv::Mat_<double>(3,1) << 1,1,1);
        for(int joint_ptr=0; joint_ptr<bodyPoints2D_bilateral_interpolated[i][0].size(); joint_ptr++){
            vector<cv::Point2d> same_joint_vec = {bodyPoints2D_bilateral_interpolated[i][0][joint_ptr], bodyPoints2D_bilateral_interpolated[i][1][joint_ptr]};
            cv::Mat same_joint_3d;
            triangulation_and_3dreconstruction::triangulateWithOptimization(same_joint_3d, Pp, same_joint_vec);
            std::cout<<"same_joint_3d"<<same_joint_3d(cv::Rect(0,0,1,3))<<std::endl;
            cv::hconcat(_points3d, same_joint_3d(cv::Rect(0,0,1,3)), _points3d);
        }
        cv::Mat _points3d_reshaped = _points3d(cv::Rect(1,0,18,3));
        _output_3d_fs << frameCount << _points3d_reshaped;
    	//std::cout<<"i"<<i<<std::endl;
        //「Mat形式の関節位置のベクトル」のベクトルを取得
        //std::cout<<"bodyPoints3D"<<_bodyPoints3D<<std::endl;
        //bodyPoints3D.push_back(_bodyPoints3D);
        //std::cout<<"bodyPoints3D"<<_bodyPoints3D<<std::endl;

        /*
        cv::imshow("camera1Image",camera1Img);
        cv::imshow("camera2Image",camera2Img);
        for(int i = 0;i<18;i++){
            for(int j=0;j<2;j++){
                points3d.at<float>(i,j) = points3d.at<float>(i,j)/points3d.at<float>(18,j);
            }
        }

        std::vector<cv::viz::WLine> lines;
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(0,0),points3d.at<float>(0,1),points3d.at<float>(0,2)),Point3f(points3d.at<float>(1,0),points3d.at<float>(1,1),points3d.at<float>(1,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(0,0),points3d.at<float>(0,1),points3d.at<float>(0,2)),Point3f(points3d.at<float>(14,0),points3d.at<float>(14,1),points3d.at<float>(14,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(0,0),points3d.at<float>(0,1),points3d.at<float>(0,2)),Point3f(points3d.at<float>(15,0),points3d.at<float>(15,1),points3d.at<float>(15,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(1,0),points3d.at<float>(1,1),points3d.at<float>(1,2)),Point3f(points3d.at<float>(2,0),points3d.at<float>(2,1),points3d.at<float>(2,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(1,0),points3d.at<float>(1,1),points3d.at<float>(1,2)),Point3f(points3d.at<float>(5,0),points3d.at<float>(5,1),points3d.at<float>(5,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(1,0),points3d.at<float>(1,1),points3d.at<float>(1,2)),Point3f(points3d.at<float>(8,0),points3d.at<float>(8,1),points3d.at<float>(8,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(1,0),points3d.at<float>(1,1),points3d.at<float>(1,2)),Point3f(points3d.at<float>(11,0),points3d.at<float>(11,1),points3d.at<float>(11,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(2,0),points3d.at<float>(2,1),points3d.at<float>(2,2)),Point3f(points3d.at<float>(3,0),points3d.at<float>(3,1),points3d.at<float>(3,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(3,0),points3d.at<float>(3,1),points3d.at<float>(3,2)),Point3f(points3d.at<float>(4,0),points3d.at<float>(4,1),points3d.at<float>(4,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(5,0),points3d.at<float>(5,1),points3d.at<float>(5,2)),Point3f(points3d.at<float>(6,0),points3d.at<float>(6,1),points3d.at<float>(6,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(6,0),points3d.at<float>(6,1),points3d.at<float>(6,2)),Point3f(points3d.at<float>(7,0),points3d.at<float>(7,1),points3d.at<float>(7,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(8,0),points3d.at<float>(8,1),points3d.at<float>(8,2)),Point3f(points3d.at<float>(9,0),points3d.at<float>(9,1),points3d.at<float>(9,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(10,0),points3d.at<float>(10,1),points3d.at<float>(10,2)),Point3f(points3d.at<float>(9,0),points3d.at<float>(9,1),points3d.at<float>(9,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(12,0),points3d.at<float>(12,1),points3d.at<float>(12,2)),Point3f(points3d.at<float>(11,0),points3d.at<float>(11,1),points3d.at<float>(11,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(12,0),points3d.at<float>(12,1),points3d.at<float>(12,2)),Point3f(points3d.at<float>(13,0),points3d.at<float>(13,1),points3d.at<float>(13,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(14,0),points3d.at<float>(14,1),points3d.at<float>(14,2)),Point3f(points3d.at<float>(16,0),points3d.at<float>(16,1),points3d.at<float>(16,2)), cv::viz::Color::red()));
        lines.push_back(cv::viz::WLine(Point3f(points3d.at<float>(15,0),points3d.at<float>(15,1),points3d.at<float>(15,2)),Point3f(points3d.at<float>(17,0),points3d.at<float>(17,1),points3d.at<float>(17,2)), cv::viz::Color::red()));

        for(int i= 0;i<16;i++){
            stringstream ss;
            ss << i;
            std::string istr = ss.str();
            visualizer.showWidget("line"+istr, lines[i]);     
        }
        

        visualizer.spinOnce();
        //cv::waitKey(1);
        */
        
        /*
        if(int key = 'q){
            return 1;
        }
        */
    }
    cout<<"3d pose estimated!"<<endl;
    output_3d_fs << "frameNum" << frameNum;
    _output_3d_fs << "frameNum" << frameNum;
    output_2d_fs1.release();
    output_2d_fs2.release();
    output_3d_fs.release();
    return 0;
}
