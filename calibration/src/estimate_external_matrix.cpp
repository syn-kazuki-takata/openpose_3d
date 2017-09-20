#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "fisheye.hpp"

using namespace std;
using namespace cv;
// reads images in a directory
vector<Mat> read_images(const string& directory_path) {
    vector<Mat> images;

    boost::filesystem::directory_iterator dir(directory_path);
    boost::filesystem::directory_iterator end;

    for(dir; dir != end; dir++) {
        cout << dir->path().string() << endl;
        Mat image = imread(dir->path().string());
        if(image.data) {
            images.push_back(image);
        } else {
            cerr << "couldn't read the image file!!" << dir->path().string() << endl;
        }
    }
    return images;
}

// finds chessboard corners on an image
vector<Point2f> find_chessboard_corners(const Mat& image, int rows, int cols) {
    vector<Point2f> corners;

    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);

    bool found = findChessboardCorners(gray, Size(cols, rows), corners);
    if(!found || corners.size() != rows * cols) {
        cerr << "failed to find the corners!!" << endl;
        return vector<Point2f>();
    }

    find4QuadCornerSubpix(gray, corners, Size(3, 3));
    Mat canvas = image.clone();
    drawChessboardCorners(canvas, Size(cols, rows), corners, found);
    imshow("image", canvas);
    waitKey(1);

    return corners;
}

// undistorts an fisheye image
// @ref http://stackoverflow.com/questions/38983164/opencv-fisheye-undistort-issues
Mat undistort(const Mat& K, const Mat& D, const Mat& frame) {
    Mat newK, map1, map2;
    Mat rview(frame.size(), frame.type());
    fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, frame.size(), Matx33d::eye(), newK, 1);
    fisheye::initUndistortRectifyMap(K, D, Matx33d::eye(), newK, frame.size(), CV_16SC2, map1, map2);
    remap(frame, rview, map1, map2, INTER_LINEAR);
    return rview;
}

int main(int argc, char* argv[]) {
	// chessboard parameters
    const int chess_rows = 3;
    const int chess_cols = 4;
    const double chess_size = 204.0 / 1000.0;

    string videoStr = "video";
    string cameraStr = "camera";
    string inputStr = string(argv[1]);
    //カメラ、内部行列、歪みベクトルの初期化
    int camera_value = stoi(argv[2]);
    vector<VideoCapture> cameras(camera_value);
    vector<string> window_names(camera_value);
    vector<string> undistorted_window_names(camera_value);
    vector<Mat> intrinsic(camera_value);
    vector<Mat> distortion(camera_value);
    for(int i=0; i<camera_value; i++){
        if(inputStr == videoStr){
            cameras[i] = VideoCapture(argv[6+(i*2)]);
            if(!cameras[i].isOpened()){
                //読み込みに失敗したときの処理
                return -1;
            }
        }else{
            //cameras[i] = VideoCapture(stoi(argv[2+i]));
            cameras[i] = VideoCapture(stoi(argv[6+(i*2)]));
            if(!cameras[i].isOpened()){
                //読み込みに失敗したときの処理
                return -1;
            }
        }
        cameras[i].set(CV_CAP_PROP_FPS, stoi(argv[3]));
        cameras[i].set(CV_CAP_PROP_FRAME_WIDTH, stoi(argv[4]));
        cameras[i].set(CV_CAP_PROP_FRAME_HEIGHT, stoi(argv[5]));
        FileStorage inputfs = FileStorage(argv[7+(i*2)], FileStorage::READ);
        if (!inputfs.isOpened()){
            cout << "File can not be opened." << endl;
            return -1;
        }
        inputfs["intrinsic"] >> intrinsic[i];
        inputfs["distortion"] >> distortion[i];
        inputfs.release();
        window_names[i] = "camera_" + to_string(i);
        undistorted_window_names[i] = "undistorted_" + to_string(i);
    }

    vector<Mat> frames(camera_value);
    vector<Mat> estimate_images(camera_value);
    vector<vector<Point2f>> image_points;
    //bool capture_switch = false;
    while(1){
        for(int i=0; i<camera_value; i++){
            cameras[i] >> frames[i];
            if(frames[i].empty()) break;
            imshow(window_names[i], frames[i]);
        }
        int key = waitKey(30);
        if(key == 'c'){
            cout<<"capture!"<<endl;
            vector<vector<Point2f>> image_points_tmp;
            for(int i = 0; i<camera_value; i++){
                // 歪みを取り除いた画像の生成
                Mat undistorted = undistort(intrinsic[i], distortion[i], frames[i]);
                // 歪み補正後の画像に対してチェッカーボード検出
                auto corners_undistorted = find_chessboard_corners(undistorted, chess_rows, chess_cols);
                if(corners_undistorted.empty()){
                    break;
                }else{
                    image_points_tmp.push_back(corners_undistorted);
                }
            }
            if(image_points_tmp.size()==camera_value){
                cout<<"detect complete:)"<<endl;
                cout<<"image_points_size : "<<image_points_tmp.size()<<endl;
                image_points = image_points_tmp;
                break;
            }else{
                cout<<"detect imcomplete:("<<endl;
                cout<<"image_points_size : "<<image_points_tmp.size()<<endl;
            }
        }
        if(key == 'e'){
            cout<<"finish capture!"<<endl;
            break;
        }
    }

    /*
    // 検出された格子点に緑色の円を描画
    for(int i=0; i<corners_distorted.size();i++){
        circle(image, corners_distorted[i], 5, Scalar(0,255,0), 3, 4);
    }
    for(int i=0; i<corners_undistorted.size();i++){
        circle(undistorted, corners_undistorted[i], 5, Scalar(0,255,0), 3, 4);
    }
    */

    // 三次元座標
    vector<Point3f> points;
    for(int i=0; i<chess_rows; i++) {
        for(int j=0; j<chess_cols; j++) {
            points.push_back(Point3f(i*chess_size, j*chess_size, 0.0f));
        }
    }

    vector<Mat>rvec_undistorted(camera_value);
    vector<Mat>tvec_undistorted(camera_value);
    for(int i=0; i<camera_value; i++){
        solvePnP(points, image_points[i], intrinsic[i], distortion[i], rvec_undistorted[i], tvec_undistorted[i], false);
        
        cout << "--- rvec_undistorted ---\n" << rvec_undistorted[i] << endl;
        cout << "--- tvec_undistorted ---\n" << tvec_undistorted[i] << endl;

        FileStorage outputfs(argv[7+(i*2)], FileStorage::APPEND);
        if (!outputfs.isOpened()){
            cout << "File can not be opened." << endl;
            return -1;
        }
        outputfs << "rvec_undistorted" << rvec_undistorted[i];
        outputfs << "tvec_undistorted" << tvec_undistorted[i];
        outputfs.release();
    }

    for(int i=0; i<camera_value; i++){
        vector<Point2f> center_points2D_undistorted;
        Mat undistorted = undistort(intrinsic[i], distortion[i], frames[i]);
        projectPoints(points, rvec_undistorted[i], tvec_undistorted[i], intrinsic[i], {}, center_points2D_undistorted);

        for(int j=0; j<center_points2D_undistorted.size(); j++){
            circle(undistorted, center_points2D_undistorted[j], 5, Scalar(255,0,0), 3, 4);
        }
        imshow(window_names[i],frames[i]);
        imshow(undistorted_window_names[i],undistorted);
    }
    int key = waitKey(0);
    if(key == 'q'){
        return 0;
    }

    /*
	for(const Mat& image : images) {
		// 歪みを取り除いた画像の生成
        Mat undistorted = undistort(intrinsic, distortion, image);
        imshow("undistorted",undistorted);
        // 歪み補正前の画像に対してチェッカーボード検出
        //auto corners_distorted = find_chessboard_corners(image, chess_rows, chess_cols);

        // 歪み補正後の画像に対してチェッカーボード検出
        auto corners_undistorted = find_chessboard_corners(undistorted, chess_rows, chess_cols);

        // 三次元座標
        vector<Point3f> points;
        for(int i=0; i<chess_rows; i++) {
            for(int j=0; j<chess_cols; j++) {
                points.push_back(Point3f(i*chess_size, j*chess_size, 0.0f));
            }
        }

        Mat rvec, tvec;
        // 歪み補正前の外部行列を求める
        if(corners_undistorted.size()==0) continue;
        solvePnP(points, corners_undistorted, intrinsic, distortion, rvec, tvec, false);

        cout << "--- rvec ---\n" << rvec << endl;
        cout << "--- tvec ---\n" << tvec << endl;
        int key = waitKey(0);
        if(key == 'n'){
            continue;
        }
	}*/
}