#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include "fisheye_calibration.hpp"
#include "fisheye.hpp"

using namespace std;
using namespace cv;

/*
void capture_image(VideoCapture& camera, Mat& frame, string camera_name){
	while(1){
		mtx.lock();
		camera >> frame;
		mtx.unlock();
		imshow(camera_name, frame);
	}
}
*/

int main(int argc, char* argv[]){
	string videoStr = "video";
    string cameraStr = "camera";
    string inputStr = string(argv[1]);
    
    VideoCapture camera;
    string window_name;
    if(inputStr == videoStr){
        camera.open(argv[5]);
    }else if(inputStr == cameraStr){
        int num = stoi(argv[5]);
        camera.open(num);
    }
    if (!camera.isOpened())
    {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }
    camera.set(CV_CAP_PROP_FPS, stoi(argv[2]));
    camera.set(CV_CAP_PROP_FRAME_WIDTH, stoi(argv[3]));
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, stoi(argv[4]));

    //std::cout<<"size:"<<camera.get(cv::CAP_PROP_FRAME_WIDTH)<<","<<camera.get(cv::CAP_PROP_FRAME_HEIGHT)<<std::endl;
    window_name = "camera_" + string(argv[5]);
	int frameNum = 50;
	bool capture_switch = false;
	bool _switch = false;
	vector<Mat> frames_log_raw;
	/*
	vector<thread> capture_image_threads(camera_value);
	for(int i=0; i<camera_value; i++){
		string camera_name = "camera_" + to_string(i);
		capture_image_threads[i] = thread(capture_image, ref(cameras[i]), ref(frames[i]), camera_name);
		capture_image_threads[i].detach();
	}
	*/
	
	while(1){
		if(capture_switch){
			Mat frame;
			camera >> frame;
			imshow(window_name, frame);
			frames_log_raw.push_back(frame);
			cout<<"frame_"<<frames_log_raw.size()<<endl;
		}else{
			Mat frame;
			camera >> frame;
			imshow(window_name, frame);
			cout<<"frame_"<<frames_log_raw.size()<<endl;
		}
		int key = waitKey(30);
		if(key == 's'){
			cout<<"start capture!"<<endl;
			capture_switch = true;
		}
		if(key == 'e'){
			cout<<"finish capture!"<<endl;
			break;
		}
	}

	vector<Mat> frames_log;
	int frame_step = frames_log_raw.size()/frameNum;
	for(int i=0; i<frames_log_raw.size(); i++){
		if(!frames_log_raw[i].empty() && i%frame_step==0){
			frames_log.push_back(frames_log_raw[i]);
		}
	}

	Mat intrinsic;
	Mat distortion;
	vector<Mat> tmp = fisheye_calibration::fisheye_calibration_(frames_log);
	intrinsic = tmp[0];
	distortion = tmp[1];

	string output_file_name = argv[6];
	FileStorage outputfs(output_file_name, FileStorage::WRITE);
    outputfs << "intrinsic" << intrinsic;
    outputfs << "distortion" << distortion;
    outputfs.release();

	for(int i=0; i<frames_log.size(); i++){
		Mat undistorted = fisheye_calibration::undistort(intrinsic, distortion, frames_log[i]);
		while(1){
			imshow("distorted", frames_log[i]);
            imshow("undistorted", undistorted);
            int key = waitKey(0);
            if(key=='n'){
                break;
            }
        }
	}
	destroyAllWindows();
	return 0;
}

/*
int main(int argc, char* argv[]){
	int camera_value = stoi(argv[1]);
	vector<VideoCapture> cameras(camera_value);
	vector<string> window_names(camera_value);
	for(int i=0; i<camera_value; i++){
		//cameras[i] = VideoCapture(stoi(argv[2+i]));
		cameras[i] = VideoCapture(argv[2+i]);
		if(!cameras[i].isOpened()){
	        //読み込みに失敗したときの処理
	        return -1;
	    }
	    window_names[i] = "camera_" + to_string(i);
	}

	int frameNum = 50;
	bool capture_switch = false;
	bool _switch = false;
	vector<vector<Mat>> frames_log_raw(camera_value);
	
	while(1){
		if(capture_switch){
			for(int i=0; i<camera_value; i++){
				Mat frame;
				cameras[i] >> frame;
				imshow(window_names[i], frame);
				frames_log_raw[i].push_back(frame);
			}
			cout<<"frame_"<<frames_log_raw[0].size()<<endl;
		}else{
			for(int i=0; i<camera_value; i++){
				Mat frame;
				cameras[i] >> frame;
				imshow(window_names[i], frame);
			}
			cout<<"frame_"<<frames_log_raw[0].size()<<endl;
		}
		int key = waitKey(30);
		if(key == 's'){
			cout<<"start capture!"<<endl;
			capture_switch = true;
		}
		if(key == 'e'){
			cout<<"finish capture!"<<endl;
			break;
		}
	}

	vector<vector<Mat>> frames_log;
	for(int i=0; i<camera_value; i++){
		vector<Mat> tmp;
		int frame_step = frames_log_raw[i].size()/frameNum;
		for(int j=0; j<frames_log_raw[i].size(); j++){
			if(!frames_log_raw[i][j].empty() && j%frame_step==0){
				tmp.push_back(frames_log_raw[i][j]);
			}
		}
		frames_log.push_back(tmp);
	}

	vector<Mat> intrinsics(camera_value);
	vector<Mat> distortions(camera_value);
	for(int i=0; i<camera_value; i++){
		vector<Mat> tmp = fisheye_calibration::fisheye_calibration_(frames_log[i]);
		intrinsics[i] = tmp[0];
		distortions[i] = tmp[1];
	}

	for(int i=0; i<camera_value; i++){
		for(int j=0; j<frames_log[i].size(); j++){
			Mat undistorted = fisheye_calibration::undistort(intrinsics[i], distortions[i], frames_log[i][j]);
			while(1){
				imshow("distorted", frames_log[i][j]);
	            imshow("undistorted", undistorted);
	            int key = waitKey(0);
	            if(key=='n'){
	                break;
	            }
	        }
		}
	}
	destroyAllWindows();
	return 0;
}
*/