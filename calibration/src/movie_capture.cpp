#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>

/*
int main(int argc, char** argv)
{
  //入力動画ファイルを指定
  cv::VideoCapture cap(std::stoi(argv[1]));
  //出力動画ファイルの指定
  cv::VideoWriter writer(, cv::VideoWriter::fourcc('M','J','P','G'), cap.get(cv::CAP_PROP_FPS), cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

  cv::Mat frame;
  while( 1 )
  {
    //1フレーム読み込み
    cap >> frame;
    if( frame.empty() )
      break;

    //出力動画ファイルへ書き込み
    writer << frame;
  }
  return 0;
}
*/


void get_image(cv::Mat &frame, cv::VideoCapture &camera, std::mutex &mutex){
    while(1){
        std::cout<<"get image!"<<std::endl;
        mutex.lock();
        camera >> frame;
        mutex.unlock();
    }
}

void cap_image(cv::Mat &frame, cv::VideoWriter &writer, std::mutex &mutex){
    while(1){
       if(!frame.empty()){
            mutex.lock();
            writer << frame;
            mutex.unlock();
        }
    }
}

int main(int argh, char* argv[])
{
    int camera_num = std::stoi(argv[1]);
    std::vector<cv::VideoCapture> cameras;
    std::vector<cv::VideoWriter> writers;
    std::vector<std::mutex> mutexs(camera_num);
    std::vector<std::thread> get_threads(camera_num);
    std::vector<std::thread> cap_threads(camera_num);
    std::vector<cv::Mat> frames(camera_num);
    for(int i=0; i<camera_num; i++){
        cameras.emplace_back();
        cameras[i] = cv::VideoCapture(std::stoi(argv[5+(2*i)]));
        if(!cameras[i].isOpened())//カメラデバイスが正常にオープンしたか確認．
        {
            //読み込みに失敗したときの処理
            return -1;
        }
        cameras[i].set(CV_CAP_PROP_FPS, std::stoi(argv[2]));
        cameras[i].set(CV_CAP_PROP_FRAME_WIDTH, std::stoi(argv[3]));
        cameras[i].set(CV_CAP_PROP_FRAME_HEIGHT, std::stoi(argv[4]));
        writers.emplace_back();
        std::cout<<"size:"<<(int)cameras[i].get(cv::CAP_PROP_FRAME_WIDTH)<<","<<(int)cameras[i].get(cv::CAP_PROP_FRAME_HEIGHT)<<std::endl;
        writers[i] = cv::VideoWriter(argv[6+(2*i)], cv::VideoWriter::fourcc('M','J','P','G'), cameras[i].get(cv::CAP_PROP_FPS), cv::Size((int)cameras[i].get(cv::CAP_PROP_FRAME_WIDTH), (int)cameras[i].get(cv::CAP_PROP_FRAME_HEIGHT)));
        get_threads[i] = std::thread(get_image, std::ref(frames[i]), std::ref(cameras[i]), std::ref(mutexs[i]));
        get_threads[i].detach();
        cap_threads[i] = std::thread(cap_image, std::ref(frames[i]), std::ref(writers[i]), std::ref(mutexs[i]));
        cap_threads[i].detach();
    }

    std::cout<<"a"<<std::endl;
    while(1){
        for(int i=0; i<camera_num; i++){
            std::cout<<"b"<<std::endl;
            //mutexs[i].lock();
            if(!frames[i].empty()){
                std::cout<<"show image"<<std::endl;
                std::string win_name = "camera_" + std::to_string(i);
                cv::imshow(win_name, frames[i]);
                int key = cv::waitKey(1);
                if(key == 113)//qボタンが押されたとき
                {
                    break;//whileループから抜ける．
                }
            }
            //mutexs[i].unlock();
        }
        if(frames[0].empty()){
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}