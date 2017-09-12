//convert video to image
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp> // videoioのヘッダーをインクルード
#include <opencv2/highgui.hpp> // highguiのヘッダーをインクルード
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[])
{
  char filename[80];
  int filenumber;
  // （1）動画ファイルを開くための準備を行/       う
  cv::VideoCapture camera1(argv[1]);
  int frameNum = camera1.get(cv::CAP_PROP_FRAME_COUNT);
  int frameStep = frameNum/50;
  // （2）動画ファイルが正しく開けているかをチェックする（正しく開けていなければエラー終了する）
  if (!camera1.isOpened())
    return -1;

  // 画像データを格納するための変数を宣言する
  cv::Mat frame;
  filenumber =1;
  int framePtr = 0;
  std::string directory_name = std::string(argv[2]);
  while(1)
  {
    // （3）動画ファイルから1フレーム分の画像データを取得して、変数frameに格納する
    camera1 >> frame;

    // 画像データ取得を取得できたら書き込み。
    if (!frame.empty()&&framePtr%frameStep==0){
        sprintf(filename,"/img%d.jpg",filenumber++);
        std::string file_name_str = directory_name + std::string(filename);
        std::cout<<"7"<<std::endl;
        cv::imwrite(file_name_str,frame);
    }
    std::cout<<filename<<std::endl;

    framePtr++;

    if(frame.empty()) break;
    if (cv::waitKey(30) >= 0) break;

  }
  return 0;
}