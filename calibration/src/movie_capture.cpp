#include "opencv2/opencv.hpp"

int main(int argh, char* argv[])
{
    int camera_num_1 = std::stoi(argv[1]);
    //int camera_num_2 = std::stoi(argv[2]);
    cv::VideoCapture camera1(camera_num_1);
    //cv::VideoCapture camera2(camera_num_2);

    //出力動画ファイルの指定
    cv::VideoWriter writer1(argv[2], cv::VideoWriter::fourcc('M','J','P','G'), camera1.get(cv::CAP_PROP_FPS), cv::Size((int)camera1.get(cv::CAP_PROP_FRAME_WIDTH), (int)camera1.get(cv::CAP_PROP_FRAME_HEIGHT)));
    //cv::VideoWriter writer2(argv[4], cv::VideoWriter::fourcc('M','J','P','G'), camera2.get(cv::CAP_PROP_FPS), cv::Size((int)camera2.get(cv::CAP_PROP_FRAME_WIDTH), (int)camera2.get(cv::CAP_PROP_FRAME_HEIGHT)));

    if(!camera1.isOpened())//カメラデバイスが正常にオープンしたか確認．
    {
        //読み込みに失敗したときの処理
        return -1;
    }

    while(1)//無限ループ
    {
        cv::Mat frame1, frame2;
        camera1 >> frame1;
        writer1 << frame1;
        //camera2 >> frame2;
        //writer2 << frame2;

        cv::imshow("camera1", frame1);
        //cv::imshow("camera2", frame2);

        int key = cv::waitKey(30);
        if(key == 113)//qボタンが押されたとき
        {
            break;//whileループから抜ける．
        }
    }
    cv::destroyAllWindows();
    return 0;
}