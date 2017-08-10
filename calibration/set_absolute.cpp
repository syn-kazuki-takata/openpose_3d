#include <fstream>
#include "kinectNI2.hpp"

#define DEVICE_NUM 1
#define SQUARE_SIZE 0.105
#define CORNERS_WIDTH 7
#define CORNERS_HEIGHT 4
#define CORNERS_NUM CORNERS_WIDTH*CORNERS_HEIGHT

using namespace std;
using namespace cv;

string IntToString(int number)
{
  stringstream ss;
  ss << number;
  return ss.str();
}
string FloatToString(float number)
{
  stringstream ss;
  ss << number;
  return ss.str();
}

class CalibrationApp
{
public:

  void initialize(string serial1)
  {

    this->serial1 = serial1;

    // 接続されているデバイスの一覧を取得する
    openni::Array<openni::DeviceInfo> deviceInfoList;
	openni::OpenNI::enumerateDevices( &deviceInfoList );
    
	std::cout << "接続されているデバイスの数 : " << deviceInfoList.getSize()/3 << std::endl; // for libfreenect2

    for ( int i = 0; i < deviceInfoList.getSize(); i+=3 ) {
      std::cout << deviceInfoList[i].getName() << ", "
      << deviceInfoList[i].getVendor() << ", "
      << deviceInfoList[i].getUri() << std::endl;
      openDevice( deviceInfoList[i].getUri() );
    }

    for ( std::vector<KinectNI2*>::iterator it = sensors.begin();
          it != sensors.end(); ++it ) {

    }

  }

  void update()
  {
    for ( std::vector<KinectNI2*>::iterator it = sensors.begin();
          it != sensors.end(); ++it ) {
      (*it)->update();
      cv::imshow((*it)->getSerial(),(*it)->getUndistortedIrImage());
    }
  }

  void detectCorners()
  {

    //条件
    cv::Size patternSize(CORNERS_WIDTH, CORNERS_HEIGHT);
    cv::Size winSizeForSubPix(5, 5);


    //どうせkinect１台なんだけど、今までと同じ書き方。
    for ( std::vector<KinectNI2*>::iterator it = sensors.begin();
          it != sensors.end(); ++it ) {

      int index = (*it)->getDeviceIndex();
      cv::Mat irImage = (*it)->getUndistortedIrImage();
      string serial = (*it)->getSerial();

      //コーナー検出
      cv::Mat corner_tmp;
      bool is_find = cv::findChessboardCorners(irImage, patternSize, corner_tmp);
      //見つからなかったら終了
      if (is_find != true)
      {
        cout << "sorry, calibration failed. please press space key one more time" << endl;
        irImage_corners[index] = irImage;
        corners[index] = corner_tmp;
        cv::imshow((*it)->getUri()+"_corners", irImage);
        continue;
      }
      cout << irImage.type() << endl;
      cv::cvtColor(irImage, irImage, CV_RGB2GRAY);
      //サブピクセル推定
      cv::cornerSubPix(irImage, corner_tmp, winSizeForSubPix,
                       cv::Size(-1, -1),
                       cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));
      corners[index] = corner_tmp;
      //コーナー描画
      cv::drawChessboardCorners(irImage, patternSize, corner_tmp, true);
      irImage_corners[index] = irImage;

      //画像を表示
      cv::imshow((*it)->getUri()+"_corners", irImage);

    }
  }


  void calculateCalibrationMatrix()
  {
    //world座標の設定
    float square_size = SQUARE_SIZE;
    cv::Mat worldPoints(CORNERS_NUM, 3, CV_32FC1);
    for (int i = 0; i < CORNERS_NUM; i++){
      worldPoints.at<float>(i, 0) = (2 - (i / CORNERS_WIDTH))*square_size; //Xw
      worldPoints.at<float>(i, 1) = 0; //Yw
      worldPoints.at<float>(i, 2) = ((i % CORNERS_WIDTH) - 3)*square_size; //Zw
    }

    //キャリブレーション
    for ( std::vector<KinectNI2*>::iterator it = sensors.begin();
          it != sensors.end(); ++it ) {
      int index = (*it)->getDeviceIndex();
      string serial = (*it)->getSerial();
      float fx, fy, cx, cy, k1, k2, p1, p2, k3;
      (*it)->getIrParams(fx, fy, cx, cy, k1, k2, p1, p2, k3);
      cv::Mat cameraMatrix = (cv::Mat_<float>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
      cv::Mat distCoeffs = (cv::Mat_<float>(5,1) << k1, k2, p1, p2, k3);
      cv::Mat t, rvec, R;
      cv::solvePnP(worldPoints, corners[index], cameraMatrix, distCoeffs, rvec, t);
      cv::Rodrigues(rvec, R);

      //パラメータを求めるために保存しておく
      if(serial == serial1){
        t1 = t.clone();
        r1 = R.clone();
        cv::Mat calibmat1 = cv::Mat::zeros(4, 4, CV_32FC1);
        for(int i=0; i<3; i++)
          for(int j=0; j<3; j++){
            calibmat1.at<float>(i,j) = r1.at<double>(i,j);
          }
        for(int i=0; i<3; i++){
          calibmat1.at<float>(i,3) = t1.at<double>(i);
        }
        calibmat1.at<float>(3,3) = 1;

        //kinect1はチェスボードに向かった変換
        cv::invert(calibmat1, calibmat1);

        // これまでの方式で残しておく ***********************************************************************
        cv::FileStorage fs1("output_absolute/absolute_"+serial1+"_to_world.txt", cv::FileStorage::WRITE);
        cv::FileStorage fs2("output_absolute/absolute_"+serial1+"_to_world_org.txt", cv::FileStorage::WRITE);
        fs1 << "serial" << serial1;
        fs1 << "calibMatrix" << calibmat1;
        fs2 << "serial" << serial1;
        fs2 << "calibMatrix" << calibmat1;
        fs1.release();
        fs2.release();

        cout << "*****************************************************************" << endl;
        cout << calibmat1 << endl;
        cout << "*****************************************************************" << endl;

      }
      else{
        cout << "error" << endl;
      }

    }


  }


private:
  
  void openDevice( const char* uri )
  {
    KinectNI2* sensor = new KinectNI2();
    sensor->initialize( uri );
    sensor->setDeviceIndex(0);
    if(sensor->getSerial() == serial1){
      sensors.push_back( sensor );
    }
  }
  
private:

  std::vector<KinectNI2*> sensors;
  cv::Mat irImage_corners[DEVICE_NUM]; //床にキャリブレーションボード（y=0）
  cv::Mat corners[DEVICE_NUM]; //床にキャリブレーションボード（y=0）
  string serial1;
  cv::Mat r1, t1;

};

int main(int argc, const char * argv[])
{

  //1つのシリアル番号を引数にとる
  if(argc != 2){
    cout << "please run this file with a serial" << endl;
    cout << "ex) bin/set_absolute serial" << endl;
    return -1;
  }
  string serial1(argv[1]);

  try {
    // OpenNI を初期化する
    openni::OpenNI::initialize();
    // NiTE を初期化する
    nite::NiTE::initialize();



    CalibrationApp app;
    //２つのserial番号を指定する。１つ目から２つ目の外部パラメータを求めるプログラム。
    app.initialize(serial1);
    //app.update();
    while ( 1 ) {
      app.update();

      int key = cv::waitKey( 33 );
      if ( key == 'q' ) {
        break;
      }
      if(key=='1'){
        app.detectCorners();
      }
      if(key=='2'){
        app.calculateCalibrationMatrix();
      }

    }
  }
  catch ( std::exception& ) {
    std::cout << openni::OpenNI::getExtendedError() << std::endl;
  }

  return 0;
}

