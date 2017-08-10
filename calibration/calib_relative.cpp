#include <fstream>
#include "kinectNI2.hpp"

#define DEVICE_NUM 2
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

  void initialize(string serial1, string serial2)
  {

    this->serial1 = serial1;
    this->serial2 = serial2;

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

    //それぞれのキャリブレーション
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

      //2つのkinect間のパラメータを求めるために保存しておく
      if(serial == serial1){
        t1 = t.clone();
        r1 = R.clone();
      }
      else if(serial == serial2){
        t2 = t.clone();
        r2 = R.clone();
      }
      else{
        cout << "error" << endl;
      }

    }


    //kinect1からkinect2への変換を求める
    cv::Mat calibmat1 = cv::Mat::zeros(4, 4, CV_32FC1);
    cv::Mat calibmat2 = cv::Mat::zeros(4, 4, CV_32FC1);
    for(int i=0; i<3; i++)
      for(int j=0; j<3; j++){
        calibmat1.at<float>(i,j) = r1.at<double>(i,j);
        calibmat2.at<float>(i,j) = r2.at<double>(i,j);
      }
    for(int i=0; i<3; i++){
      calibmat1.at<float>(i,3) = t1.at<double>(i);
      calibmat2.at<float>(i,3) = t2.at<double>(i);
    }
    calibmat1.at<float>(3,3) = 1;
    calibmat2.at<float>(3,3) = 1;

    //kinect1はチェスボードに向かった変換
    cv::invert(calibmat1, calibmat1);

    //2つをかける
    cv::Mat calibmat = calibmat2*calibmat1;

    // これまでの方式で残しておく ***********************************************************************
    cv::FileStorage fs("output_relative/relative_"+serial1+"_to_"+serial2+".txt", cv::FileStorage::WRITE);
    fs << "serial1" << serial1;
    fs << "serial2" << serial2;
    fs << "calibMatrix" << calibmat;
    fs.release();


    cout << "*****************************************************************" << endl;
    cout << calibmat << endl;
    cout << "*****************************************************************" << endl;


  }


private:
  
  void openDevice( const char* uri )
  {
    KinectNI2* sensor = new KinectNI2();
    sensor->initialize( uri );
    sensor->setDeviceIndex(count);
    if(sensor->getSerial() == serial1 || sensor->getSerial() == serial2){
      sensors.push_back( sensor );
      count++;
    }
  }
  
private:

  std::vector<KinectNI2*> sensors;
  cv::Mat irImage_corners[DEVICE_NUM]; //床にキャリブレーションボード（y=0）
  cv::Mat corners[DEVICE_NUM]; //床にキャリブレーションボード（y=0）
  string serial1, serial2;
  cv::Mat r1, t1, r2, t2;
  int count = 0; // deviceIndexを決める

};

int main(int argc, const char * argv[])
{

  //2つのシリアル番号を引数にとる
  if(argc != 3){
    cout << "please run this file with two serials" << endl;
    cout << "ex) bin/calib_relative serial1 serial2" << endl;
    return -1;
  }
  string serial1(argv[1]);
  string serial2(argv[2]);

  try {
    // OpenNI を初期化する
    openni::OpenNI::initialize();
    // NiTE を初期化する
    nite::NiTE::initialize();



    CalibrationApp app;
    //２つのserial番号を指定する。１つ目から２つ目の外部パラメータを求めるプログラム。
    app.initialize(serial1, serial2);
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

