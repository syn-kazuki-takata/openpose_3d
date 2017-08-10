#include <fstream>
#include "kinectNI2.hpp"


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



class DataHolder
{
public:
    DataHolder(){};

    void initialize(string serial){
      this->serial = serial;

      cv::Mat cameraMatrix, distCoeffs, Rmat, T;

      cameraMatrix = cv::Mat::eye(3,3,CV_64FC1);
      distCoeffs = cv::Mat(5,1,CV_64FC1, cv::Scalar::all(0));
      Rmat = cv::Mat(3,3,CV_64FC1,cv::Scalar::all(0));
      T = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));

      //TODO catch err
      std::cout << "serial: " << this->serial << std::endl;
      cv::FileStorage fs1("kinect2.yaml", cv::FileStorage::READ);
      fs1["kinect2_"+serial]["camera_matrix"] >> cameraMatrix;
      fs1["kinect2_"+serial]["distortion_coefficients"] >> distCoeffs;
      fs1.release();

      cv::FileStorage fs2("output_absolute/absolute_"+serial+"_to_world.txt", cv::FileStorage::READ);
      cv::Mat calibmat;
      fs2["calibMatrix"] >> calibmat;
      fs2.release();

      calibmat = calibmat.inv();

      for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
          Rmat.at<double>(i,j) = calibmat.at<float>(i,j);
        }
      }
      for(int i=0; i<3; i++){
        T.at<double>(i) = calibmat.at<float>(i,3);
      }


      //データを持っておく
      this->camera_matrix = cameraMatrix;
      this->distortion_coefficients = distCoeffs;
      this->rotation = Rmat;
      this->translation = T;

    }


public:
    string serial;
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    cv::Mat rotation;
    cv::Mat translation;
};




int main(int argc, const char * argv[])
{

  //複数のシリアル番号を引数にとる
  if(argc < 3){
    cout << "please run this file with multiple serials" << endl;
    cout << "ex) bin/calc_make_yaml serial1 serial2 serial3 ..." << endl;
    return -1;
  }

  int devicenum = argc-1;

  std::vector<DataHolder*> datas;

  for(int i=1; i<=devicenum; i++){
    string serial(argv[i]);
    DataHolder* data = new DataHolder;
    data->initialize(serial);
    datas.push_back(data);
  }

  cv::FileStorage outfs("output_yaml/kinect2_out.yaml", cv::FileStorage::WRITE);
  for(int i=0; i<devicenum; i++){
    outfs << "kinect2_" + datas[i]->serial << "{";
    outfs << "camera_matrix" << datas[i]->camera_matrix;
    outfs << "distortion_coefficients" << datas[i]->distortion_coefficients;
    outfs << "rotation" << datas[i]->rotation;
    outfs << "translation" << datas[i]->translation;
    outfs << "}";
  }
  outfs.release();

  return 0;
}

