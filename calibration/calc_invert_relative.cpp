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


int main(int argc, const char * argv[])
{

  //2つのシリアル番号を引数にとる
  if(argc != 3){
    cout << "please run this file with 2 serials" << endl;
    cout << "ex) bin/calc_invert_relative serial1 serial2" << endl;
    return -1;
  }
  string serial1(argv[1]);
  string serial2(argv[2]);

  cv::Mat calibmat;

  cv::FileStorage fs1("output_relative/relative_"+serial1+"_to_"+serial2+".txt", cv::FileStorage::READ);
  fs1["calibMatrix"] >> calibmat;
  fs1.release();

  //逆の変換にする
  cv::invert(calibmat, calibmat);


  // これまでの方式で残しておく ***********************************************************************
  cv::FileStorage fs2("output_relative/relative_"+serial2+"_to_"+serial1+".txt", cv::FileStorage::WRITE);
  fs2 << "serial1" << serial2;
  fs2 << "serial2" << serial1;
  fs2 << "calibMatrix" << calibmat;
  fs2.release();


  cout << "*****************************************************************" << endl;
  cout << calibmat << endl;
  cout << "*****************************************************************" << endl;


  return 0;
}

