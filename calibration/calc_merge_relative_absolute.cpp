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
    cout << "ex) bin/calc_merge_relative_absolute serial1 serial2" << endl;
    return -1;
  }
  string serial1(argv[1]);
  string serial2(argv[2]);

  cv::Mat calibmat1, calibmat2;

  cv::FileStorage fs1("output_relative/relative_"+serial1+"_to_"+serial2+".txt", cv::FileStorage::READ);
  cv::FileStorage fs2("output_absolute/absolute_"+serial2+"_to_world.txt", cv::FileStorage::READ);

  fs1["calibMatrix"] >> calibmat1;
  fs2["calibMatrix"] >> calibmat2;

  fs1.release();
  fs2.release();

  cv::Mat calibmat = calibmat2*calibmat1;



  // これまでの方式で残しておく ***********************************************************************
  cv::FileStorage fs("output_absolute/absolute_"+serial1+"_to_world.txt", cv::FileStorage::WRITE);
  fs << "serial" << serial1;
  fs << "calibMatrix" << calibmat;
  fs.release();


  cout << "*****************************************************************" << endl;
  cout << calibmat << endl;
  cout << "*****************************************************************" << endl;


  return 0;
}

