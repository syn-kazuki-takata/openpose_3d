#ifndef KINECTNI2_H
#define KINECTNI2_H

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <Eigen/LU>
#include <Eigen/Core>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cctype>
#include <algorithm>
#include <cmath>

#include <OpenNI.h>
#include <NiTE.h>
#include <opencv2/opencv.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/statistical_outlier_removal.h>



namespace pt = boost::posix_time;
using namespace std;

class KinectNI2
{
public:
    KinectNI2() :
            pCloud (new pcl::PointCloud<pcl::PointXYZRGBA> ()),
            pCloud_scene (new pcl::PointCloud<pcl::PointXYZRGBA> ()),
            pCloud_user (new pcl::PointCloud<pcl::PointXYZRGBA> ()),
            pCloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>())
    {}

    ~KinectNI2()
    {
      if(colorStream.isValid()){
        colorStream.stop();
        colorStream.destroy();
      }
      if(depthStream.isValid()){
        depthStream.stop();
        depthStream.destroy();
      }
      if(irStream.isValid()){
        irStream.stop();
        irStream.destroy();
      }
      device.close();
    }

    void initialize( const char* uri = openni::ANY_DEVICE )
    {
      // デバイスを取得する
      openni::Status ret = device.open( uri );
      if ( ret != openni::STATUS_OK ) {
        throw std::runtime_error( "openni::Device::open() failed." );
      }

      //load serial
      char serialNumber[64];
      ret = device.getProperty(ONI_DEVICE_PROPERTY_SERIAL_NUMBER, &serialNumber);
      if(ret == openni::STATUS_OK)
        this->serial.assign(serialNumber, strlen(serialNumber));
      else
        std::cout << "property not supported" << std::endl;

      //load camera param
      if(this->serial!="") {
        loadConfig();
      }
      // ビデオストリームを有効にする
      if (device.hasSensor(openni::SENSOR_COLOR)) {
        ret = colorStream.create( device, openni::SENSOR_COLOR );
        const openni::Array<openni::VideoMode> *supportedVideoModes =
                &(colorStream.getSensorInfo().getSupportedVideoModes());
        ret = colorStream.start();
        if ( ret != openni::STATUS_OK ) {
          throw std::runtime_error( "SENSOR_COLOR failed." );
        }
      }

      if (device.hasSensor(openni::SENSOR_DEPTH)) {
        ret = depthStream.create( device, openni::SENSOR_DEPTH );

        const openni::Array<openni::VideoMode> *supportedVideoModes =
                &(depthStream.getSensorInfo().getSupportedVideoModes());
        int numOfVideoModes = supportedVideoModes->getSize();
        if (numOfVideoModes == 0) {
          throw std::runtime_error( "VideoMode failed." );
        }

        for (int i = 0; i < numOfVideoModes; i++) {
          openni::VideoMode vm = (*supportedVideoModes)[i];
          cout << i << endl;
          printf("%c. %dx%d at %dfps with %d format \r\n",
                 '0' + i,
                 vm.getResolutionX(),
                 vm.getResolutionY(),
                 vm.getFps(),
                 vm.getPixelFormat());
        }
        openni::VideoMode vm = (*supportedVideoModes)[1];// NiTE can only work with VGA resolution
        ret = depthStream.setVideoMode(vm);
        ret = depthStream.start();
        if ( ret != openni::STATUS_OK ) {
          throw std::runtime_error( "SENSOR_DEPTH failed." );
        }
      }

      if (device.hasSensor(openni::SENSOR_IR)) {
        ret = irStream.create( device, openni::SENSOR_IR );

        const openni::Array<openni::VideoMode> *supportedVideoModes =
                &(irStream.getSensorInfo().getSupportedVideoModes());
        int numOfVideoModes = supportedVideoModes->getSize();
        if (numOfVideoModes == 0) {
          throw std::runtime_error( "VideoMode failed." );
        }

        for (int i = 0; i < numOfVideoModes; i++) {
          openni::VideoMode vm = (*supportedVideoModes)[i];
          printf("%c. %dx%d at %dfps with %d format \r\n",
                 '0' + i,
                 vm.getResolutionX(),
                 vm.getResolutionY(),
                 vm.getFps(),
                 vm.getPixelFormat());
        }
        openni::VideoMode vm = (*supportedVideoModes)[1];
        ret = irStream.setVideoMode(vm);

        ret = irStream.start();
        if ( ret != openni::STATUS_OK ) {
          throw std::runtime_error( "SENSOR_IR failed." );
        }
      }

      //mirror if supported
      if(!device.isFile()) {
        ret = colorStream.setMirroringEnabled(true);
        ret = depthStream.setMirroringEnabled(true);
        ret = irStream.setMirroringEnabled(true);
        if ( ret != openni::STATUS_OK ) {
          throw std::runtime_error( "mirroring failed." );
        }
      }

      // colorはデプスサイズに変換する
      if (colorStream.isValid() && depthStream.isValid()) {
        ret = device.setDepthColorSyncEnabled(true);
        // if ( ret != openni::STATUS_OK )
        //   throw std::runtime_error( "sync failed." );
        //ret = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_OFF);
        ret = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
        if ( ret != openni::STATUS_OK )
          throw std::runtime_error( "registration failed." );
      }

      // ユーザートラッカーを有効にする(deviceを指定しないとANY_DEVICE)
      userTracker.create(&device);

      // URIを保存しておく
      this->uri = uri;

      cout << "kinect_" << serial << " initialized" << endl;
    }

    // フレームの更新処理
    void update()
    {
      openni::VideoFrameRef colorFrame;
      openni::VideoFrameRef depthFrame;
      openni::VideoFrameRef irFrame;
      nite::UserTrackerFrameRef userFrame;

      // 更新されたフレームを取得する
      openni::Status ret;
      if (colorStream.isValid())
        ret = colorStream.readFrame( &colorFrame );
      if (irStream.isValid())
        ret = irStream.readFrame( &irFrame );
      if (userTracker.isValid()) {
        userTracker.readFrame( &userFrame );
        depthFrame = userFrame.getDepthFrame(); // depthFrameはNiTEから取得する
      }
      if ( ret != openni::STATUS_OK ) {
        throw std::runtime_error( "readFrame failed." );
      }

      // 画像データをアップデートする
      if (colorStream.isValid()){
          colorImage = updateColorImage( colorFrame );
      }
      if (irStream.isValid()){
          irImage = updateIrImage( irFrame );
      }
      if (userTracker.isValid()) {
        rawDepthImage = updateRawDepthImage( depthFrame );
        depthImage = updateDepthImage( depthFrame );
        userImage = updateUserImage(userFrame); //メンバ変数のpLabels使ってない
        //ポイントクラウドデータをアップデートする
        updatePointCloud();
      }

    }

    // get **************************************************************************************************************************
    cv::Mat getColorImage(){ //BGR
      return colorImage; // clone()は呼び出し側で
    }
    cv::Mat getDepthImage(){ //8bit, near is high
      return depthImage;
    }
    cv::Mat getRawDepthImage(){ //16bit, far is high
      return rawDepthImage;
    }
    cv::Mat getIrImage(){
        boost::mutex::scoped_lock lock(mutex_);
        return irImage;
    }
    cv::Mat getUndistortedIrImage(){
        cv::Mat undistortedIrImage;
        boost::mutex::scoped_lock lock(mutex_);
        cv::undistort(irImage, undistortedIrImage, cameraMatrix, distCoeffs);
        return undistortedIrImage;
    }
    cv::Mat getUserImage(){
      return userImage;
    }
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getPCloud(int type = 0){
        if(type==0){
          pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_return(pCloud);
          return cloud_return;
        }
        else if(type==1){
          return pCloud;
        }
    }
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getPCloud_scene(int type = 0){
        if(type==0){
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_return(pCloud_scene);
            return cloud_return;
        }
        else if(type==1){
            return pCloud_scene;
        }
    }
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getPCloud_user(int type = 0){
        if(type==0){
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_return(pCloud_user);
            return cloud_return;
        }
        else if(type==1){
            return pCloud_user;
        }
    }
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr getPCloud_filtered(int type=1){
        if(type==1){
            pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
            sor.setInputCloud (pCloud);
            sor.setMeanK (50);
            sor.setStddevMulThresh (1.0);
            sor.filter (*pCloud_filtered);
            return pCloud_filtered;
        }
    }
    const std::string& getUri() const {
      return uri;
    }
    const std::string& getSerial() const {
      return serial;
    }
    int getDeviceIndex(){
      return deviceIndex;
    }
    void getIrParams(float& fx, float& fy, float& cx, float& cy, float& k1, float& k2, float& p1, float& p2, float& k3){
      fx = this->fx;
      fy = this->fy;
      cx = this->cx;
      cy = this->cy;
      k1 = this->k1;
      k2 = this->k2;
      p1 = this->p1;
      p2 = this->p2;
      k3 = this->k3;
    }

    // set **************************************************************************************************************************
    void setIrParams(float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2, float k3){
      this->fx = fx;
      this->fy = fy;
      this->cx = cx;
      this->cy = cy;
      this->k1 = k1;
      this->k2 = k2;
      this->p1 = p1;
      this->p2 = p2;
      this->k3 = k3;
    }

    void setDeviceIndex(int deviceIndex){
      this->deviceIndex = deviceIndex;
    }

    void setCalibMatrix(){
      // ファイル入力ストリームの初期化
      cv::FileStorage fs("calib/output/exParams_"+serial+".txt", cv::FileStorage::READ);
      std::cout << "calib/output/exParams_"+serial+".txt" << std::endl;
      cv::Mat rot, tr;
      fs["rotation"] >> rot;
      fs["translation"] >> tr;
      std::cout << rot << std::endl << tr << std::endl;

      for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
          //calibMatrix (i,j) = rot.at<float>(i,j);
          calibMatrix (i,j) = (float)rot.at<double>(i,j);
      for(int i=0; i<3; i++)
        //calibMatrix (i, 3) = tr.at<float>(i);
        calibMatrix (i, 3) = (float)tr.at<double>(i);
      std::cout << "calib" << calibMatrix << std::endl;
      fs.release();

      calibMatrix = calibMatrix.inverse(); //カメラから世界座標の変換だった場合必要（全てtz>0のとき）


      // OpenNIの座標系がKinectSDKと違っていたための補正 *****************************
      // ほんとはキャリブレーション側でやるべきかも
      // カメラが自分の目だとしたとき
      // SDK:    x->右、y->上、z->前
      // OpenNI: x->左、y->下、z->前
      //Eigen::Matrix4f correction = Eigen::Matrix4f::Identity();
      //correction(0,0) = -1;
      //correction(1,1) = -1;
      //calibMatrix = calibMatrix*correction;
      // *************************************************************************

    }

    void loadConfig() {
      if(this->serial == "")
        throw std::runtime_error( "loadConfig: invalid serial number" );

      cv::Mat cameraMatrix, distCoeffs, Rmat, T;
      Eigen::Matrix4f tmpMatrix = Eigen::Matrix4f::Identity();

      cameraMatrix = cv::Mat::eye(3,3,CV_64FC1);
      distCoeffs = cv::Mat(5,1,CV_64FC1, cv::Scalar::all(0));
      Rmat = cv::Mat(3,3,CV_64FC1,cv::Scalar::all(0));
      T = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));


        //TODO catch err
      std::cout << "serial: " << this->serial << std::endl;
      cv::FileStorage fs("kinect2.yaml", cv::FileStorage::READ);
      fs["kinect2_"+serial]["camera_matrix"] >> cameraMatrix;
      fs["kinect2_"+serial]["distortion_coefficients"] >> distCoeffs;
      fs["kinect2_"+serial]["rotation"] >> Rmat;
      fs["kinect2_"+serial]["translation"] >> T;
      fs.release();


        setIrParams(cameraMatrix.ptr<double>(0)[0],
                  cameraMatrix.ptr<double>(0)[4],
                  cameraMatrix.ptr<double>(0)[2],
                  cameraMatrix.ptr<double>(0)[5],
                  distCoeffs.ptr<double>(0)[0],
                  distCoeffs.ptr<double>(0)[1],
                  distCoeffs.ptr<double>(0)[2],
                  distCoeffs.ptr<double>(0)[3],
                  distCoeffs.ptr<double>(0)[4]
      );
      //dump ir
      std::cout << "fx=" << this->fx << ","
      << "fy=" << this->fy << ","
      << "cx=" << this->cx << ","
      << "cy=" << this->cy << std::endl;




      //追加(個人的にfloatがいい)
      cameraMatrix.convertTo(this->cameraMatrix, CV_32FC1);
      distCoeffs.convertTo(this->distCoeffs, CV_32FC1);

      for(int i=0; i<3; i++)
          for(int j=0; j<3; j++)
              tmpMatrix (i,j) = Rmat.at<double>(i, j);
      for(int i=0; i<3; i++)
          tmpMatrix(i, 3) = T.at<double>(i);
      this->calibMatrix = tmpMatrix.inverse();


    }


    //save *************************************************************************************
    void savePcloud(){
        pcl::io::savePCDFileBinary("pCloud_"+serial+".pcd", *pCloud);
    }
    void savePcloud_user(){
        pcl::io::savePCDFileBinary("pCloud_user_"+serial+".pcd", *pCloud_user);
    }
    void savePcloud_scene(){
        pcl::io::savePCDFileBinary("pCloud_scene_"+serial+".pcd", *pCloud_scene);
    }

private:

    // カラーストリームを表示できる形に変換する
    cv::Mat updateColorImage( const openni::VideoFrameRef& colorFrame )
    {
      // OpenCV の形に変換する
      cv::Mat colorImage = cv::Mat( colorFrame.getHeight(),
                                    colorFrame.getWidth(),
                                    CV_8UC3, (unsigned char*)colorFrame.getData() );
      // RGB の並びを BGR に変換する
      cv::cvtColor( colorImage, colorImage, CV_RGB2BGR );

      return colorImage;
    }

    cv::Mat updateRawDepthImage( const openni::VideoFrameRef& depthFrame )
    {
      // 距離データを画像化する(16bit)
      cv::Mat rawDepthImage = cv::Mat( depthFrame.getHeight(),
                                       depthFrame.getWidth(),
                                       CV_16UC1, (unsigned short*)depthFrame.getData() );

        //512*424で必要な部分だけメモリをコピーしてくる
        cv::Mat rawDepthImage_resized = cv::Mat(424, 512, CV_16UC1);
        unsigned short* depth = (unsigned short*)rawDepthImage.data;
        unsigned short* depth_resized = (unsigned short*)rawDepthImage_resized.data;
        int depthIndex, depth_resizedIndex;
        for(int i=0; i<424; i++){
            depthIndex = i*640;
            depth_resizedIndex = i*512;
            memcpy(&depth_resized[depth_resizedIndex], &depth[depthIndex], 512*2);
        }
        return rawDepthImage_resized;
    }

    cv::Mat updateDepthImage( const openni::VideoFrameRef& depthFrame )
    {
      // 距離データを画像化する(16bit)
      cv::Mat depthImage = cv::Mat( depthFrame.getHeight(),
                                    depthFrame.getWidth(),
                                    CV_16UC1, (unsigned short*)depthFrame.getData() );
      depthImage.convertTo( depthImage, CV_8U, 255.0 / 10000 );

        //512*424で必要な部分だけメモリをコピーしてくる
        cv::Mat depthImage_resized = cv::Mat(424, 512, CV_8UC1);
        unsigned char* depth = (unsigned char*)depthImage.data;
        unsigned char* depth_resized = (unsigned char*)depthImage_resized.data;
        int depthIndex, depth_resizedIndex;
        for(int i=0; i<424; i++){
            depthIndex = i*640;
            depth_resizedIndex = i*512;
            memcpy(&depth_resized[depth_resizedIndex], &depth[depthIndex], 512);
        }
        return depthImage_resized;
    }

    cv::Mat updateIrImage( const openni::VideoFrameRef& irFrame )
    {
      boost::mutex::scoped_lock lock(mutex_);
      // 距離データを画像化する(16bit)
      cv::Mat irImage = cv::Mat( irFrame.getHeight(),
                                 irFrame.getWidth(),
                                 CV_16UC1, (unsigned short*)irFrame.getData() );
      irImage.convertTo( irImage, CV_8U, 255.0 / 10000 );
      cv::cvtColor( irImage, irImage, CV_GRAY2RGB ); // for calibration
      return irImage;
    }



    cv::Mat updateUserImage( nite::UserTrackerFrameRef& userFrame )
    {
      static const cv::Scalar colors[] = {
              cv::Scalar(255, 0, 0),
              cv::Scalar(0, 255, 0),
              cv::Scalar(0, 0, 255),
              cv::Scalar(255, 255, 0),
              cv::Scalar(255, 0, 255),
              cv::Scalar(0, 255, 255),
              cv::Scalar(127, 0, 0),
              cv::Scalar(0, 127, 0),
              cv::Scalar(0, 0, 127),
              cv::Scalar(127, 127, 0),
      };

      cv::Mat userImage;

      //get depth frame
      openni::VideoFrameRef depthFrame = userFrame.getDepthFrame();
      if(depthFrame.isValid()){
        userImage = cv::Mat(depthFrame.getHeight(), depthFrame.getWidth(), CV_8UC3);

        // get data and userId
        openni::DepthPixel* depth = (openni::DepthPixel*)depthFrame.getData();
        pLabels = userFrame.getUserMap().getPixels();
          //時間かかるからとりあえず省略。どうせuserImage使わない。pLabelsは使う。
          /*
        //for(int i=0; i<(depthFrame.getDataSize()/sizeof(openni::DepthPixel)); i++){
        for(int i=0; i<(depthFrame.getWidth()*depthFrame.getHeight()); i++){
          int index = i*3;
          //visualize depth data
          uchar* data = &userImage.data[index];
          if(pLabels[i]!=0){
            data[0] = colors[pLabels[i]][0];
            data[1] = colors[pLabels[i]][1];
            data[2] = colors[pLabels[i]][2];
          }
          else{
            int gray = (depth[i]*255)/10000;
            data[0] = gray;
            data[1] = gray;
            data[2] = gray;
          }
        }
           */
      }
      return userImage;
    }

    void updatePointCloud(){
      unsigned char* color = (unsigned char*)colorImage.data;
      unsigned short* depth = (unsigned short*)rawDepthImage.data;
      pCloud->clear();
      pCloud_scene->clear();
      pCloud_user->clear();
      int xpixel, ypixel;
      float xw, yw, zw;
      int niteIndex = 0;
      for(int i=0; i<rawDepthImage.size().width*rawDepthImage.size().height; i++){ //512*424で回してる
        if(depth[i]!=0 && (color[i*3]!=0 || color[i*3+1]!=0 || color[i*3+2]!=0)){
          xpixel = i%rawDepthImage.size().width;
          ypixel = i/rawDepthImage.size().width;
          zw = depth[i];
          zw/=1000;
          xw = (xpixel-cx)*zw/fx;
          yw = (ypixel-cy)*zw/fy;
          pcl::PointXYZRGBA point;
          point.x = xw;
          point.y = yw;
          point.z = zw;
          point.b = color[i*3];
          point.g = color[i*3+1];
          point.r = color[i*3+2];
          pCloud->push_back(point);
          if(pLabels[niteIndex]!=0){ //user部分
            pCloud_user->push_back(point);
          }
          else{ //scene部分
            pCloud_scene->push_back(point);
          }
        }
        niteIndex++;
        if(i%512 == 511) niteIndex+=128; //pLabelsだけ640*480ではいってるから帳尻合わせ。
      }
        pcl::transformPointCloud(*pCloud, *pCloud, calibMatrix);
        pcl::transformPointCloud(*pCloud_scene, *pCloud_scene, calibMatrix);
        pcl::transformPointCloud(*pCloud_user, *pCloud_user, calibMatrix);
    }

private:
    openni::Device device;
    std::string uri;
    std::string serial;
    int deviceIndex = 0;

    openni::VideoStream colorStream;
    openni::VideoStream depthStream;
    openni::VideoStream irStream;
    nite::UserTracker userTracker;
    const nite::UserId* pLabels;

    boost::mutex mutex_;

    //IR camera parameter, can set if necessary
    //TODO: read from file
    float fx = 365.275;
    float fy = 365.275;
    float cx = 260.131;
    float cy = 202.222;
    float k1 = 0.0914707;
    float k2 = -0.266826;
    float p1 = 0;
    float p2 = 0;
    float k3 = 0.0902061;
    Eigen::Matrix4f calibMatrix = Eigen::Matrix4f::Identity();

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    // can get following data ******************************************

    cv::Mat colorImage; //BGR
    cv::Mat rawDepthImage;
    cv::Mat depthImage;
    cv::Mat irImage;
    cv::Mat userImage;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pCloud;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pCloud_scene;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pCloud_user;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pCloud_filtered;
};

#endif // KINECTNI2_H
