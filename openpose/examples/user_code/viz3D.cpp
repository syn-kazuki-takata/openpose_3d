#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

static void help()
{
    cout
    << "--------------------------------------------------------------------------"   << endl
    << "This program shows how to visualize a cube rotated around (1,1,1) and shifted "
    << "using Rodrigues vector."                                                      << endl
    << "Usage:"                                                                       << endl
    << "./widget_pose"                                                                << endl
    << endl;
}

int main(int argc, char* argv[])
{
    help();

    // create window
    viz::Viz3d myWindow("3dpose");

    // show coordinate axes
    //myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem(100));

    // read xml file
    FileStorage fs(argv[1], FileStorage::READ);
    if (!fs.isOpened()){
        cout << "File can not be opened." << endl;
        return -1;
    }

    // get frameNumber
    //int frameNum = (int)fs["frameNum"];

    int framePtr = 0;
    while(!myWindow.wasStopped())
    {
    	// name readframe
    	string frame = "frame" + to_string(framePtr);
		cout<<frame<<endl;
    	// read 3row * 19col Mat
    	Mat bodyJoint;
    	fs[frame] >> bodyJoint;
    	
    	// make vector of body joints Point3d
    	vector<Point3d> bodyJointVec;
    	for(int col=0; col<18; col++){
    		Point3d tmp(bodyJoint.at<double>(0,col), bodyJoint.at<double>(1,col), bodyJoint.at<double>(2,col));
    		bodyJointVec.push_back(tmp);
    	}
    	cout<<"bodyJointVecSize"<<bodyJointVec.size()<<endl;

    	// 関節
		vector<viz::WSphere> bodyJointSphere;
    	for(int i=0; i<bodyJointVec.size(); i++){
    		viz::Color color(255*i/bodyJointVec.size(), 255*i/bodyJointVec.size(), (255-(255*i/bodyJointVec.size())));
    		viz::WSphere sphere(bodyJointVec[i], 0.01, 0.01, color);
    		sphere.setRenderingProperty(viz::LINE_WIDTH, 0.1);
    		string jointName = "bodyJoint" + to_string(i);
    		myWindow.showWidget(jointName, sphere);
    	}

    	// 骨格
    	vector<viz::WLine> lines;
    	lines.push_back(viz::WLine(bodyJointVec[0],bodyJointVec[1],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[0],bodyJointVec[14],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[0],bodyJointVec[15],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[1],bodyJointVec[2],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[1],bodyJointVec[5],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[1],bodyJointVec[8],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[1],bodyJointVec[11],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[2],bodyJointVec[3],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[3],bodyJointVec[4],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[5],bodyJointVec[6],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[6],bodyJointVec[7],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[8],bodyJointVec[9],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[9],bodyJointVec[10],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[11],bodyJointVec[12],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[12],bodyJointVec[13],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[14],bodyJointVec[16],viz::Color::red()));
    	lines.push_back(viz::WLine(bodyJointVec[15],bodyJointVec[17],viz::Color::red()));

    	for(int i= 0;i<lines.size();i++){
    		string lineName = "bodyLines" + to_string(i);
    		lines[i].setRenderingProperty(viz::LINE_WIDTH, 2.0);
            myWindow.showWidget(lineName, lines[i]);
        }
        
        // proceed frame
        framePtr++;
        // animate the rotation
        myWindow.spinOnce(30, true);
    }

    fs.release();
    return 0;
}
