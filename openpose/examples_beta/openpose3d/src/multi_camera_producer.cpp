#include <chrono>
#include <iostream>
#include <thread>
// OpenCV
#include <opencv2/core.hpp>
//#include <opencv2/sfm.hpp>
#include <opencv2/opencv.hpp>
//#include <openpose3d/cameraParameters.hpp>
//#include <openpose3d/pointGrey.hpp>
#include <openpose3d/multi_camera_producer.hpp>

// This function acquires and displays images from each device.
std::vector<cv::Mat> acquireImages(std::vector<cv::VideoCapture> &cameras, std::vector<cv::Mat> &intrinsics, std::vector<cv::Mat> &distortions)
{
    try
    {
		std::vector<cv::Mat> cvMats;
        //std::cout<<"cameras_size : "<<cameras.size()<<std::endl;
        /*
        for(int i=0;i<200;i++){
            cv::Mat frame0, frame1;
            cameras[0] >> frame0;
            cameras[1] >> frame1;
            cv::imshow("camera0",frame0);
            cv::imshow("camera1",frame1);
            cv::waitKey(1);
        }
        */
        
        for(int i=0u; i<cameras.size(); i++){
            cv::Mat frame;
            cameras[i] >> frame;
            cvMats.emplace_back();
            cv::Mat new_intrinsic, mapx, mapy;
            cv::Size video_size = cv::Size(cameras[i].get(CV_CAP_PROP_FRAME_WIDTH),cameras[i].get(CV_CAP_PROP_FRAME_HEIGHT));
            cv::initUndistortRectifyMap(intrinsics[i], distortions[i], cv::Mat(), new_intrinsic, video_size, CV_32FC1, mapx, mapy);
            cv::remap(frame, cvMats[i], mapx, mapy, cv::INTER_LINEAR);
            //std::string window_name = "camera" + std::to_string(i);
            //cv::imshow(window_name, frame);
            //cv::waitKey(1);
            //cv::undistort(frame, cvMats[i], intrinsics[i], distortions[i]);
        }
        /*
        // Retrieve, convert, and return an image for each camera
        // In order to work with simultaneous camera streams, nested loops are
        // needed. It is important that the inner loop be the one iterating
        // through the cameras; otherwise, all images will be grabbed from a
        // single camera before grabbing any images from another.
        
        // Get cameras
        std::vector<Spinnaker::ImagePtr> imagePtrs(cameraPtrs.size());

        // Getting frames
        // Retrieve next received image and ensure image completion
        // Spinnaker::ImagePtr imagePtr = cameraPtrs.at(i)->GetNextImage();
        // Clean buffer + retrieve next received image + ensure image completion
        auto durationMs = 0.;
        // for (auto counter = 0 ; counter < 10 ; counter++)
        while (durationMs < 1.)
        {
            const auto begin = std::chrono::high_resolution_clock::now();
            for (auto i = 0u; i < cameraPtrs.size(); i++)
                imagePtrs.at(i) = cameraPtrs.at(i)->GetNextImage();
            durationMs = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-begin).count() * 1e-6;
			// op::log("Time extraction (ms): " + std::to_string(durationMs), op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
        }

        // Original format -> RGB8
        bool imagesExtracted = true;
        for (auto& imagePtr : imagePtrs)
        {
            if (imagePtr->IsIncomplete())
            {
                op::log("Image incomplete with image status " + std::to_string(imagePtr->GetImageStatus()) + "...",
					    op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
                imagesExtracted = false;
                break;
            }
            else
            {
                // Print image information
                // Convert image to RGB
                // Interpolation methods
                // http://softwareservices.ptgrey.com/Spinnaker/latest/group___spinnaker_defs.html
                // DEFAULT     Default method.
                // NO_COLOR_PROCESSING     No color processing.
                // NEAREST_NEIGHBOR    Fastest but lowest quality. Equivalent to FLYCAPTURE_NEAREST_NEIGHBOR_FAST in FlyCapture.
                // EDGE_SENSING    Weights surrounding pixels based on localized edge orientation.
                // HQ_LINEAR   Well-balanced speed and quality.
                // RIGOROUS    Slowest but produces good results.
                // IPP     Multi-threaded with similar results to edge sensing.
                // DIRECTIONAL_FILTER  Best quality but much faster than rigorous.
                // Colors
                // http://softwareservices.ptgrey.com/Spinnaker/latest/group___camera_defs__h.html#ggabd5af55aaa20bcb0644c46241c2cbad1a33a1c8a1f6dbcb4a4eaaaf6d4d7ff1d1
                // PixelFormat_BGR8
                // Time tests
                // const auto reps = 1e3;
                // // const auto reps = 1e2; // for RIGOROUS & DIRECTIONAL_FILTER
                // const auto begin = std::chrono::high_resolution_clock::now();
                // for (auto asdf = 0 ; asdf < reps ; asdf++){
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::DEFAULT); // ~ 1.5 ms but pixeled
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::NO_COLOR_PROCESSING); // ~0.5 ms but BW
                imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::HQ_LINEAR); // ~6 ms, looks as good as best
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::EDGE_SENSING); // ~2 ms default << edge << best
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::RIGOROUS); // ~115, too slow
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::IPP); // ~2 ms, slightly worse than HQ_LINEAR
                // imagePtr = imagePtr->Convert(Spinnaker::PixelFormat_BGR8, Spinnaker::DIRECTIONAL_FILTER); // ~30 ms, ideally best quality?
                // imagePtr = imagePtr;
                // }
                // durationMs = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-begin).count() * 1e-6;
				// op::log("Time conversion (ms): " + std::to_string(durationMs / reps), op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
            }
        }

        // Convert to cv::Mat
        if (imagesExtracted)
        {
            for (auto i = 0u; i < imagePtrs.size(); i++)
            {
                // Baseline
                // cvMats.emplace_back(pointGreyToCvMat(imagePtrs.at(i)).clone());
                // Undistort
                // http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
                auto auxCvMat = pointGreyToCvMat(imagePtrs.at(i));
                cvMats.emplace_back();
                cv::undistort(auxCvMat, cvMats[i], INTRINSICS[i], DISTORTIONS[i]);
            }
        }
        */

		return cvMats;
    }
    /*
    catch (Spinnaker::Exception &e)
    {
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		return {};
	}
    */
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		return {};
	}
}

//WMultiCamera::WMultiCamera(std::vector<cv::VideoCapture> &_cameras, std::vector<cv::FileStorage> &_camerafs) :
WMultiCamera::WMultiCamera(std::vector<std::string> &_camera_path, std::vector<cv::FileStorage> &_camerafs) :
    initialized{false},
    //cameras{_cameras},
    camera_path{_camera_path},
    camerafs{_camerafs}
{
    /*
    try
    {

    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
    */
}

WMultiCamera::~WMultiCamera()
{
    try
    {
		op::log("Done! Exitting...", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void WMultiCamera::initializationOnThread()
{
    try
    {
        initialized = true;
        for(int i=0; i<camera_path.size(); i++){
            cameras.emplace_back();
            cameras[i] = cv::VideoCapture(camera_path[i]);
        }
        for(int i=0; i<camerafs.size(); i++){
            intrinsics.emplace_back();
            camerafs[i]["intrinsic"] >> intrinsics[i];
            distortions.emplace_back();
            camerafs[i]["distortion"] >> distortions[i];
            cv::Mat rvec, rotation, translation, external;
            camerafs[i]["rvec_undistorted"] >> rvec;
            cv::Rodrigues(rvec, rotation);
            camerafs[i]["tvec_undistorted"] >> translation;
            cv::hconcat(rotation, translation, external);
            camera_matrixs.emplace_back();
            camera_matrixs[i] = intrinsics[i] * external;
            std::cout<<"intrinsic["<<i<<"] : "<<intrinsics[i]<<std::endl;
            std::cout<<"distortion["<<i<<"] : "<<distortions[i]<<std::endl;
            std::cout<<"camera_matrix["<<i<<"]"<<camera_matrixs[i]<<std::endl;
        }
        /*
        // Print application build information
		op::log(std::string{ "Application build date: " } + __DATE__ + " " + __TIME__, op::Priority::High, __LINE__, __FUNCTION__, __FILE__);

        // Retrieve singleton reference to mSystemPtr object
        mSystemPtr = Spinnaker::System::GetInstance();

        // Retrieve list of cameras from the mSystemPtr
        mCameraList = mSystemPtr->GetCameras();

        unsigned int numCameras = mCameraList.GetSize();

		op::log("Number of cameras detected: " + std::to_string(numCameras), op::Priority::High, __LINE__, __FUNCTION__, __FILE__);

        // Finish if there are no cameras
        if (numCameras == 0)
        {
            // Clear camera list before releasing mSystemPtr
            mCameraList.Clear();

            // Release mSystemPtr
            mSystemPtr->ReleaseInstance();

			op::log("Not enough cameras!\nPress Enter to exit...", op::Priority::High);
            getchar();

            op::error("No cameras detected.", __LINE__, __FUNCTION__, __FILE__);
        }
		op::log("Camera system initialized...", op::Priority::High);

        //
        // Retrieve transport layer nodemaps and print device information for
        // each camera
        //
        // *** NOTES ***
        // This example retrieves information from the transport layer nodemap
        // twice: once to print device information and once to grab the device
        // serial number. Rather than caching the nodemap, each nodemap is
        // retrieved both times as needed.
        //
		op::log("\n*** DEVICE INFORMATION ***\n", op::Priority::High);

        for (int i = 0; i < mCameraList.GetSize(); i++)
        {
            // Select camera
            auto cameraPtr = mCameraList.GetByIndex(i);

            // Retrieve TL device nodemap
            auto& iNodeMapTLDevice = cameraPtr->GetTLDeviceNodeMap();

            // Print device information
            auto result = printDeviceInfo(iNodeMapTLDevice, i);
            if (result < 0)
                op::error("Result > 0, error " + std::to_string(result) + " occurred...", __LINE__, __FUNCTION__, __FILE__);
        }

        for (auto i = 0; i < mCameraList.GetSize(); i++)
        {
            // Select camera
            auto cameraPtr = mCameraList.GetByIndex(i);

            // Initialize each camera
            // You may notice that the steps in this function have more loops with
            // less steps per loop; this contrasts the acquireImages() function
            // which has less loops but more steps per loop. This is done for
            // demonstrative purposes as both work equally well.
            // Later: Each camera needs to be deinitialized once all images have been
            // acquired.
            cameraPtr->Init();

            // Retrieve GenICam nodemap
            // auto& iNodeMap = cameraPtr->GetNodeMap();

            // // Configure trigger
            // result = configureTrigger(iNodeMap);
            // if (result < 0)
			// op::error("Result > 0, error " + std::to_string(result) + " occurred...", __LINE__, __FUNCTION__, __FILE__);

            // // Configure chunk data
            // result = configureChunkData(iNodeMap);
            // if (result < 0)
            //     return result;

            // Remove buffer --> Always get newest frame
            Spinnaker::GenApi::INodeMap& snodeMap = cameraPtr->GetTLStreamNodeMap();
            Spinnaker::GenApi::CEnumerationPtr ptrBufferHandlingMode = snodeMap.GetNode("StreamBufferHandlingMode");
            if (!Spinnaker::GenApi::IsAvailable(ptrBufferHandlingMode) || !Spinnaker::GenApi::IsWritable(ptrBufferHandlingMode))
				op::error("Unable to change buffer handling mode", __LINE__, __FUNCTION__, __FILE__);

            Spinnaker::GenApi::CEnumEntryPtr ptrBufferHandlingModeNewest = ptrBufferHandlingMode->GetEntryByName("NewestFirstOverwrite");
            if (!Spinnaker::GenApi::IsAvailable(ptrBufferHandlingModeNewest) || !IsReadable(ptrBufferHandlingModeNewest))
				op::error("Unable to set buffer handling mode to newest (entry 'NewestFirstOverwrite' retrieval). Aborting...", __LINE__, __FUNCTION__, __FILE__);
            int64_t bufferHandlingModeNewest = ptrBufferHandlingModeNewest->GetValue();

            ptrBufferHandlingMode->SetIntValue(bufferHandlingModeNewest);
        }

        // Prepare each camera to acquire images
        //
        // *** NOTES ***
        // For pseudo-simultaneous streaming, each camera is prepared as if it
        // were just one, but in a loop. Notice that cameras are selected with
        // an index. We demonstrate pseduo-simultaneous streaming because true
        // simultaneous streaming would require multiple process or threads,
        // which is too complex for an example.
        //
        // Serial numbers are the only persistent objects we gather in this
        // example, which is why a std::vector is created.
        std::vector<Spinnaker::GenICam::gcstring> strSerialNumbers(mCameraList.GetSize());
        for (auto i = 0u; i < strSerialNumbers.size(); i++)
        {
            // Select camera
            auto cameraPtr = mCameraList.GetByIndex(i);

            // Set acquisition mode to continuous
            Spinnaker::GenApi::CEnumerationPtr ptrAcquisitionMode = cameraPtr->GetNodeMap().GetNode("AcquisitionMode");
            if (!Spinnaker::GenApi::IsAvailable(ptrAcquisitionMode) || !Spinnaker::GenApi::IsWritable(ptrAcquisitionMode))
				op::error("Unable to set acquisition mode to continuous (node retrieval; camera " + std::to_string(i) + "). Aborting...", __LINE__, __FUNCTION__, __FILE__);

            Spinnaker::GenApi::CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
            if (!Spinnaker::GenApi::IsAvailable(ptrAcquisitionModeContinuous) || !Spinnaker::GenApi::IsReadable(ptrAcquisitionModeContinuous))
				op::error("Unable to set acquisition mode to continuous (entry 'continuous' retrieval " + std::to_string(i) + "). Aborting...", __LINE__, __FUNCTION__, __FILE__);

            int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

            ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

			op::log("Camera " + std::to_string(i) + " acquisition mode set to continuous...", op::Priority::High);

            // Begin acquiring images
            cameraPtr->BeginAcquisition();

			op::log("Camera " + std::to_string(i) + " started acquiring images...", op::Priority::High);

            // Retrieve device serial number for filename
            strSerialNumbers[i] = "";

            Spinnaker::GenApi::CStringPtr ptrStringSerial = cameraPtr->GetTLDeviceNodeMap().GetNode("DeviceSerialNumber");

            if (Spinnaker::GenApi::IsAvailable(ptrStringSerial) && Spinnaker::GenApi::IsReadable(ptrStringSerial))
            {
                strSerialNumbers[i] = ptrStringSerial->GetValue();
                op::log("Camera " + std::to_string(i) + " serial number set to " + strSerialNumbers[i].c_str() + "...", op::Priority::High);
            }
			op::log(" ", op::Priority::High);
        }
        */

		op::log("\nRunning for all cameras...\n\n*** IMAGE ACQUISITION ***\n", op::Priority::High);
    }
    /*
    catch (const Spinnaker::Exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
    */
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

std::shared_ptr<std::vector<Datum3D>> WMultiCamera::workProducer()
{
    try
    {
        // Profiling speed
        const auto profilerKey = op::Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
        // Get image from each camera
        //const auto cvMats = acquireImages(mCameraList);
        /*
        for(int i=0;i<200;i++){
            op::log("m2");
            cv::Mat frame0, frame1;
            op::log("m3");
            cameras[0] >> frame0;
            cameras[1] >> frame1;
            op::log("m4");
            if(!frame0.empty()){
                op::log("m5");
                cv::imshow("camera0",frame0);
                cv::waitKey(1);
            }
            if(!frame1.empty()){
                op::log("m6");
                cv::imshow("camera1",frame1);
                cv::waitKey(1);
            }
            op::log("m5");
            cv::waitKey(1);
        }
        */
        std::vector<cv::Mat> cvMats = acquireImages(cameras, intrinsics, distortions);
        // Images to userDatum
        auto datums3d = std::make_shared<std::vector<Datum3D>>(cvMats.size());
        for (auto i = 0u ; i < cvMats.size() ; i++){
            datums3d->at(i).cvInputData = cvMats.at(i);
        }
        // Profiling speed
        if (!cvMats.empty())
        {
            op::Profiler::timerEnd(profilerKey);
            op::Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, 100);
        }
        // Return Datum
        return datums3d;
    }
    catch (const std::exception& e)
    {
        this->stop();
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return nullptr;
    }
}
