#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <usr/prediction.hpp>

std::vector<std::vector<std::vector<cv::Point2d>>> prediction::bilateral_prediction(std::vector<std::vector<std::vector<cv::Point2d>>> bodyPoints2D){
    
    std::vector<std::vector<std::vector<cv::Point2d>>> output_vector(bodyPoints2D.size());
    std::copy(bodyPoints2D.begin(), bodyPoints2D.end(), output_vector.begin());

    for(int camera_ptr=0; camera_ptr<bodyPoints2D[0].size(); camera_ptr++){

        std::vector<std::vector<std::vector<int>>> lack_frame_joint; //lack_frame_joint[joint_num][lack_num] = {lack_frame_start, lack_frame_length}
        
        for(int joint_ptr=0; joint_ptr<bodyPoints2D[0][0].size(); joint_ptr++){
            int frame_ptr=0;
            std::vector<std::vector<int>> lack_start_and_len; //lack_frame[lack_num] = {lack_frame_start, lack_frame_length}
            while(frame_ptr<bodyPoints2D.size()){
                if(bodyPoints2D[frame_ptr][camera_ptr][joint_ptr].x!=0){ //検出している
                    frame_ptr++;
                }else{
                    int lack_frame_start = frame_ptr;
                    int lack_frame_length = 0;
                    while(bodyPoints2D[lack_frame_start+lack_frame_length][camera_ptr][joint_ptr].x==0){
                        //if処理入れる。bodyPoints2Dのサイズを超えてアクセスする可能性あり
                        lack_frame_length++;
                        if(lack_frame_start+lack_frame_length==bodyPoints2D.size()){ //最終フレームに到達
                            break;
                        }
                    }
                    lack_start_and_len.push_back({lack_frame_start,lack_frame_length});
                    frame_ptr += lack_frame_length;
                }
            }
            lack_frame_joint.push_back(lack_start_and_len);
        }
        
        for(int joint_ptr=0; joint_ptr<lack_frame_joint.size(); joint_ptr++){
            for(int lack_ptr=0; lack_ptr<lack_frame_joint[joint_ptr].size(); lack_ptr++){
                int start = lack_frame_joint[joint_ptr][lack_ptr][0];
                int length = lack_frame_joint[joint_ptr][lack_ptr][1];
                if(start==0){ //0フレームから欠落していた場合
                    for(int i=0; i<length; i++){
                        output_vector[start+i][camera_ptr][joint_ptr] = output_vector[start+length][camera_ptr][joint_ptr];
                    }
                }else if(start+length==bodyPoints2D.size()){ //最終フレームまで欠落していた場合
                    for(int i=0; i<length; i++){
                        output_vector[start+i][camera_ptr][joint_ptr] = output_vector[start-1][camera_ptr][joint_ptr];
                    }
                }else{ //途中で欠落していた場合
                    for(int i=0; i<length; i++){
                        output_vector[start+i][camera_ptr][joint_ptr].x = output_vector[start-1][camera_ptr][joint_ptr].x * ((float)(length-i))/(length+1) + bodyPoints2D[start+length][camera_ptr][joint_ptr].x * ((float)(i+1))/(length+1);
                        output_vector[start+i][camera_ptr][joint_ptr].y = output_vector[start-1][camera_ptr][joint_ptr].y * ((float)(length-i))/(length+1) + bodyPoints2D[start+length][camera_ptr][joint_ptr].y * ((float)(i+1))/(length+1);
                    }
                }
            }
        }
    }
    return output_vector;
}