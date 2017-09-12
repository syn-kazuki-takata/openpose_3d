#実行方法
http://qiita.com/kazukitakata/private/d7a2a558a81e4f9309a0

#環境設定
##ubuntu
tested with ubuntu 16.04, CUDA 8.0,61

```

#複数カメラのキャリブレーション
$ cd fish_eye_calib
##内部行列推定（各カメラについて行う）
$ ./bin/calibration {camera or video} {camera_num or cideo_file_path} {output_xml_file_path}
s : キャプチャスタート
e : キャプチャエンド
##外部行列推定（使用するカメラで同時に行う）
$ ./bin/estimate_external_matrix {camera_num} {camera1_port} {camera1_xml_file_path} {camera2_port} {camera2_xml_file_path]}
c : キャプチャスタート

##（以下でも可能）
##全てのカメラのキャプチャ範囲内にチェッカーボードが入っている状態で動画を撮影する
$ ./bin movie capture {camera1_port_num} {camera2_port_num} {movie1_file_path} {movie2_file_path}
##撮影した映像からキャリブレーションに使う画像を生成
$ ./bin/video_convert_image {movie1_file_path} {image_directory1_path}
$ ./bin/video_convert_image {movie2_file_path} {image_directory2_path}
##内部行列推定
$ ./bin/fisheye_calib {image_directory1_path} {xml1_path}
$ ./bin/fisheye_calib {image_directory2_path} {xml2_path}
##外部行列推定（同じフレームの画像を用いる）
$  ./bin/estimate_external_matrix {image_directory1_path}/img1.jpg {xml1_path}
$  ./bin/estimate_external_matrix {image_directory2_path}/img1.jpg {xml2_path}

#人体が動く様子を録画
$ ./build/examples/user_code/surfpose3D_2.bin ./media/stand_pose1.MP4 ./media/stand_pose2.MP4 stand_camera1.xml stand_camera2.xml stand_camera_2d_pose1.xml stand_camera_2d_pose2.xml stand_camera_3d_pose.xml

#上で撮影した人体の骨格を再生
$ ./build/examples/user_code/viz3D.bin ./stand_camera_3d_pose.xml
```