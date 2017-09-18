#実行方法
http://qiita.com/kazukitakata/private/d7a2a558a81e4f9309a0

#環境設定
##ubuntu
tested with ubuntu 16.04, CUDA 8.0,61

```

#複数カメラのキャリブレーション
$ cd calibration
##内部行列推定（各カメラについて行う）
$ ./bin/calibration {camera or video} {camera_num or cideo_file_path} {output_xml_file_path}
s : キャプチャスタート
e : キャプチャエンド
##外部行列推定（使用するカメラで同時に行う）
$ ./bin/estimate_external_matrix {camera_num} {camera1_port} {camera1_xml_file_path} {camera2_port} {camera2_xml_file_path]}
c : キャプチャスタート

#openpose3d
##再生
$ ./build/examples/openpose3d/openpose3D.bin {camera_num} {camera1_path} {camera1_xml_path} {camera2_num} {camera2_xml_path} ...

##座標のみ取得
$ ./build/examples/openpose3d/openpose3D_coordinate.bin {camera_num} {camera1_path} {camera1_xml_path} {camera2_num} {camera2_xml_path} ...

#人体が動く様子を録画
$ ./build/examples/user_code/surfpose3D_2.bin ./media/stand_pose1.MP4 ./media/stand_pose2.MP4 stand_camera1.xml stand_camera2.xml stand_camera_2d_pose1.xml stand_camera_2d_pose2.xml stand_camera_3d_pose.xml

#上で撮影した人体の骨格を再生
$ ./build/examples/user_code/viz3D.bin ./stand_camera_3d_pose.xml
```

#caution!
・openpose3d.cppにおいてdisablemultithreadingにしています。multithreadingにする方法を検討中。。。