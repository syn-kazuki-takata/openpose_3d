#実行方法
http://qiita.com/kazukitakata/private/d7a2a558a81e4f9309a0

#環境設定
##ubuntu
tested with ubuntu 16.04, CUDA 8.0,61

```
#複数カメラのキャリブレーション
$ cd fish_eye_calib
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