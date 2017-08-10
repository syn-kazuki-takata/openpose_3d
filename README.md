#実行方法
http://qiita.com/kazukitakata/private/d7a2a558a81e4f9309a0

#環境設定
##ubuntu
tested with ubuntu 16.04, CUDA 8.0,61

```
#カメラのシリアル番号取得
$ cd libfreenect2
$ cmake . && make all -j{num_of_CPU}
$ cd ../calibration
$ cmake . && make all -j${num_of_CPU}
```
