# buildするとbinに以下の6つのプログラムができる。
//実際にkinectを使う
- calib_relative（以下1で使用）
- set_absolute（以下2で使用）
//上の２つから得たファイルを使う
- calc_merge_relative_absolute（以下3で使用）
- calc_merge_relatives（以下3.5で使用）
- calc_invert_relative（以下3.5で使用）
- calc_make_yaml（以下4で使用）


# 1. ２つのKinect間の対応付け
 bin/calib_relative serial1 serial2
を実行すると、２つのシリアル番号のKinectのIR画像が表示される。（undistort済み）
opencvのwaitkeyで"1"を入力すると、IR画像の中のコーナーを探し、結果が表示される。
２つのKinectでうまくコーナーが検出されていた場合、"2"を押す。
これによって、output_relative/relative_serial1_to_serial2.txt
が出力される。


# 2. あるKinectと絶対座標を対応付ける。
 bin/set_absolute serial
を実行すると、指定したシリアル番号のKinectのIR画像が表示される。（undistort済み）
calib_relativeと同様の手順で"1", "2"を押す。
これによって、
 output_absolute/absolute_serial_to_world.txt
 output_absolute/absolute_serial_to_world_org.txt
が出力される。
以下に示す3の手順で..._to_world.txtは増えていくが、
orgがついたファイルはこの手順でしか作成されない。
明示的に絶対座標と対応付けたkinectを忘れないために出力している。


# 3. 絶対座標と対応づいているkinectと相対的に対応づいているkinectを、絶対座標と対応付ける
例えば、以下の２つのファイルがあるとする。
 output_relative/relative_serial3_to_serial5.txt
 output_absolute/absolute_serial5_to_world.txt
この場合、以下の手順でserial3を絶対座標と対応付ける。
 bin/calc_merge_relative_absolute serial3 serial5
これによって、
 output_absolute/absolute_serial3_to_world.txt
が出力される。


# 3.5. 3のためにうまいことrelativeファイルを作成する便利ツール
* calc_merge_relatives
 output_relative/relative_serial1_to_serial2.txt
 output_relative/relative_serial2_to_serial3.txt
がある場合、
 bin/calc_merge_relatives serial1 serial2 serial3
を実行すると
 output_relative/relative_serial1_to_serial3.txt
が出力される。
* calc_invert_relative
 output_relative/relative_serial1_to_serial2.txt
がある場合、
 bin/calc_invert_relative serial1 serial2
を実行すると
 output_relative/relative_serial2_to_serial1.txt
が出力される。


# 4. 絶対座標と対応づいている複数のkinectから、kinect2_out.yamlを作成する。
 calc_make_yaml serial1 serial2 serial3 serial4
のようにすればいい。
