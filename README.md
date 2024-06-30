# Vision-Based Human Detection

このプロジェクトは、HOG (Histogram of Oriented Gradients) と SVM (Support Vector Machine) を使用して、人間の検出を行うプログラムである。データセットとして INRIA Person Dataset を使用し、データの前処理、モデルのトレーニング、リアルタイム検出を行う。また、OpenCV標準のHOG人検出器を使用した検出も行う。

## プロジェクトの構成

- `download_and_prepare.py` : データセットをダウンロードし、画像を保存するスクリプト
- `preprocess.py` : 画像データの前処理を行い、HOG特徴量を計算して保存するスクリプト
- `train.py` : SVMモデルをトレーニングし、最適なモデルを保存するスクリプト
- `detect.py` : 学習済みモデルを使用して動画ファイル内の人間を検出するスクリプト
- `detect_opencv_hog.py` : OpenCVの標準HOG人検出器を使用して動画ファイル内の人間を検出するスクリプト
- `detect_images.py` : 学習済みモデルを使用してディレクトリ内の画像ファイルから人間を検出するスクリプト
- `detect_images_opencv_hog.py` : OpenCVの標準HOG人検出器を使用してディレクトリ内の画像ファイルから人間を検出するスクリプト
- `requirements.txt` : 必要なPythonライブラリを記述したファイル

## 必要なライブラリ

以下のライブラリが必要である。`requirements.txt`に記述されている。

- opencv-python
- numpy
- scikit-learn
- joblib
- tqdm
- datasets
- matplotlib

## インストール

以下の手順で必要なライブラリをインストールする。

1. Pythonをインストールする (バージョン3.6以上を推奨)。
2. 仮想環境を作成し、アクティベートする (任意)。

    ```sh
    python -m venv venv
    source venv/bin/activate  # Windowsの場合は venv\Scripts\activate
    ```

3. `requirements.txt`から必要なライブラリをインストールする。

    ```sh
    pip install -r requirements.txt
    ```

## 実行手順

### データセットのダウンロードと準備

1. データセットをダウンロードして準備する。

    ```sh
    python download_and_prepare.py
    ```

### 画像データの前処理

2. 画像データの前処理を行い、HOG特徴量を計算して保存する。

    ```sh
    python preprocess.py
    ```

### SVMモデルのトレーニング

3. SVMモデルをトレーニングし、最適なモデルを保存する。

    ```sh
    python train.py
    ```

### 動画ファイル内の人間の検出 (SVMモデル使用)

4. 学習済みモデルを使用して動画ファイル内の人間を検出する。

    ```sh
    python detect.py <input_video.mp4>
    ```

### 動画ファイル内の人間の検出 (OpenCV HOG使用)

5. OpenCVの標準HOG人検出器を使用して動画ファイル内の人間を検出する。

    ```sh
    python detect_opencv_hog.py <input_video.mp4>
    ```

### ディレクトリ内の画像ファイルから人間の検出 (SVMモデル使用)

6. 学習済みモデルを使用してディレクトリ内の画像ファイルから人間を検出する。

    ```sh
    python detect_images.py <input_directory> <output_directory>
    ```

### ディレクトリ内の画像ファイルから人間の検出 (OpenCV HOG使用)

7. OpenCVの標準HOG人検出器を使用してディレクトリ内の画像ファイルから人間を検出する。

    ```sh
    python detect_images_opencv_hog.py <input_directory> <output_directory>
    ```

## 学習結果

`train.py`を実行すると、以下のような結果が表示される。

learning_curve_rbf.png


```
{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
Model saved as svm_model.pkl
Classification Report:
              precision    recall  f1-score   support

         0.0       0.99      1.00      1.00       104
         1.0       1.00      0.99      0.99        96

    accuracy                           0.99       200
   macro avg       1.00      0.99      0.99       200
weighted avg       1.00      0.99      0.99       200

Accuracy: 0.9950

```