# TrafficSignRecognition  

## 1 テーマ  
交通標識の意味を教えてくれるCNNモデル

## 2 画像認識システムの概要 
### a.ユーザー＆利用シーン 
ユーザー：日本の交通ルールを詳しく知らない外国人  
利用シーン：交通標識の意味をリアルタイムで教えてくれる

なぜ：
自分は今留学生寮に住んでいて、外国人と接する機会が多い。
そこで、外国人とが日本にいて不便なことを自分の力で少しでも解決できればと思っていた。そこで留学生の友達がレンタカーを借りれるけど、日本の交通ルールについて知らないと言われ、そのことについて調べてみるとコロナ前は外国人のレンタカーの利用者数が増加して、また外国人のドライバーによる事故が増えていた。そこで、交通標識の意味を運転中に教えてくれるものがあれば便利だと思った。


### b.課題解決イメージ  
目標としてはカメラの前に標識の画像をおくとその画像の意味を教えてくれる。


## 3 システムの詳細  
### a.システムの全体像  
1.Training用のデータセットを用いて機械学習モデルに学習をさせる  
2.画像認識のモデルをpickleオブジェクトとして保存する  
3.動画を静止画の集まりとして、そのそれぞれの静止画についてpickleの学習済のデータと比較してどの意味かを考える 


### b.実現したい機能  
実現したい機能のマイルストーンを以下のようにおく  
1.画像を入力したらどの画像か分類する  
2.動画からリアルタイムで標識を分類する
3.速度などの数字がついているものについてはその数字がどの数字かも判断できるようにして、どのような数字が出てきても対応できるようにする

### c.想定される技術的課題  
１.テストデータの画像の質がよくないため、より多くの画像数が必要になる。
２.映像を読み込む際にFPSが高すぎると認識するのにかかる時間と次のフレームを表示させる時間で誤差が生じる

## 4 開発イメージ
### a.入力画像、ラベル数 

今回入力画像として使用するのは International Joint Conference on Neural Networks 2011　に開かれた The German Traffic Sign Benchmarkのデータセットを利用した。理由としては42ラベル、写真は5万枚以上に上り、入力画像の枚数、ラベル数を幅広いレンジで変更できるため、難易度を段階的にあげていくことができ、最適なモデルを作るのには適していると考えた。

The German Traffic Sign Benchmark -> https://benchmark.ini.rub.de/gtsrb_news.html


### b.使用する技術   
Python3.9.5  
Keras version2.4.3  
Tensorflow version2.5.0  
OpenCV version4.5.36  


### c.ファイル階層構造  

IMAGE_RECONGNITION  
|── README.md  
|── codes&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;//ソースコード格納場所  
│&emsp;&emsp;&emsp;|── cnn_model.ipynb    //使用するAIモデル  
│&emsp;&emsp;&emsp;|── create_datasets.py //使用する画像を決める  
│&emsp;&emsp;&emsp;|── video.py&emsp;     //webcamに写っている被写体が何かを判別する  
|── image_datasets&emsp;   //写真データ格納場所  
│&emsp;&emsp;&emsp;|── Meta&emsp;&emsp;   //メタデータ  
│&emsp;&emsp;&emsp;|── Meta.csv&emsp;     //Meta含まれるデータの画像の情報が書かれている  
│&emsp;&emsp;&emsp;|── Test&emsp;&emsp;   //テストデータ  
│&emsp;&emsp;&emsp;|── Test.csv&emsp;     //Test含まれるデータの画像の情報が書かれている  
│&emsp;&emsp;&emsp;|── Train&emsp;&emsp;  //学習用データ  
│&emsp;&emsp;&emsp;|── Train.csv&emsp;    //Trainに含まれるデータのそれぞれの画像の情報が書かれている  
|── package.json  