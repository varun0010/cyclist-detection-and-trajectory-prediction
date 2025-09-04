# Cyclist-detection-and-trajectory-prediction
Detect one or more cyclist in video frame and predict their future trajectory in order to avoid collision
<br>There are mainly three parts of this project:
1. Cyclist detection using **YOLO**
2. Giving unique ids for different cyclist in frame using **Deepsort**
3. Predicting future trajectory using a **LSTM model**
## Cyclist detection
We trained yolo v8 version to detect cyclists. With help of opencv we were able to draw bounding boxes around the detected cyclist.
<br>**Dataset:**
<br>We utilised already existing dataset from roboflow, you can access the dataset  [using this link]( https://universe.roboflow.com/cycler-vi9vn/cyclists-lt9pl/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true) It has 1542 images.
# Installation
-`pip install roboflow`<br>
-`pip install ultralytics` <br>
-`pip install pillow`<br>
# To run cyclist detection:
-`python run_inference.py --weights models/best1.pt --source test_images/`
# Trajectary prediction using LSTM model
<br>We trained a Long term short memory machine learning model to take 5 previous frame's coordinates as input.<br>
we trained it on [this](https://www.kaggle.com/datasets/zcyan2/onsitevru-trajectory-prediction-dataset?select=train_data_y.npy) dataset<br>
# Installation
-`pip install torch  `<br>
-`pip install numpy` <br>
-`pip install matplotlib `<br>
-`pip install scikit-learn` <br>
# To run trajectary prediction
-`python run_trajectory_prediction.py --model models/normal_lstm_model.pth --test_data images/test.npy`<br>
# To run Combined 
-`python combined_pipeline.py --yolo_weights models/best1.pt --lstm_weights models/normal_lstm_model.pth --video "test image/video.mp4"`






