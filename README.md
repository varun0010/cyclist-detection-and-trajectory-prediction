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
-`pip install ipython`<br> 
# To execute the entire notebook from the command line and save the output:
-`jupyter nbconvert --to notebook --execute notebooks/cyclist_Detection.ipynb --output executed_notebook.ipynb`


