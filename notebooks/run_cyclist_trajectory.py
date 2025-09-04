import os
import cv2
import time
import math
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

YOLO_WEIGHTS    = "models/best1.pt"
TRAJ_MODEL_PTH  = "models/normal_lstm_model.pth"
INPUT_VIDEO     = "test images/video_input"
OUTPUT_VIDEO    = "models/cyclist_tracking_output.mp4"
LOG_TXT_PATH    = "models/LOG_TXT_PATH.txt"
WRITE_LOG_FILE  = True

K = np.array([[1000.0, 0.0, 640.0],
              [0.0, 1000.0, 360.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)
DIST_COEFFS = np.zeros((1, 5), dtype=np.float64)
APPLY_UNDISTORT = False
CAMERA_HEIGHT_M = 1.5
GROUND_NORMAL   = np.array([0.0, 1.0, 0.0], dtype=np.float64)
VO_SCALE_M_PER_UNIT = 1.0

OBS_LEN  = 8
PRED_LEN = 12
CYCLIST_CLASS_INDEX = 0
safety_distance = 1.0  # meters, collision threshold
MIN_HEIGHT = 50
MIN_WIDTH = 20
MIN_MOVE_THRESHOLD = 5.0  # min movement to change direction
CONFIDENCE_THRESHOLD = 0.5

# DeepSort tracker init
tracker = DeepSort(max_age=30, n_init=3)

class NormalLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim)
    def forward(self, obs):
        _, (h, c) = self.lstm(obs)
        decoder_input = obs[:, -1, :]
        outputs = []
        hidden = (h, c)
        for _ in range(PRED_LEN):
            out, hidden = self.lstm(decoder_input.unsqueeze(1), hidden)
            pred_pos = self.fc(out.squeeze(1))
            outputs.append(pred_pos)
            decoder_input = pred_pos
        return torch.stack(outputs, dim=1)

def preprocess_for_lstm(seq_xy):
    arr  = np.asarray(seq_xy, dtype=np.float32)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0) + 1e-6
    norm = (arr - mean) / std
    return norm, mean, std

def denormalize_lstm_output(preds, mean, std):
    return preds * std + mean

# ... Include class ORBVO, homography_from_pose, warp_points, discretize_direction here as needed ...

def main():
    detector = YOLO(YOLO_WEIGHTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traj_model = NormalLSTM().to(device)
    traj_model.load_state_dict(torch.load(TRAJ_MODEL_PTH, map_location=device))
    traj_model.eval()

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_VIDEO}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    if APPLY_UNDISTORT:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, DIST_COEFFS, (width, height), 1, (width, height))
        map1, map2 = cv2.initUndistortRectifyMap(K, DIST_COEFFS, None, newK, (width, height), cv2.CV_16SC2)
        K_use = newK
    else:
        map1 = map2 = None
        K_use = K

    # Create VO and buffers structures here (copy your ORBVO and buffer initializations)...

    # Your detection, tracking, VO, prediction, drawing, and logging loop goes here (based on your code)...
    # For brevity, refer to your already provided main loop code for the full implementation.

    # Close video, logging, and print summary as in your existing script

if __name__ == "__main__":
    main()
