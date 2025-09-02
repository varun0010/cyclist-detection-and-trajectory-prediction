import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader, Dataset

OBS_LEN = 8
PRED_LEN = 12

# Dataset class as in notebook
class TrajectoryDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        s = self.seqs[idx]
        return torch.FloatTensor(s[:OBS_LEN]), torch.FloatTensor(s[OBS_LEN:])

# Model class as in notebook
class NormalLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

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

def load_test_data(path):
    # Load and preprocess test data from .npy or custom format
    # Adapt according to how test data is stored
    seqs = np.load(path)
    return TrajectoryDataset(seqs)

def plot_predictions(model, test_loader, device, num_examples=3):
    model.eval()
    with torch.no_grad():
        for i, (obs, fut) in enumerate(test_loader):
            if i >= num_examples:
                break
            obs, fut = obs.to(device), fut

            pred = model(obs).cpu().numpy()[0]
            obs = obs.cpu().numpy()[0]
            fut = fut.numpy()[0]

            plt.figure(figsize=(7,7))
            plt.grid(True)
            plt.plot(obs[:,0], obs[:,1], 'bo-', label="Observed Past")
            plt.plot(fut[:,0], fut[:,1], 'gx--', label="Ground Truth Future")
            plt.plot(pred[:,0], pred[:,1], 'r^-', label="Predicted Future")
            plt.axis('equal')
            plt.legend()
            plt.title(f"Cyclist Trajectory Prediction (Normal LSTM) - Example {i+1}")
            plt.xlabel("Normalized X")
            plt.ylabel("Normalized Y")
            plt.show()

def main(model_path, test_data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test dataset
    test_dataset = load_test_data(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Load model
    model = NormalLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Plot predictions on test data
    plot_predictions(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trajectory prediction inference")
    parser.add_argument('--model', type=str, required=True, help="models/normal_lstm_model.pth")
    parser.add_argument('--test_data', type=str, required=True, help="test images/test.npy")
    args = parser.parse_args()

    main(args.model, args.test_data)
