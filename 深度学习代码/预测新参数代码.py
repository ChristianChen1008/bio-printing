from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn


MODEL_DIR = Path(__file__).resolve().parent / "line_width_model_output"


# Change these values to predict a new experiment.
NEW_SAMPLE = {
    "arm_speed": 10.0,
    "extrusion_speed": 1.2,
    "nozzle_height": 0.8,
    "alginate_concentration": 2.0,
    "nozzle_diameter": 0.41,
}


class LineWidthNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.model(x)


def main():
    checkpoint_path = MODEL_DIR / "line_width_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Train the model first by running train_line_width_model.py")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_columns = checkpoint["feature_columns"]

    model = LineWidthNet(input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_scaler = joblib.load(MODEL_DIR / "x_scaler.joblib")
    y_scaler = joblib.load(MODEL_DIR / "y_scaler.joblib")

    x = np.array([[NEW_SAMPLE[column] for column in feature_columns]], dtype=np.float32)
    x_scaled = x_scaler.transform(x)

    with torch.no_grad():
        y_scaled = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy()
    y = y_scaler.inverse_transform(y_scaled)

    print("Input parameters:")
    for key, value in NEW_SAMPLE.items():
        print(f"  {key}: {value}")
    print(f"\nPredicted line width: {float(y[0, 0]):.6f}")


if __name__ == "__main__":
    main()
