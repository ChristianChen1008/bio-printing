from pathlib import Path
import json
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Change this to your real experimental data CSV.
DATA_PATH = Path(__file__).resolve().parent / "line_width_training_template.csv"


FEATURE_COLUMNS = [
    "arm_speed",
    "extrusion_speed",
    "nozzle_height",
    "alginate_concentration",
    "nozzle_diameter",
]
TARGET_COLUMN = "line_width"


OUTPUT_DIR = Path(__file__).resolve().parent / "line_width_model_output"
RANDOM_SEED = 42


# For small first-round datasets, keep the model small.
BATCH_SIZE = 8
EPOCHS = 500
LEARNING_RATE = 0.001
TEST_SIZE = 0.25


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(csv_path):
    data = pd.read_csv(csv_path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    data = data[required_columns].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    if len(data) < 6:
        raise ValueError(
            "Too few valid rows. Add more experiments before training. "
            "A first try should ideally have at least 20-30 rows."
        )
    return data


def split_and_scale(data):
    x = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = data[[TARGET_COLUMN]].to_numpy(dtype=np.float32)

    if len(data) < 12:
        # With very few rows, a random test split can be unstable. This still
        # gives a quick check, but do not over-trust the score.
        test_size = max(1, int(round(len(data) * TEST_SIZE)))
    else:
        test_size = TEST_SIZE

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, y_test, x_scaler, y_scaler


def make_loader(x_train, y_train):
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True)


def train_model(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        history.append(mean_loss)

        if epoch == 1 or epoch % 50 == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS}, train_loss={mean_loss:.6f}")

    return history


def evaluate_model(model, x_test, y_test_scaled, y_test_original, y_scaler):
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(torch.tensor(x_test, dtype=torch.float32)).numpy()

    prediction = y_scaler.inverse_transform(prediction_scaled)
    y_true = y_test_original

    mae = mean_absolute_error(y_true, prediction)
    rmse = mean_squared_error(y_true, prediction) ** 0.5
    r2 = r2_score(y_true, prediction) if len(y_true) >= 2 else float("nan")
    return prediction, {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def save_outputs(model, x_scaler, y_scaler, history, predictions, y_true, metrics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "input_size": len(FEATURE_COLUMNS),
        },
        OUTPUT_DIR / "line_width_model.pt",
    )
    joblib.dump(x_scaler, OUTPUT_DIR / "x_scaler.joblib")
    joblib.dump(y_scaler, OUTPUT_DIR / "y_scaler.joblib")

    result = pd.DataFrame(
        {
            "true_line_width": y_true.reshape(-1),
            "predicted_line_width": predictions.reshape(-1),
            "error": predictions.reshape(-1) - y_true.reshape(-1),
        }
    )
    result.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8")

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Line Width Model Training Curve")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, predictions)
    min_value = float(min(y_true.min(), predictions.min()))
    max_value = float(max(y_true.max(), predictions.max()))
    plt.plot([min_value, max_value], [min_value, max_value], "r--")
    plt.xlabel("True line width")
    plt.ylabel("Predicted line width")
    plt.title("Prediction Check")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_check.png", dpi=200)
    plt.close()


def main():
    set_seed(RANDOM_SEED)
    data = load_data(DATA_PATH)

    print(f"Loaded rows: {len(data)}")
    print("Feature columns:", ", ".join(FEATURE_COLUMNS))
    print("Target column:", TARGET_COLUMN)

    x_train, x_test, y_train, y_test_scaled, y_test_original, x_scaler, y_scaler = split_and_scale(data)
    train_loader = make_loader(x_train, y_train)

    model = LineWidthNet(input_size=len(FEATURE_COLUMNS))
    history = train_model(model, train_loader)

    predictions, metrics = evaluate_model(
        model,
        x_test,
        y_test_scaled,
        y_test_original,
        y_scaler,
    )
    save_outputs(model, x_scaler, y_scaler, history, predictions, y_test_original, metrics)

    print("\nEvaluation on test data:")
    print(f"MAE  = {metrics['mae']:.6f}")
    print(f"RMSE = {metrics['rmse']:.6f}")
    print(f"R2   = {metrics['r2']:.6f}")
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
