
"""
industry_lstm_forecast.py

Production-style LSTM multi-step forecasting (PyTorch)
Features:
 - Modular functions: prepare_data, TrainDataset, build_model, train, evaluate, predict_rollout
 - Temporal train/val/test split (no leakage)
 - Early stopping on validation loss
 - Proper scaling & inverse-scaling (handles multi-step)
 - Save / load model checkpoints
 - Rolling forecast function for deployment (simulate production)
"""

import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --------------------------
# Dataset & Utilities
# --------------------------
class SalesDataset(Dataset):
    """
    Dataset that returns (seq_len window, forecast_horizon target)
    Expects `data` as numpy array shape (num_days, 1)
    """
    def __init__(self, data: np.ndarray, seq_length: int, forecast_horizon: int):
        self.data = data.astype(np.float32)
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.n_samples = len(self.data) - self.seq_length - self.forecast_horizon + 1
        if self.n_samples < 1:
            raise ValueError("Not enough data for the chosen seq_length and forecast_horizon.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_length]          # shape (seq_length, 1)
        y = self.data[idx + self.seq_length: idx + self.seq_length + self.forecast_horizon]  # (horizon, 1)
        return torch.from_numpy(x), torch.from_numpy(y).squeeze(-1)  # x: (seq_len,1), y: (horizon,)

def temporal_train_val_test_split(data_len: int, train_ratio=0.7, val_ratio=0.1):
    """
    Return indices split for chronological dataset
    """
    train_end = int(data_len * train_ratio)
    val_end = train_end + int(data_len * val_ratio)
    return (0, train_end), (train_end, val_end), (val_end, data_len)

# --------------------------
# Model
# --------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2, dropout=0.1, forecast_horizon: int = 7):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)                 # out: (batch, seq_len, hidden)
        last = out[:, -1, :]                  # take last time-step -> (batch, hidden)
        out = self.fc(last)                   # (batch, forecast_horizon)
        return out

# --------------------------
# Training / Evaluation
# --------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 7,
    save_path: str = "best_lstm.pth"
) -> Dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)

    best_val_loss = float("inf")
    best_epoch = -1
    history = {"train_loss": [], "val_loss": []}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- Train loop ---
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)          # xb: (batch, seq_len, 1)
            yb = yb.to(device)          # yb: (batch, horizon)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)

        # --- Val loop ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
                n_val += 1
        val_loss = val_loss / max(1, n_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

        # Early stopping & checkpoint
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss}, save_path)
            print(f"  Saved best model (epoch {epoch}) to {save_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}, val_loss: {best_val_loss:.6f}")
                break

    # load best model
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return {"model": model, "history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss}

def evaluate_model(model: nn.Module, data_loader: DataLoader, scaler: MinMaxScaler, device: torch.device) -> Dict:
    model.eval()
    preds_list, actuals_list = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            out = model(xb)           # (batch, horizon)
            preds_list.append(out.cpu().numpy())
            actuals_list.append(yb.cpu().numpy())
    preds = np.vstack(preds_list)    # shape (N_samples, horizon)
    actuals = np.vstack(actuals_list)

    # inverse scale (flatten, inverse, reshape)
    preds_flat = preds.reshape(-1, 1)
    actuals_flat = actuals.reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds_flat).reshape(preds.shape)
    actuals_inv = scaler.inverse_transform(actuals_flat).reshape(actuals.shape)

    # metrics aggregated across all horizon elements
    mae = mean_absolute_error(actuals_inv.flatten(), preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(actuals_inv.flatten(), preds_inv.flatten()))
    # MAPE safe: avoid divide by zero
    denom = np.where(actuals_inv == 0, 1e-8, actuals_inv)
    mape = (np.abs((actuals_inv - preds_inv) / denom)).mean() * 100

    return {"mae": mae, "rmse": rmse, "mape": mape, "preds": preds_inv, "actuals": actuals_inv}

# --------------------------
# Rolling (production) forecast
# --------------------------
def rollout_forecast(
    model: nn.Module,
    recent_window: np.ndarray,
    forecast_horizon: int,
    scaler: MinMaxScaler,
    device: torch.device
) -> np.ndarray:
    """
    Given the most recent `seq_length` raw values (unscaled), return forecast_horizon raw predictions.
    Uses model to predict in a direct multi-output fashion (preferred).
    Input:
      - recent_window: shape (seq_length, 1) in original scale (not scaled)
    """
    model.eval()
    with torch.no_grad():
        # scale the window using the scaler fitted earlier
        scaled_window = scaler.transform(recent_window.reshape(-1, 1)).astype(np.float32)  # (seq_len,1)
        xb = torch.from_numpy(scaled_window).unsqueeze(0).to(device)  # (1, seq_len, 1)
        raw_preds = model(xb).cpu().numpy().reshape(-1, 1)  # (horizon, 1) or (horizon,) -> ensure column
        preds_inv = scaler.inverse_transform(raw_preds)    # inverse transform each predicted step
    return preds_inv.flatten()  # shape (horizon,)

# --------------------------
# End-to-end helper
# --------------------------
def run_training_for_product(
    csv_path: str,
    product_id: str,
    date_col: str = "InvoiceDate",
    qty_col: str = "Quantity",
    stockcode_col: str = "StockCode",
    seq_length: int = 90,
    forecast_horizon: int = 7,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_size: int = 128,
    num_layers: int = 2,
    val_ratio: float = 0.1,
    device: Optional[torch.device] = None,
    model_save_path: str = "best_lstm.pth"
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load and aggregate to daily
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col], format='%d-%m-%Y %H:%M', errors='coerce')
    df = df.dropna(subset=[date_col, qty_col, stockcode_col])
    df.set_index(date_col, inplace=True)
    prod_df = df[df[stockcode_col] == product_id].copy()
    if prod_df.empty:
        raise ValueError(f"No data for product {product_id}")
    daily_sales = prod_df.resample('D')[qty_col].sum().fillna(0).values.reshape(-1, 1)  # (days,1)

    # 2) scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily_sales)  # shape (days,1)

    # 3) create dataset
    dataset = SalesDataset(scaled, seq_length=seq_length, forecast_horizon=forecast_horizon)

    # 4) temporal split indices for dataset (not raw days)
    total = len(dataset)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * val_ratio)
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, total))
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 5) model
    model = LSTMForecaster(input_size=1, hidden_size=hidden_size, num_layers=num_layers, forecast_horizon=forecast_horizon)
    model.to(device)

    # 6) train
    train_res = train_model(model=model, train_loader=train_loader, val_loader=val_loader,
                            device=device, epochs=epochs, lr=lr, patience=8, save_path=model_save_path)

    # 7) evaluation
    best_model = train_res["model"]
    eval_res = evaluate_model(best_model, test_loader, scaler, device)

    # 8) Save predictions CSV & model
    preds = eval_res["preds"]    # shape (n_test_samples, horizon)
    # save last row(s) as representative sample(s)
    preds_df = pd.DataFrame(preds, columns=[f"Day+{i+1}" for i in range(forecast_horizon)])
    preds_df.to_csv(f"future_forecast_{product_id}_h{forecast_horizon}.csv", index=False)
    # Save final model again
    torch.save(best_model.state_dict(), model_save_path.replace(".pth", f"_{product_id}_h{forecast_horizon}.pth"))

    results = {
        "train_history": train_res["history"],
        "best_epoch": train_res["best_epoch"],
        "best_val_loss": train_res["best_val_loss"],
        "metrics": {"mae": eval_res["mae"], "rmse": eval_res["rmse"], "mape": eval_res["mape"]},
        "predictions_file": f"future_forecast_{product_id}_h{forecast_horizon}.csv",
        "model_file": model_save_path.replace(".pth", f"_{product_id}_h{forecast_horizon}.pth"),
        "scaler": scaler,
        "model": best_model
    }
    return results

# --------------------------
# Example usage (uncomment to run)
# --------------------------
if __name__ == "__main__":
    # Example: adapt params as needed
    CSV = "/content/drive/MyDrive/refine_file.csv"
    PRODUCT = "85123A"
    out = run_training_for_product(
        csv_path=CSV,
        product_id=PRODUCT,
        seq_length=90,
        forecast_horizon=7,
        batch_size=64,
        epochs=30,
        lr=1e-3,
        hidden_size=128,
        num_layers=2,
        val_ratio=0.1,
        model_save_path="best_lstm.pth"
    )
    print("Finished. Metrics:", out["metrics"])
    # Example of production rollout forecast - get most recent raw days
    df = pd.read_csv(CSV)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M', errors='coerce')
    df.set_index('InvoiceDate', inplace=True)
    daily_sales = df[df['StockCode'] == PRODUCT].resample('D')['Quantity'].sum().fillna(0).values.reshape(-1, 1)
    recent_window = daily_sales[-90:].flatten()
    preds = rollout_forecast(out["model"], recent_window, forecast_horizon=7, scaler=out["scaler"], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Rollout preds (next 7 days):", preds)

    # plot the forcast result vs actual result
    import matplotlib.pyplot as plt
    plt.plot(preds, label='Forecast')
    plt.plot(daily_sales[-7:], label='Actual')
    plt.title(f'Forecast vs Actual for {PRODUCT}')
    plt.xlabel('Days')
    plt.ylabel('Quantity')
    plt.legend()
    plt.show()
