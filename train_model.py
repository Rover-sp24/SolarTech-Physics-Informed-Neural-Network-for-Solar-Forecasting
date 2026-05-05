"""
Physics-Informed Neural Network (PINN) for Solar Power Forecasting
===================================================================
Base: original working version (R² = 0.93, normalised throughout)

Safe improvements added on top:
  [1] LR Scheduler  — ReduceLROnPlateau, patience=200, 1500 epochs
  [2] Grad clipping — max_norm=1.0, prevents parameter explosion
  [3] Softplus      — DataNN output >= 0, no negative power predictions
  [4] α clamp       — (0.1, 0.9), physics pipeline can never be ignored

What we deliberately kept from the original:
  - Both pipelines work in NORMALISED space throughout
  - P_phys uses normalised G and T — same scale as P_data and P_true
  - No raw/normalised conversion anywhere — this is why R²=0.93 worked
  - η and β are numerically consistent (not real-world interpretable)
    but the model predictions are physically correct

Architecture:
  Inputs [G_norm, T_norm]
      ├──► [BLUE]  DataNN          → P_data  (normalised, softplus >= 0)
      └──► [GREEN] ParameterNN     → η_ω, β_ω
                        └──► PhysicsEq(G_norm, T_norm, η, β) → P_phys
                                    ↓
                         LearnablePhysicsParams  (α ∈ 0.1–0.9)
                          P_pred = α·P_data + (1−α)·P_phys
                                    ↓
                    inverse_transform → real Watts for evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.style.use("seaborn-v0_8")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────
# 1. DATA — from preprocessing cell
#    df, X_scaled, y_scaled, scaler_X, scaler_y = preprocess(path)
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# 2. BLUE PIPELINE — Data Neural Network
#    Normalised [G, T] → P_data (normalised scale)
# ─────────────────────────────────────────────────────────────

class DataNN(nn.Module):
    """
    Normalised [G, T, hour_sin, hour_cos, doy_sin, doy_cos] → P_data.
    6 inputs instead of 2 — time features let the model learn
    diurnal and seasonal patterns that G and T alone cannot capture.
    Softplus output ensures P_data >= 0 in normalised space.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),   # ← 6 inputs: G, T + 4 time features
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return torch.nn.functional.softplus(self.net(x))


# ─────────────────────────────────────────────────────────────
# 3. GREEN PIPELINE — Parameter Neural Network + Physics Equation
#    Normalised [G, T] → η_ω, β_ω → P_phys (normalised scale)
#
#    NOTE: G and T are normalised here — η and β are therefore
#    numerically scaled to match. They are not directly comparable
#    to real panel datasheets but the model predictions are correct.
# ─────────────────────────────────────────────────────────────

class ParameterNN(nn.Module):
    """
    Normalised [G, T, hour_sin, hour_cos, doy_sin, doy_cos] → η_ω, β_ω.
    Time features allow η and β to vary with time of day and season,
    capturing panel behaviour that irradiance and temperature alone miss.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),   # ← 6 inputs: G, T + 4 time features
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        out  = self.net(x)
        eta  = torch.sigmoid(out[:, 0:1])          # η ∈ (0, 1)
        beta = torch.sigmoid(out[:, 1:2]) * 0.02   # β ∈ (0, 0.02)
        return eta, beta


def pv_physics_equation(G, T, eta, beta, T_ref=25.0):
    """
    Physics equation uses only G and T from the 6-feature input.
    x[:, 0] = G_norm, x[:, 1] = T_norm — time features don't enter
    the physics equation directly, only through η and β via ParameterNN.
    Output P_phys is in normalised scale.
    """
    return eta * G * (1.0 - beta * (T - T_ref))


# ─────────────────────────────────────────────────────────────
# 4. CONVERGENCE BLOCK — Learnable Physics Parameters
# ─────────────────────────────────────────────────────────────

class LearnablePhysicsParams(nn.Module):
    """
    P_pred = α · P_data + (1−α) · P_phys

    [4] α clamped to (0.1, 0.9) so neither pipeline is ever
    fully ignored. Previously α could escape to 1.0 and kill
    the physics pipeline entirely.
    """
    def __init__(self):
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, P_data, P_phys):
        alpha  = 0.1 + 0.8 * torch.sigmoid(self.alpha_raw)
        P_pred = alpha * P_data + (1.0 - alpha) * P_phys
        return P_pred, alpha


# ─────────────────────────────────────────────────────────────
# 5. FULL PINN
# ─────────────────────────────────────────────────────────────

class PINN(nn.Module):
    def __init__(self, data_hidden=64, param_hidden=32):
        super().__init__()
        self.data_nn  = DataNN(data_hidden)
        self.param_nn = ParameterNN(param_hidden)
        self.merge    = LearnablePhysicsParams()

    def forward(self, x):
        """
        x : normalised [G, T]
        Returns P_pred, P_phys, eta, beta, alpha — all in normalised scale.
        Inverse-transform P_pred at evaluation time to get real Watts.
        """
        G = x[:, 0:1]
        T = x[:, 1:2]

        P_data        = self.data_nn(x)
        eta, beta     = self.param_nn(x)
        P_phys        = pv_physics_equation(G, T, eta, beta)
        P_pred, alpha = self.merge(P_data, P_phys)

        return P_pred, P_phys, eta, beta, alpha


# ─────────────────────────────────────────────────────────────
# 6. HYBRID LOSS
#    Everything is in normalised scale — no conversion needed.
# ─────────────────────────────────────────────────────────────

def hybrid_loss(P_pred, P_phys, P_true, lambda_phys=0.5):
    data_loss = torch.mean((P_pred - P_true) ** 2)
    phys_loss = torch.mean((P_phys - P_true) ** 2)
    total     = data_loss + lambda_phys * phys_loss
    return total, data_loss, phys_loss


# ─────────────────────────────────────────────────────────────
# 7. TRAINING
# ─────────────────────────────────────────────────────────────

def train(model, X_train_t, y_train_t,
          epochs=1500, lr=1e-3, lambda_phys=0.5, log_every=100):
    """
    [1] ReduceLROnPlateau scheduler — halves LR if no improvement
        for 200 epochs, prevents LR from decaying too aggressively.
    [2] Gradient clipping — max_norm=1.0 after every backward pass.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200
    )

    history = {"total": [], "data": [], "physics": [], "alpha": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        P_pred, P_phys, eta, beta, alpha = model(X_train_t)
        loss, d_loss, p_loss = hybrid_loss(
            P_pred, P_phys, y_train_t, lambda_phys
        )
        loss.backward()

        # [2] Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step(loss)

        history["total"].append(loss.item())
        history["data"].append(d_loss.item())
        history["physics"].append(p_loss.item())
        history["alpha"].append(alpha.item())
        history["lr"].append(optimizer.param_groups[0]['lr'])

        if epoch % log_every == 0:
            print(
                f"  Epoch {epoch:>5d} | "
                f"Total: {loss.item():.5f} | "
                f"Data: {d_loss.item():.5f} | "
                f"Phys: {p_loss.item():.5f} | "
                f"α: {alpha.item():.3f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    return history


# ─────────────────────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(model, X_t, y_t, scaler_y):
    model.eval()
    with torch.no_grad():
        P_pred, P_phys, eta, beta, alpha = model(X_t)

    preds = scaler_y.inverse_transform(P_pred.numpy())
    phys  = scaler_y.inverse_transform(P_phys.numpy())
    truth = scaler_y.inverse_transform(y_t.numpy())

    rmse = np.sqrt(mean_squared_error(truth, preds))
    mae  = mean_absolute_error(truth, preds)
    r2   = r2_score(truth, preds)

    print(f"\n── Test Evaluation ────────────────────────────────")
    print(f"  RMSE        : {rmse:.4f} W")
    print(f"  MAE         : {mae:.4f} W")
    print(f"  R²          : {r2:.4f}")
    print(f"  α           : {alpha.item():.3f}  "
          f"({'data-dominant' if alpha.item() > 0.5 else 'physics-dominant'})")
    print(f"  η mean      : {eta.mean().item():.4f}")
    print(f"  β mean      : {beta.mean().item():.5f}")

    return preds, phys, truth, eta.numpy(), beta.numpy()


# ─────────────────────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────────────────────

def plot_loss(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history["total"],   label="Total Loss")
    axes[0].plot(history["data"],    label="Data Loss",    linestyle="--")
    axes[0].plot(history["physics"], label="Physics Loss", linestyle=":")
    axes[0].set_title("Hybrid Loss Breakdown")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()

    axes[1].plot(history["alpha"], color="purple")
    axes[1].axhline(0.5, linestyle="--", color="gray",  linewidth=0.8)
    axes[1].axhline(0.1, linestyle=":",  color="red",   linewidth=0.8, label="floor (0.1)")
    axes[1].axhline(0.9, linestyle=":",  color="blue",  linewidth=0.8, label="ceiling (0.9)")
    axes[1].set_title("α — Data vs Physics Weight")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("α  (1=data, 0=physics)")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)

    axes[2].plot(history["lr"], color="darkorange")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    plt.show()


def plot_predictions(truth, preds, phys):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y, label, color in [
        (axes[0], preds, "PINN (Data + Physics)", "steelblue"),
        (axes[1], phys,  "Physics Pipeline Only",  "seagreen"),
    ]:
        ax.scatter(truth, y, s=5, alpha=0.4, color=color)
        lims = [min(truth.min(), y.min()), max(truth.max(), y.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5)
        ax.set_xlabel("True PV Power (W)")
        ax.set_ylabel("Predicted PV Power (W)")
        ax.set_title(label)

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    plt.show()


def plot_learned_parameters(X_test_t, eta, beta, scaler_X):
    # Column 0 is G_tilt — inverse transform full array then take col 0
    X_orig = scaler_X.inverse_transform(X_test_t.numpy())
    G_orig = X_orig[:, 0]   # real irradiance in W/m²

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(G_orig, eta,  s=4, alpha=0.4, color="steelblue")
    axes[0].set_xlabel("Irradiance G (W/m²)")
    axes[0].set_ylabel("η_ω  (efficiency)")
    axes[0].set_title("Learned η vs Irradiance")

    axes[1].scatter(G_orig, beta, s=4, alpha=0.4, color="seagreen")
    axes[1].set_xlabel("Irradiance G (W/m²)")
    axes[1].set_ylabel("β_ω  (temperature coefficient)")
    axes[1].set_title("Learned β vs Irradiance")

    plt.tight_layout()
    plt.savefig("learned_params.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Reproducibility ───────────────────────────────────────
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load via preprocessing cell ───────────────────────────
    # df, X_scaled, y_scaled, scaler_X, scaler_y = preprocess("Dataset-SolarTechLab.csv")

    # ── Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=SEED
    )

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    X_train_t, X_test_t = to_t(X_train), to_t(X_test)
    y_train_t, y_test_t = to_t(y_train), to_t(y_test)

    # ── Build & train ─────────────────────────────────────────
    model = PINN(data_hidden=64, param_hidden=32)

    print("=" * 55)
    print("Training PINN (Data NN + Parameter NN → Hybrid Loss)")
    print("=" * 55)

    history = train(
        model, X_train_t, y_train_t,
        epochs=4000, lr=1e-3, lambda_phys=0.5
    )

    # ── Evaluate ──────────────────────────────────────────────
    preds, phys, truth, eta, beta = evaluate(
        model, X_test_t, y_test_t, scaler_y
    )

    # ── Plots ─────────────────────────────────────────────────
    plot_loss(history)
    plot_predictions(truth, preds, phys)
    plot_learned_parameters(X_test_t, eta, beta, scaler_X)
