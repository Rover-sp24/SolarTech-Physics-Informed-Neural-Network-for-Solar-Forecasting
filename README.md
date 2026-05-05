# ☀️ SolarPINN — Physics-Informed Neural Network for Solar Power Prediction

##  Overview

SolarPINN is a hybrid machine learning system that predicts photovoltaic (PV) power output using both **data-driven learning** and **physical modeling**.

Unlike traditional models, this project combines:

* Neural Networks (data learning)
* Physics equations (domain knowledge)

This results in **higher accuracy and better generalization** for real-world solar energy prediction.

---

##  Key Idea

Instead of relying only on data, the model learns:

> **PV Power = f(Data Patterns + Physical Laws)**

The final prediction is a blend of:

* Data Neural Network (learned patterns)
* Physics Equation (solar energy behavior)

---

##  Project Structure

```
SolarPINN/
├── data/
│   └── Dataset-SolarTechLab.csv
├── preprocessing.py
├── train_model.py
├── dashboard.html
├── requirements.txt
└── README.md
```

---

##  Features

###  Data Preprocessing

* Cleans dataset (removes NaNs, invalid values)
* Drops nighttime data (no solar generation)
* Removes outliers (physical + statistical)
* Feature engineering:

  * Irradiance (G_tilt)
  * Temperature (T_air)
  * Cyclical time encoding (hour, day of year)
* Normalization using StandardScaler

---

###  Model (PINN Architecture)

The model consists of:

#### 1. Data Neural Network

* Learns patterns from data
* Ensures non-negative output using Softplus

#### 2. Parameter Neural Network

* Learns physical parameters:

  * Efficiency (η)
  * Temperature coefficient (β)

#### 3. Physics Equation

```
P = η · G · (1 − β (T − T_ref))
```

#### 4. Hybrid Output

```
P_pred = α · P_data + (1 − α) · P_phys
```

* α is learnable (balances data vs physics)

---

###  Training Enhancements

* Learning rate scheduler (ReduceLROnPlateau)
* Gradient clipping
* Physics-based loss function
* Stable convergence with normalized inputs

---

###  Dashboard (Frontend)

Interactive UI to:

* Adjust environmental inputs:

  * Irradiance
  * Temperature
  * Time of day
  * Day of year
* View:

  * Predicted power output
  * Efficiency estimation
  * Data vs Physics contribution
  * Model metrics (R², RMSE)

---

##  Model Performance

| Metric          | Value   |
| --------------- | ------- |
| R² Score        | ~0.96   |
| RMSE            | ~13.7 W |
| Training Epochs | 4000    |

---

##  Dataset

The dataset is automatically downloaded from Google Drive.

No manual download required.

---

##  Installation

### 1. Clone the repository

```
git clone https://github.com/Rover-sp24/SolarPINN.git
cd SolarPINN
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
pip install gdown
```

---

##  How to Run

### Step 1: Download dataset + preprocess

```python
import gdown

url = "https://drive.google.com/uc?id=1TW-MC6Uhfd08YB9zFpNfzqZhVW1cXhVF"
gdown.download(url, "Dataset-SolarTechLab.csv", quiet=False)

df, X_scaled, y_scaled, scaler_X, scaler_y = preprocess(
    "Dataset-SolarTechLab.csv", plot=True
)
```

---

### Step 2: Train the model

```
python train_model.py
```

---

### Step 3: Open dashboard

Simply open:

```
dashboard.html
```

in your browser.

---

##  Tech Stack

* Python
* PyTorch
* NumPy / Pandas
* Scikit-learn
* Matplotlib
* HTML / CSS / JavaScript

---

##  Applications

* Solar energy forecasting
* Smart grid optimization
* Renewable energy research
* Physics-informed machine learning

---

##  Future Improvements

* Deploy model as API (Flask / FastAPI)
* Connect dashboard to live backend
* Add real-time weather integration
* Extend to multi-panel systems

---

##  Acknowledgements

Inspired by research in:

* Physics-Informed Neural Networks (PINNs)
* Renewable energy forecasting

---

##  Author

Anamika K T & Sibani B
BTech CSE (AI & ML)

---

## ⭐ If you like this project

Give it a star and share it!
