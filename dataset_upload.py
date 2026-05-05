import gdown

# Download dataset from Google Drive
url = "https://drive.google.com/uc?id=1TW-MC6Uhfd08YB9zFpNfzqZhVW1cXhVF"
output = "Dataset-SolarTechLab.csv"

gdown.download(url, output, quiet=False)

# Now run your preprocessing
df, X_scaled, y_scaled, scaler_X, scaler_y = preprocess(
    "Dataset-SolarTechLab.csv", plot=True
)

print(f"Rows: {len(df)}")
print(f"Time range: {df['Time'].min()} → {df['Time'].max()}")
