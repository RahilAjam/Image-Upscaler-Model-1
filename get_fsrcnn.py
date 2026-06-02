import requests

# Downloading FSRCNN (Fast Super-Resolution) - Small & Stable
url = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
filename = "FSRCNN_x2.pb"

print(f"⬇️ Downloading {filename}...")
response = requests.get(url, stream=True)

with open(filename, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("✅ Download Complete! You can now run app.py")