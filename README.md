# Driver Drowsiness Detection With Lightweight Model
Developed a real-time drowsiness detection system on edge devices (NVIDIA Jetson Nano, Raspberry Pi 4) . Implemented and benchmarked lightweight models (MobileNet, EfficientNet,..)

# Dataset
NTHU-DDD 2 : https://www.kaggle.com/datasets/banudeep/nthuddd2
# Data Preprocessing
The preprocessing pipeline first applies background blurring on the original image to suppress irrelevant contextual information, while preserving the facial region of interest. Subsequently, the focused facial region is cropped and used as the final input for model training and inference. The final output image both limits information to the facial area (content cropping) and blurs irrelevant areas (attention blurring), allowing the model to focus maximally on the eye features.
![data preprocessing](https://drive.google.com/uc?id=1FoywLigZ_XGx1F9xtbYlYCg6BiUIrEbR)

# Edge devices
**Nvidia Jetson Nano :**
<p align="center">
  <img 
    src="https://drive.google.com/uc?id=1ITnU0EXX1q2Ea6xA6INjn2sHLBRM9Usd" 
    width="500"
    style="transform: rotate(-90deg);"
  />
</p>

**Raspberry Pi 4 (4GB Ram) :**
<p align="center">
  <img 
    src="https://drive.google.com/uc?id=1xIY8R8dQHlhVCkgCGIA1Da86MDq26d91" 
    width="500"
    style="transform: rotate(-90deg);"
  />
</p>

# Benchmark
# PC : GPU Nivida RTX3050 VRAM 6GB
![compare metrics](https://drive.google.com/uc?id=1Pp-cANCdO-ErajQ19Bx1jD6YxXwEkBrO)
<p align="center">
  <img 
    src="https://drive.google.com/uc?id=1qU5fV1MkSSpDWhFvxA9zzin6mwZQLVug" 
    width="500";"
  />
</p>

# Nvidia Jetson Nano
Total images inferences : 6651 . 
- MobileNet : 12.4 FPS
- ShuffleNet : 15.8 FPS
- Resnet-18 : 11.7 FPS
- Resnet-50 : update ....
- EfficientNet : update ....
# Raspberrp Pi 4
This device has very limited hardware resources (CPU only).
Total images inferences : 6651 
- Resnet-18 : 1.14 FPS
# Quantization & Pruning
Apply this technique only to resnet-50 
