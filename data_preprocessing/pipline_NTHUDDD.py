import cv2
import numpy as np
from skimage import io
import face_alignment
import matplotlib.pyplot as plt
import os

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

input_path = "test.jpg"  
img = io.imread(input_path)

# Detect landmarks
landmarks = fa.get_landmarks(img)
if not landmarks:
    raise Exception("No face found")
landmarks = landmarks[0]

eye_landmarks = landmarks[36:48]  
min_x = int(np.min(eye_landmarks[:, 0])) - 30
max_x = int(np.max(eye_landmarks[:, 0])) + 30
min_y = int(np.min(eye_landmarks[:, 1])) - 40
mid_y = int(np.mean(landmarks[48:68, 1])) + 40  

# Limit of picture
h, w = img.shape[:2]
x1 = max(min_x, 0)
x2 = min(max_x, w)
y1 = max(min_y, 0)
y2 = min(mid_y, h)

face_crop = img[y1:y2, x1:x2]

# blurred background
mask = np.zeros_like(img)
mask[y1:y2, x1:x2] = 255
blurred = cv2.GaussianBlur(img, (31, 31), 0)
img_blurred_focus = np.where(mask == 255, img, blurred)


plt.figure(figsize=(12, 4))
titles = ["Original", "Face Crop", "Blurred Background"]
images = [img, face_crop, img_blurred_focus]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray' if images[i].ndim == 2 else None)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
