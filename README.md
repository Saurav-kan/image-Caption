# Image- Captioner

Takes an image and returns a caption that fits that image
Also detects common objects in the image and draws red boxes around them
Upload any image (PNG, JPG, JPEG)


---
Tech Stack

Steamlit for UI
Pytorch & Torchvison for object detection
Transformers for image captoning (BLIP)
Pillow for images
NumPy for tensor operations

Packages listed in requirements.txt

Installation

1. Clone or download this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

---
Running the App

  ```bash
  python -m streamlit run app.py
```


Open the URL printed in your terminal (usually http://localhost:8501)

Upload an image to see captions and object detections
