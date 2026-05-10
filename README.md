# Image Processing Desktop Application

A simple desktop application built with **Python**, **OpenCV**, and **Tkinter** that allows users to load images and apply basic image-processing operations through an interactive graphical interface.

---

## 📌 Project Overview

This project provides a user-friendly GUI for performing common image-processing tasks without using the command line.

Users can:

* Load images from disk
* Apply filters and transformations
* Detect edges
* Compare histograms
* Save processed results

The application is designed as an educational project for learning the fundamentals of image processing and GUI development using Python.

---

## 🚀 Features

### ✅ Image Operations

* Open JPEG/PNG images
* Reset image to original state
* Save processed image

### 🎨 Color Processing

* Display RGB channels separately
* Convert image to grayscale

### 🔄 Geometric Transformations

* Rotate images
* Zoom using:

  * Nearest Neighbor interpolation
  * Bilinear interpolation
* Flip horizontally or vertically

### ✨ Image Enhancement

* Histogram Equalization
* Gamma Correction

### 🌫️ Filtering

* Gaussian Blur with adjustable kernel size slider

### 🧠 Edge Detection

* Canny Edge Detection

### 📊 Comparison Mode

* Display processed image and histogram simultaneously

---

## 🛠️ Technologies Used

* Python 3
* OpenCV
* Tkinter
* Pillow (PIL)

---

## 📂 Project Structure

```bash
project-folder/
│
├── main.py
├── README.md
├── requirements.txt
└── assets/
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image-processing-app.git
cd image-processing-app
```

### 2. Install Dependencies

```bash
pip install opencv-python pillow
```

> Tkinter comes pre-installed with Python.

---

## ▶️ Run the Application

```bash
python main.py
```

---

## 📷 Supported Operations

| Feature                | Description                         |
| ---------------------- | ----------------------------------- |
| Open Image             | Load image from disk                |
| RGB Channels           | Show Red, Green, Blue channels      |
| Grayscale              | Convert image to black & white      |
| Rotation               | Rotate image                        |
| Zoom                   | Nearest Neighbor & Bilinear scaling |
| Flip                   | Horizontal & Vertical flipping      |
| Histogram Equalization | Improve image contrast              |
| Gamma Correction       | Adjust brightness                   |
| Gaussian Blur          | Smooth image using kernel slider    |
| Edge Detection         | Detect edges using Canny            |
| Compare                | Show image + histogram              |
| Reset                  | Restore original image              |
| Save Image             | Save processed image                |

---

## 🧪 Example OpenCV Functions Used

```python
# Read image
img = cv2.imread('file.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Split channels
b, g, r = cv2.split(img)

# Gaussian blur
blur = cv2.GaussianBlur(img, (11,11), 0)

# Edge detection
edges = cv2.Canny(gray, 100, 200)
```

---

## 🎯 Project Goals

* Learn the basics of digital image processing
* Practice GUI development with Tkinter
* Understand OpenCV image operations
* Build an interactive desktop application

---

## 👥 Team Requirements

* Team size: **3–5 members**
* Each member should explain at least one feature during the demo

---

## 📋 Deliverables

* Fully working Python application
* Clean and readable code
* Printed report explaining the implementation
* Live demonstration of all features

---

## 📊 Grading Criteria

| Criteria                    | Weight |
| --------------------------- | ------ |
| All features work correctly | 50%    |
| GUI design & usability      | 20%    |
| Code readability & report   | 15%    |
| Live demo & explanation     | 15%    |

---

## 💡 Notes

* Keep the design simple and functional
* Focus on completing all required features
* Comment your code for readability

---

## 📄 License

This project is for educational purposes.
