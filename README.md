# Face Mask Detection

## Overview

This project implements a **Face Mask Detection system** using Python and OpenCV. It detects human faces in real-time through a webcam or video feed and classifies whether a person is **wearing a mask** or **not wearing a mask** using a trained deep learning model.

## Features

* Real-time **face detection** using OpenCV's Haar cascades or DNN.
* **Mask classification** using a trained Convolutional Neural Network (CNN) or pre-trained deep learning model.
* Highlights detected faces with **bounding boxes**:

  * Green box for **mask detected**
  * Red box for **no mask detected**
* Can process **live webcam feed** or **video files**.

## Technologies Used

* Python 3
* OpenCV (`cv2`) for computer vision
* TensorFlow / Keras for deep learning model
* NumPy for numerical computations
* Matplotlib for optional visualization

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Open `Face-mask-detection.ipynb` in Jupyter Notebook.
2. Run all cells sequentially.
3. If using webcam, ensure your camera is connected.
4. The notebook will display the webcam feed with **real-time mask detection**.

### Example

* **Mask detected:** Green bounding box appears around the face with label `Mask`.
* **No mask detected:** Red bounding box appears with label `No Mask`.

## Notes

* Accuracy depends on the quality of the trained model.
* Ensure the environment has OpenCV, TensorFlow, and other dependencies installed.
* Can be extended to **mobile or edge devices** for real-time public safety monitoring.

## Future Enhancements

* Integrate **YOLOv5 or SSD** for faster and more accurate detection.
* Deploy as a **real-time monitoring system** in public areas.
* Add **alert system** for no-mask detection.
* Optimize the model for low-latency real-time inference.

## License

This project is licensed under the MIT License.
