# Computer Vision & Natural Language Processing
This is the third team project at Startup Campus

## Case 1: Computer Vision - Digital Image Processing 

**File:** [03_Tim4_1](https://github.com/Elvhan/Computer-Vision-and-Natural-Language-Processing/blob/main/03_Tim4_1.ipynb)

### Overview
This project aims to enhance low-light images using two distinct techniques: Max Pooling and Contrast Limited Adaptive Histogram Equalization (CLAHE). Both methods are applied to brighten dark images, followed by a comparative analysis to determine their effectiveness.

### Objectives
- Implement Max Pooling to enhance brightness by merging pixel values through the grouping of adjacent pixels in low-light images.
- Utilize CLAHE to improve image contrast and brightness, ensuring local details are preserved while preventing over-amplification.
- Compare the results of Max Pooling and CLAHE to assess which method achieves better low-light enhancement.

---

## Case 2: Computer Vision - Transfer Learning with Pre-trained CNN 

**File:** [03_Tim4_2](https://github.com/Elvhan/Computer-Vision-and-Natural-Language-Processing/blob/main/03_Tim4_2.ipynb)

### Overview
This project develops a computer vision model to recognize handwritten digits (0-9) using the Modified National Institute of Standards and Technology (MNIST) dataset. Pre-trained models from PyTorch, such as DenseNet, ResNet, and Vision Transformer (ViT), are implemented to achieve high classification accuracy, along with an evaluation of the effects of freezing specific parts of the neural network layers.

### Objectives
- To investigate the applicability of transfer learning for image classification tasks.
- To assess the impact of different training strategies, specifically comparing the performance of models with frozen layers and fine-tuned models on a target dataset.

---

## Case 3: Computer Vision - Real-time Object Detection

**File:** [03_Tim4_3](https://github.com/Elvhan/Computer-Vision-and-Natural-Language-Processing/blob/main/03_Tim4_3.ipynb)

### Overview
This project demonstrates the use of YOLOv5, a pre-trained convolutional neural network, for real-time object detection. The model is sourced from the PyTorch Hub and applied to identify and locate multiple objects in a YouTube video.

### Objectives
- Develop a real-time object detection pipeline using YOLOv5 and OpenCV.
- Process YouTube video frames and detect objects with bounding boxes.
- Visualize detection results in real-time and output the processed video.

---

## Case 4: Natural Language Processing - Text Classification

**File:** [03_Tim4_4](https://github.com/Elvhan/Computer-Vision-and-Natural-Language-Processing/blob/main/03_Tim4_4.ipynb)

### Overview
This project utilizes a fine-tuned BERT model to classify disaster-related tweets from platform X (formerly Twitter), aiming to identify indirect disaster reports and support emergency response systems.

### Objectives
- Clean and preprocess the tweet text data to make it suitable for model input.
- Adapt a pre-trained BERT model for the specific task of text classification.
- Assess the model's ability to correctly classify tweets as disaster-related or non-disaster-related.
- Optimize the model through fine-tuning and transfer learning techniques to achieve high classification accuracy.
