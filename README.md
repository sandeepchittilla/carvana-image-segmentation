# carvana_segmentation
Automatically remove background image/text/noise from images with foreground object


## Dataset 
We use the car dataset from the Kaggle competition : [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/overview) for this purpose.

## Solution
We model this as an Image Segmentation problem and build a Convolutional Neural Network (CNN) for classification between foreground and background. The dataset has ~5K images of cars with masks. We use the U-Net architecture to build the CNN and train it on ~4K images and test on ~1K.

The Python notebook carvana_segmentation.ipynb has code for :
- Preliminary EDA
- Image Preprocessing (Resize, Reshape, Image Augmentation etc.)
- Defining a CNN with U-Net Architecture
- Training, Model Validation and Quality Assesment Metrics (IoU, F1, PR curves)
- Predictions on images outside Carvana Dataset

There is also a .py notebook to easily identify changes between commits
