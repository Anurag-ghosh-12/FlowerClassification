# Flower Classification Project 🌸 using Convolutional Neural Network
Flower Classification with CNN &amp; Data Augmentation
<br>
<p align="center">
  <img src="https://www.tensorflow.org/static/tutorials/load_data/images_files/output_oV9PtjdKKWyI_0.png" alt="Image">
</p>

**The flower dataset is a collection of images depicting various types of flowers. It comprises 3,670 images distributed across five categories:**
- <span style="color: #4CAF50;">roses</span>
- <span style="color: #FFC107;">tulips</span>
- <span style="color: #FF5722;">daisies</span>
- <span style="color: #9C27B0;">dandelions</span>
- <span style="color: #FF9800;">sunflowers</span>

**Each image is labeled with its corresponding flower type, making it suitable for supervised machine learning tasks such as image classification.**

<p align="center">
  <img src="https://www.tensorflow.org/static/tutorials/load_data/images_files/output_AAY3LJN28Kuy_0.png" alt="Image">
</p>

## Data Preprocessing
Before feeding the images into a machine learning model, it's essential to preprocess them to ensure uniformity and facilitate model training. The preprocessing steps applied to the flower dataset are as follows:

### 1. Image Loading and Resizing
The images are loaded using OpenCV (cv2) library, and their dimensions are resized to a uniform size of 180x180 pixels. This resizing step ensures that all images have the same dimensions, which is a requirement for training most deep learning models.

### 2. Normalization
The pixel values of the images are scaled to the range [0, 1] by dividing them by 255. This normalization step ensures that the pixel values are within a consistent range, making it easier for the model to learn and converge efficiently.

#### Normalization of pixel values
X_train_scaled = X_train / 255<br>
X_test_scaled = X_test / 255

## Model Building and Evaluation
### Neural Network (CNN) Model
Initially, a Convolutional Neural Network (CNN) model was developed to classify the flower images. The CNN architecture consisted of convolutional and pooling layers followed by fully connected layers for classification. The model was trained on the preprocessed training data and evaluated on the test data to measure its performance.

Accuracy: 58%
### CNN with Data Augmentation
To enhance the model's accuracy and robustness, data augmentation techniques were applied during training. Data augmentation involves generating new training samples by applying random transformations such as rotation, flipping, and zooming to the existing images. This augmented training data was used to train the CNN model, leading to improved performance compared to the baseline CNN model.

Accuracy:64%
<p align="center">
  <img src="https://github.com/Anurag-ghosh-12/FlowerClassification/blob/main/heatmap_cnn_dataug.png?raw=true" alt="Image">
</p>

### How to Use 🚀
1. 🌟 Clone this repository to your local machine.
2. 💻 Open the project in your favorite code editor.
3. 🚀 Train the model using the provided dataset.
4. 🌸 Test the model's performance on new flower images.
5. 🎉 Enjoy the beauty of flower classification!
<br>
If you have any suggestions or contributions to further enhance the project, feel free to submit a pull request or open an issue on the GitHub repository. Your feedback and contributions are highly appreciated and will help improve the overall quality and effectiveness of the project.



