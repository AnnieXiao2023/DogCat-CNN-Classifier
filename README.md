# DogCat-CNN-Classifier
A deep learning project using a Convolutional Neural Network (CNN) in TensorFlow to classify images of dogs and cats. The model is trained on a dataset of 4000 dog images and 4000 cat images and tested on a dataset of 1000 images each. Includes predictions for single images.

# Dataset
The dataset used in this project contains:
- **Training data**: 4,000 images of dogs and 4,000 images of cats.
- **Testing data**: 1,000 images of dogs and 1,000 images of cats.

You can download the dataset using this [link](https://drive.google.com/drive/folders/1GmRyoiziQnQ_tqi6lULRbV2LNm0V06h7?dmr=1&ec=wgc-drive-hero-goto).

# Technologies Used
- **TensorFlow**: Integrated with Keras for building and training the CNN model.
- **Python**: The primary programming language for this project.
- **Google Colab**: Used for model training and experimentation.

## Installation
git clone https://github.com/AnnieXiao2023/DogCat-CNN-Classifier.git
cd DogCat-CNN-Classifier

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows:
venv\Scripts\activate
# For Mac/Linux:
source venv/bin/activate

#Install Required Libraries
pip install tensorflow keras numpy matplotlib

#Download the Dataset
DogCat-CNN-Classifier/
│
├── train/               # Training dataset (contains dog and cat images)
├── test/                # Test dataset (contains dog and cat images)
├── single_prediction/   # Folder for prediction images

#Run the Project
python train_model.py
python evaluate_model.py
python predict.py --image_path ./single_prediction/your_image.jpg

### Read More
To learn more about this project, check out my detailed Medium post:  
https://medium.com/@taixiongbaobao/building-a-image-classifier-using-cnn-in-tensorflow-da05a01820cf

