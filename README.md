# ASL-ImageClassification

Code for computer vision task of image classification for ASL Alphabet Letters with Keras and TensorFlow in R.

This R script is aimed at building and training a deep learning model for the classification of images representing American Sign Language (ASL) alphabet letters.

Dependencies:
The code relies on the following R libraries:

reticulate: Interface to Python within R.
keras: R interface to Keras, a high-level neural networks API.
tensorflow: R interface to TensorFlow, an open-source machine learning library.

Initial Testing with the MNIST Dataset:

Before working with the ASL alphabet images, the script uses the MNIST dataset for initial testing and model tuning. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9) and their labels.

The pixel values of the images are rescaled from a range of 0-255 to a range of 0-1 by dividing them by 255. This normalization helps in speeding up the convergence during training.

The labels are converted to one-hot encoded vectors to be compatible with the categorical cross-entropy loss used during training.

A simple baseline model is defined and trained using the MNIST dataset. This step is crucial for understanding the basic structure of neural networks and for tuning the model's hyperparameters.

ASL Alphabet Image Classification:

After the initial testing with the MNIST dataset, the script focuses on the primary objective, which is ASL alphabet image classification.

The script sets the working directory to where the ASL alphabet images are located. It loads the images and labels from a specified directory and resizes them to a target size of 224x224 pixels. Additionally, it rescales the pixel values to a range of 0-1 and splits the data into training and validation sets.

We then use a pre-trained Xception model as the base, which is an advanced deep learning model pre-trained on the ImageNet dataset. The weights of this model are frozen to retain the knowledge it has gained from pre-training. This is an example of utilizing transfer learning to save time training the preliminary model from scratch.

A custom model is built on top of the Xception model by adding:

A global average pooling layer.
A fully connected layer with ReLU activation.
A dropout layer for regularization.
A final output layer with a softmax activation function for classifying the ASL alphabet letters.
The model is compiled and trained using the ASL alphabet images. The script trains the model using the fit_generator method and evaluates it on the validation dataset.

Finally, the script visualizes some of the ASL alphabet images.

Usage:

To run this script, make sure you have R installed along with the required libraries. Set the paths in the script to the directories where your ASL alphabet images are located. Run the script in R or RStudio.

