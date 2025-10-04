CNN + ANN Classification for IT/PSI Image Prediction

Overview

This project predicts IT/PSI classes from image data using a  Convolutional Neural Network (CNN) and Artificial Neural Network (ANN).
Data is preprocessed using ImageDataGenerator (rescaling, shear, zoom, horizontal flip).
CNN layers extract spatial features, followed by ANN layers for binary classification.
Hyperparameters are optimized using Optuna.
The model is trained, evaluated using AUC, and can predict both batches and individual images.
The model has been deployed and is ready for inference.

Workflow / Steps

1. Data Preparation

Load images from training and test directories using ImageDataGenerator.
Training data is augmented with shear, zoom, and horizontal flip.
Test data is rescaled (normalization).
Image target size: (64, 64), batch size: 32.

2. Model Architecture

CNN Layers:
Conv2D → MaxPool2D → SpatialDropout2D
Conv2D → MaxPool2D → SpatialDropout2D
ANN Layers:
Dense layers with ReLU activation
Dropout layers for regularization
Output layer with sigmoid activation (binary classification)

3. Hyperparameter Optimization

Use Optuna with TPESampler to optimize:
Filters, kernel size, pool size, strides in CNN
Dropout rates for CNN and ANN layers
Number of units in ANN layers
Optimizer choice (adam, sgd, rmsprop, adagrad)
Learning rate and batch size
Objective: maximize test AUC score

4. Model Training

Train the final model with best hyperparameters from Optuna.
Fit on training set with validation on test set, 20 epochs.
Evaluate model on training and test sets (loss and AUC).

5. Model Evaluation

Metrics:
Train Loss and AUC
Test Loss and AUC

6. Single Image Prediction

Load single images from a folder.
Preprocess images to (64, 64) and scale to [0,1].
Predict class using trained model.
Map prediction indices back to class names.
Display images and predicted classes in a DataFrame using base64 encoding for visualization.

7. Deployment

Model has been deployed and is ready for inference.
New images can be predicted directly using the same preprocessing pipeline.

Model has been deployed and is ready for inference.

New images can be predicted directly using the same preprocessing pipeline.
