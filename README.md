# Intel Unnati Industrial Training
## Topic : Conquering Fashion MNIST with CNNs using Computer Vision

### Problem Statement
The project aims to develop a convolutional neural network (CNN) model for accurately classifying images from the Fashion MNIST dataset. Computer vision techniques and concepts will be employed to enable the model to interpret and understand visual data, specifically focusing on clothing items.

### Our Approach
The model is a sequential neural network architecture designed for image classification tasks. It starts with a series of convolutional layers with increasing filter sizes and uses the ReLU activation function to introduce non-linearity. Batch normalization is applied after certain layers to normalize the outputs and improve training stability. Dropout layers are included to prevent overfitting by randomly deactivating a fraction of neurons during training. Max pooling layers are used for spatial downsampling. The flattened output is then passed through fully connected layers with ReLU activation. Batch normalization and dropout are again applied before the final dense layer with softmax activation, which produces the class probabilities. This model architecture leverages the power of convolutional neural networks for feature extraction from images and incorporates regularization techniques for better generalization and accuracy.
