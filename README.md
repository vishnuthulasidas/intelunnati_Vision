# Intel Unnati Industrial Training
### Topic : Conquering Fashion MNIST with CNNs using Computer Vision
### Team Members : Vishnu T, Akhil Ashok R
### College : Saintgits College of Engineering

### Problem Statement
The project aims to develop a convolutional neural network (CNN) model for accurately classifying images from the Fashion MNIST dataset. Computer vision techniques and concepts will be employed to enable the model to interpret and understand visual data, specifically focusing on clothing items.

### Our Approach
We used TensorFlow and Keras frameworks for efficient neural network development.
The model is a sequential neural network architecture designed for image classification tasks. It starts with a series of convolutional layers with increasing filter sizes and uses the ReLU activation function to introduce non-linearity. Batch normalization is applied after certain layers to normalize the outputs and improve training stability. Dropout layers are included to prevent overfitting by randomly deactivating a fraction of neurons during training. Max pooling layers are used for spatial downsampling. The flattened output is then passed through fully connected layers with ReLU activation. Batch normalization and dropout are again applied before the final dense layer with softmax activation, which produces the class probabilities. This model architecture leverages the power of convolutional neural networks for feature extraction from images and incorporates regularization techniques for better generalization and accuracy.

### Result
Our model achieved an impressive accuracy of <b>92.6%</b> on the test dataset, demonstrating
its effectiveness in accurately classifying the images. This high accuracy reflects the
model’s ability to learn and generalize well from the training data to make accurate
predictions on unseen instances. The achieved accuracy serves as a strong indicator
of the model’s performance and its capability to handle the image classification task
effectively.

### Intel Specific Optimisations Used
#### Intel Extension for TensorFlow
The inclusion of Intel Extension for TensorFlow resulted in a significant reduction in
training time for the model. Without the extension, each epoch took approximately
250 seconds to complete, whereas with the extension, the training time was reduced
to just 40 seconds. This corresponds to an impressive improvement of 84% in training
time.
#### OpenVINO Model Optimizer
The utilization of the OpenVINO Model Optimizer resulted in a significant reduction
in the inference time of the model. Prior to converting the model into the Intermediate Representation (IR) format using the OpenVINO Model Optimizer, the inference
time for each instance was approximately 1.6 seconds. However, after the conversion,
the inference time was drastically reduced to just 0.3 seconds. This corresponds to a
remarkable improvement of approximately 82% in inference latency.
