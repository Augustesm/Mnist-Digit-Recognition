# MNIST Digit Recognition

**MNIST Digit Recognition** is a deep learning project built using **TensorFlow** and **Keras**. The goal of the assignment was to load the MNIST handwritten digit dataset, train a fully connected neural network, analyze model performance, visualize training curves, and test the trained model on a custom handwritten digit image. Multiple experiments were performed by adjusting hyperparameters such as the number of layers, number of neurons, batch size, validation split, and epoch count to find the best-performing model.

## Key Features

* 🔤 **Loading and preprocessing the MNIST dataset**
* 📊 **Visualization of sample training images**
* 🧠 **Fully connected deep neural network implementation**
* 🌀 **Activation functions: ReLU (hidden), Softmax (output)**
* 🔄 **Training with categorical crossentropy loss and accuracy metrics**
* 📈 **Visualization of accuracy and loss curves**
* 🔍 **Hyperparameter testing (layers, neurons, batch size, validation size)**
* 🖼️ **Prediction on custom handwritten digit images**
* 🧪 **6–8 experimental model training runs with comparison table**

## Technologies Used

* **TensorFlow / Keras** — neural network creation, training, and evaluation
* **NumPy** — numerical operations and preprocessing
* **Matplotlib** — accuracy & loss curve visualization, dataset image preview
* **Pillow (PIL)** — loading and converting external handwritten digit images
* **Python** — core language for all implementation

## Features in Detail

### Dataset Handling

* Loaded MNIST dataset:

  * **60,000 images for training**
  * **10,000 images for testing**
* Normalized pixel values to the [0, 1] range
* Converted labels to **one-hot vectors** using `keras.utils.to_categorical`
* Visualized 15 training images to explore the dataset

### Neural Network Architecture

* **Input layer**: 28×28 grayscale image
* **Flatten layer**: converts image to 784-value vector
* **Hidden layer**:

  * Dense(100), activation = ReLU
* **Output layer**:

  * Dense(10), activation = Softmax

### Model Training

* Optimizer: **RMSprop**
* Loss: **categorical_crossentropy**
* Metrics: **accuracy**
* Performed training experiments by changing:

  * Number of layers
  * Number of neurons
  * Batch size (mini-batch)
  * Validation split (10%, 20%, 30%)
  * Number of epochs
* Recorded:

  * Training accuracy
  * Validation accuracy
  * Training loss
  * Validation loss
  * Whether the model correctly classified the handwritten test image

### Training Visualization

* Plotted:

  * Accuracy vs. epochs (train & validation)
  * Loss vs. epochs (train & validation)
* Used graphs to compare model performance across experiments

### Custom Digit Prediction

* Loaded external 28×28 grayscale PNG image
* Converted with **PIL** and normalized
* Passed through the trained model
* Displayed predicted digit using `np.argmax()`
* Visualized the custom image
