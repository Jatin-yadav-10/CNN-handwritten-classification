CNN Handwritten Digit Classifier (MNIST)

A notebook that builds, trains, and evaluates a Convolutional Neural Network to classify handwritten digits (0–9) from the MNIST dataset, and tests the trained model on a custom external image.

What it does
Load data — loads the MNIST dataset via tf.keras.datasets.mnist (60,000 training images, 10,000 test images, 28x28 grayscale).
Inspect — visualizes a sample digit and prints raw pixel values (0–255) before preprocessing.
Normalize — scales pixel values using tf.keras.utils.normalize.
Reshape — adds a channel dimension so images become (N, 28, 28, 1), suitable for 2D convolutions.
Build the CNN — a Sequential model with:
3× Conv2D(64, (3,3)) + ReLU + MaxPooling2D(2,2) blocks
Flatten
Dense(64) + ReLU
Dense(32) + ReLU
Dense(10) + Softmax (output layer, one unit per digit class)
Train — compiles with sparse_categorical_crossentropy loss and the Adam optimizer, then fits for 5 epochs with a 30% validation split.
Evaluate — reports test loss and accuracy on the 10,000-sample MNIST test set.
Predict & visualize — shows predicted vs. actual labels for sample test images.
Custom image inference — reads an external image (img2.png) with OpenCV, converts to grayscale, inverts colors, resizes to 28x28, normalizes, and runs it through the trained model to predict the digit.
Requirements
tensorflow
numpy
matplotlib
opencv-python

Install with:

pip install tensorflow numpy matplotlib opencv-python
Input data
MNIST is downloaded automatically via tf.keras.datasets.mnist (no manual download needed).
For the custom-image step, place a digit image named img2.png in the same directory as the notebook. It should contain a single handwritten digit; the notebook handles grayscale conversion, color inversion, and resizing to 28x28.
Usage
Run all cells top to bottom in Jupyter.
Training takes a few minutes on CPU (or seconds on GPU) for 5 epochs.
Check the printed test accuracy/loss after evaluation.
To test your own digit, add img2.png next to the notebook before running the final cells.
Notes
The color inversion step (cv2.bitwise_not) assumes the custom image has a dark digit on a light background (like a photo of pen-on-paper), since MNIST digits are white-on-black.
Model architecture, epoch count, and layer sizes are hardcoded; adjust epochs, Conv2D filter counts, or Dense units to experiment with accuracy/training time tradeoffs.
