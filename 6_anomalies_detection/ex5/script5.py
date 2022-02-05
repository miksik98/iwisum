"""
Based on: https://blog.keras.io/building-autoencoders-in-keras.html
"""

import keras
from keras import layers

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = keras.Input(shape=(784,))
# "encoded" is the encoded representation of the input
# aka "latent" layer
# aka "hidden" layer
# aka "code"
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# Let's also create a separate encoder model:

# this model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

# As well as the decoder model:

# create a placeholder for an encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

# Now let 's train our autoencoder to reconstruct MNIST digits.
#
# First, we 'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Let 's prepare our input data. We're using MNIST digits, and we're discarding the labels
# (since we're only interested in encoding / decoding the input images).

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

# We will normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784.

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Now let's train our autoencoder for 50 epochs:
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# After 50 epochs, the autoencoder seems to reach a stable train / test loss value of about 0.11.
# We can try to visualize the reconstructed inputs and the encoded representations.We will use Matplotlib.

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# -----------------------------------------------------------------------------
# Anomaly Detection starts here
# -----------------------------------------------------------------------------
print("\nANOMALY DETECTION using Fashion-MNIST dataset")

from keras.datasets import fashion_mnist

(_, _), (x_test_fashion, _) = fashion_mnist.load_data()

x_test_fashion = x_test_fashion.astype('float32') / 255.
x_test_fashion = x_test_fashion.reshape((len(x_test_fashion), np.prod(x_test_fashion.shape[1:])))

encoded_imgs_fashion = encoder.predict(x_test_fashion)
decoded_imgs_fashion = decoder.predict(encoded_imgs_fashion)

# collect reconstruction errors
from ex5.solution5 import reconstruction_errors, detect, calc_threshold
from utils import print_results

reconstruction_errors_mnist = reconstruction_errors(x_test, decoded_imgs)
reconstruction_errors_fashion = reconstruction_errors(x_test_fashion, decoded_imgs_fashion)

labels = np.hstack((
    np.zeros_like(reconstruction_errors_mnist),
    np.ones_like(reconstruction_errors_fashion)
))

reconstruction_errors_all = np.hstack((reconstruction_errors_mnist,
                                       reconstruction_errors_fashion))

threshold = calc_threshold(reconstruction_errors_mnist)
predictions = detect(reconstruction_errors_all, threshold)
print_results(labels, predictions)

import matplotlib.pyplot as plt

n = 10  # how many digits we will display
n += 4  # additional place for reconstruction error histograms
plt.figure(figsize=(22, 8))
for i in range(n - 1):
    # display original mnist
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction mnist
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original fashion
    ax = plt.subplot(4, n, i + 1 + n * 2)
    plt.imshow(x_test_fashion[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction fashion
    ax = plt.subplot(4, n, i + 1 + n * 3)
    plt.imshow(decoded_imgs_fashion[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# plot error histograms
hist_bins = np.linspace(0, .5, 100)
ax = plt.subplot2grid((4, n), (0, n - 2),
                      colspan=4,
                      rowspan=2)
ax.hist(reconstruction_errors_mnist,
        density=True,
        histtype="step",
        bins=hist_bins)
ax = plt.subplot2grid((4, n), (2, n - 2),
                      colspan=4,
                      rowspan=2)
ax.hist(reconstruction_errors_fashion,
        density=True,
        histtype="step",
        bins=hist_bins)
plt.tight_layout()
plt.show()
