import numpy as np
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit=train_images[1]

# displaying the digit:
import matplotlib.pyplot as plt
plt.imshow(X=digit, cmap=plt.cm.binary)
plt.show()