from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(len(train_labels), len(test_labels))
print(train_images.shape, test_images.shape)
print(train_labels, test_labels)

# building the model

from keras import layers, models

network = models.Sequential()
network.add(layer=layers.Dense(units=512, activation="relu", input_shape=(28 * 28,)))
network.add(layer=layers.Dense(units=10, activation="softmax"))

# compilation

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# preparing data

train_images = train_images.reshape((60_000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10_000, 28 * 28))
test_images = test_images.astype('float32') / 255

print(train_images.shape, test_images.shape)
#print(train_images, test_images)

# categorically encode the labels

from keras.utils import to_categorical
train_labels = to_categorical(y=train_labels)
test_labels=to_categorical(y=test_labels)

print(train_labels.shape, test_labels.shape)

network.fit(x=train_images, y=train_labels, batch_size=128, epochs=5)

test_loss, test_accuracy=network.evaluate(x=test_images, y=test_labels)

print(test_loss, test_accuracy)