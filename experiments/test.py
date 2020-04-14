import logging
import numpy as np
import tensorflow as tf
from checkmate.tf2 import get_keras_model
from tqdm import tqdm
import time
import skimage.transform
from tensorflow.keras.utils import multi_gpu_model

# tf.compat.v1.disable_eager_execution()


# load cifar10 dataset
# load cifar10 dataset
batch_size = 128
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#x_test_new = np.empty(1)
#x_train, x_test = x_train / 255.0, x_test / 255.0
one_size = 10000
#x_train_new = np.empty((one_size, 244, 244, 3))
x_train_new = np.empty((one_size, 224, 224, 3))
#print(x_train_new[0].shape)
#print(x_train_new.shape)
for num, image in enumerate(x_train[:one_size]):
  newImage = skimage.transform.resize(image, (224, 224), mode='constant')
  x_train_new[num] = newImage
  #x_train_new = np.concatenate((x_train_new, newImage), axis = 0)
  #print(x_train_new.shape)
y_train = y_train[:one_size]
x_train_new, y_train = x_train_new.astype(np.float32), y_train.astype(np.float32)
x_test, y_test = x_test.astype(np.float32), y_test.astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train_new, y_train)).batch(batch_size)
#test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# load TensorFlow model from Keras applications along with loss function and optimizer
model = get_keras_model("VGG16", input_shape=x_train_new[0].shape, num_classes=10)
#model = simple_model()
#model = linear_model(1)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss)

from checkmate.tf2.wrapper import compile_tf2
start = time.perf_counter()
element_spec = train_ds.__iter__().__next__()
train_iteration = compile_tf2(
    model,
    loss=loss,
    optimizer=optimizer,
    input_spec=element_spec[0], #retrieve first element of dataset
    label_spec=element_spec[1]
)
end = time.perf_counter() - start
print("Checkmate compile time: {0}".format(end))

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

total_time = 0
for epoch in range(2):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    start_time = time.perf_counter()
    with tqdm(total=one_size) as pbar:
        for images, labels in train_ds:
            predictions, loss_value = train_iteration(images, labels)
            train_loss(loss_value)
            train_accuracy(labels, predictions)
            pbar.update(images.shape[0])
            pbar.set_description('Train epoch {}; loss={:0.4f}, acc={:0.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))

#    with tqdm(total=x_test.shape[0]) as pbar:
#        for images, labels in test_ds:
#            predictions = model(images)
#            test_loss_value = loss(labels, predictions)
#            test_loss(test_loss_value)
#            test_accuracy(labels, predictions)
#            pbar.update(images.shape[0])
#            pbar.set_description('Valid epoch {}, loss={:0.4f}, acc={:0.4f}'.format(epoch + 1, test_loss.result(), test_accuracy.result()))
    exec_time = time.perf_counter() - start_time
    total_time += exec_time
    print("execution time: {0}".format(exec_time))
print("Total Execution time: {0}".format(total_time))
