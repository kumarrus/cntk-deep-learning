# Import the relevant components
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import cntk

%matplotlib inline

# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        cntk.device.try_set_default_device(cntk.device.cpu())
    else:
        cntk.device.try_set_default_device(cntk.device.gpu(0))

# Test for CNTK version
if not cntk.__version__ == '2.6':
    raise Exception("this lab is designed to work with 2.0. Current Version: " + cntk.__version__)

# Ensure we always get the same amount of randomness
np.random.seed(0)
cntk.cntk_py.set_fixed_random_seed(1)
cntk.cntk_py.force_deterministic_algorithms()

# Define the data dimensions
input_dim = 784
num_output_classes = 10

# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):

    labelStream = cntk.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False)
    featureStream = cntk.io.StreamDef(field='features', shape=input_dim, is_sparse=False)

    deserializer = cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(labels = labelStream, features = featureStream))

    return cntk.io.MinibatchSource(deserializer,
        randomize = is_training, max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)

# Ensure the training and test data is generated and available for this lab.
# We search in two locations in the toolkit for the cached MNIST data set.
data_found = False

for data_dir in [os.path.join("..", "Examples", "Image", "DataSets", "MNIST"), os.path.join("data", "MNIST")]:
    train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
    test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
    if os.path.isfile(train_file) and os.path.isfile(test_file):
        data_found = True
        break

if not data_found:
    raise ValueError("Please generate the data by completing Lab1_MNIST_DataLoader")

print("Data directory is {0}".format(data_dir))

# Create the input variables
input = cntk.input_variable(input_dim)
label = cntk.input_variable(num_output_classes)

# We add layers to the model to create a deep neural net
# Layer 1: { Input: 784, Output:400, Activation: RELU }
# Layer 2: { Input: 400, Output:400, Activation: RELU }
# Layer 3: { Input: 400, Output:10, Activation: None }

# Define the layer dimensions
num_hidden_layers = 2
hidden_layers_dim = 400

def create_model(features):
    with cntk.layers.default_options(init = cntk.glorot_uniform(), activation = cntk.ops.relu):
        input = features
        for _ in range(num_hidden_layers):
            input = cntk.layers.Dense(hidden_layers_dim)(input)
        r = cntk.layers.Dense(num_output_classes, activation = None)(input)
        return r

# Scale the input to 0-1 range by dividing each pixel by 255.
input_s_normalized = input/255.0
input_s_squared = cntk.square(input_s_normalized)
input_s_sqrt = cntk.sqrt(input_s_normalized)
z_model = create_model(input_s_normalized)

# Define the loss function for is_training
loss = cntk.cross_entropy_with_softmax(z_model, label)

# Classification error evaluation
label_error = cntk.classification_error(z_model, label)

# Configure training parameters
# Instantiate the trainer object to drive the model training
learning_rate = 0.2
lr_schedule = cntk.learning_rate_schedule(learning_rate, cntk.UnitType.minibatch)

# Schoastic Gradient Descent learner
learner = cntk.sgd(z_model.parameters, lr_schedule)
trainer = cntk.Trainer(z_model, (loss, label_error), [learner])

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

# Defines a utility that prints the training progress
def print_training_progress(trainer, minibatch, frequency, verbose = 1):
    training_loss = "NA"
    eval_error = "NA"

    if minibatch%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(minibatch, training_loss, eval_error*100))

    return minibatch, training_loss, eval_error

######
# We are now ready to run the trainer
# Initialize the parameters for the trainer
minibatch_size = 64
num_samples_per_sweep = 60000
num_sweeps_to_train_with = 10 # We train with 10 sweeps of the input samples
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

# Create the reader to training data set
reader_train = create_reader(train_file, True, input_dim, num_output_classes)

# Map the data streams to the input and labels.
input_map = {
    label : reader_train.streams.labels,
    input : reader_train.streams.features
}

# Run the trainer on and perform model training
training_progress_output_freq = 500

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_minibatches_to_train)):
    # Read a mini batch from the training data file
    data = reader_train.next_minibatch(minibatch_size, input_map = input_map)

    trainer.train_minibatch(data)
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)

# Compute the moving average loss to smooth out the noise in SGD
plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

# Plot the training loss and the training error
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], "b--")
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')
plt.show()

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()

# Read the training data
reader_test = create_reader(test_file, False, input_dim, num_output_classes)

test_input_map = {
    label  : reader_test.streams.labels,
    input  : reader_test.streams.features,
}

# Test data for trained model
test_minibatch_size = 512
num_samples = 10000
num_minibatches_to_test = num_samples // test_minibatch_size # // is floor division. It drops the decimal.
test_result = 0.0

for i in range(num_minibatches_to_test):

    # We are loading test data in batches specified by test_minibatch_size
    # Each data point in the minibatch is a MNIST digit image of 784 dimensions
    # with one pixel per dimension that we will encode / decode with the
    # trained model.
    data = reader_test.next_minibatch(test_minibatch_size,
                                      input_map = test_input_map)

    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))

out_softmax_func = cntk.softmax(z_model)

# Read the data for evaluation
reader_eval = create_reader(test_file, False, input_dim, num_output_classes)

eval_minibatch_size = 25
eval_input_map = {input: reader_eval.streams.features}

data = reader_test.next_minibatch(eval_minibatch_size, input_map = test_input_map)

img_label = data[label].asarray()
img_data = data[input].asarray()

# Calling softmax.eval will run the image data through our model and give the output of the softmax layer
predicted_label_prob = [out_softmax_func.eval(img_data[i]) for i in range(len(img_data))]

# Find the index with the maximum value for both predicted as well as the ground truth
predictedLabel = [np.argmax(predicted_label_prob[i]) for i in range(len(predicted_label_prob))]
actualLabel = [np.argmax(img_label[i]) for i in range(len(img_label))]

print("Label    :", actualLabel[:eval_minibatch_size])
print("Predicted:", predictedLabel)

# Predict mystery image
import imageio as iio
mystery_image = iio.imread("MysteryNumberD.bmp").flatten()
mystery_image_conv = mystery_image.astype(np.float32)

# Calling softmax.eval will run the image data through our model and give the output of the softmax layer
predicted_mystery_img_label = np.argmax(out_softmax_func.eval(mystery_image_conv))

print("Predicted Mystery Image:", predicted_mystery_img_label)
