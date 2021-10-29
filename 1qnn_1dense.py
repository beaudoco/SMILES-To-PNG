import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import time
import datetime
import argparse
import csv

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json

import pennylane as qml
from pennylane.templates import RandomLayers
from sklearn.datasets import load_digits


# Argument
parser = argparse.ArgumentParser()
parser.add_argument("-c","--lrc", help="learning rate for classical neural network", type = float)
parser.add_argument("-q","--lrq",  help="learning rate for quantum neural network", type = float)
args = parser.parse_args()

lrc = args.lrc
if lrc is None:
    raise TypeError("Specify lrc value.")

lrq = args.lrq
if lrq is None:
    raise TypeError("Specify lrq value.")

#print(lrc)
#print(type(lrc))
#print(lrq)
#print(type(lrq))

x_samples = load_digits().data
y_labels = load_digits().target

train_samples = 1198
test_samples = x_samples.shape[0]-train_samples
x_train = x_samples[:train_samples]
x_test = x_samples[-test_samples:]

y_train = y_labels[:train_samples]
y_test = y_labels[-test_samples:]

n_features = x_train.shape[1]
latent_dim = int(math.log(n_features, 2))

n_qubits = latent_dim

###########################
#Initialization for cross-entropy
n_class = 10
y_train_cross=np.zeros(shape = [len(y_train), 10])

for i in range(len(y_train)):
    y_train_cross[i][y_train[i]] = 1.0

y_test_cross=np.zeros(shape = [len(y_test), 10])

for j in range(len(y_test)):
    y_test_cross[j][y_test[j]] = 1.0

###########################
dev = qml.device("default.qubit.tf", wires=n_qubits)

# Construct encoder Variational Quantum Circuit.
@qml.qnode(dev, interface='tf', diff_method='backprop')
def qnode_e(inputs, weights):
    qml.templates.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize = True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Construct decoder Variational Quantum Circuit.
#@qml.qnode(dev, interface='tf', diff_method='backprop')
#def qnode_d(inputs, weights):
#    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
#    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
#    return qml.probs(wires=[i for i in range(n_qubits)])

# Variational quantum weights in encoder and decoder.
weight_shapes_e = {"weights": (6, n_qubits, 3)} # 6 quantum layers and each qubit has 3 parameters per layer
#weight_shapes_d = {"weights": (6, n_qubits, 3)} # 6 quantum layers and each qubit has 3 parameters per layer
qlayer_e = qml.qnn.KerasLayer(qnode_e, weight_shapes_e, output_dim=n_features)
#qlayer_e = qml.qnn.KerasLayer(qnode_e, weight_shapes_e, output_dim=latent_dim)
#qlayer_d = qml.qnn.KerasLayer(qnode_d, weight_shapes_d, output_dim=n_features)

# Define the Quantum Autoencoder (QAE) class.
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.classical_optimizer = tf.keras.optimizers.Adam(lrc) # initial classical learning rate
        self.quantum_optimizer = tf.keras.optimizers.Adam(lrq) # initial quanutm learning rate

        self.vqc_e = tf.keras.Sequential([qlayer_e])
        self.cls_e = tf.keras.Sequential([layers.Dense(n_features)])
        self.classifier = tf.keras.layers.Dense(n_class, activation='softmax')

       # self.vqc_d = tf.keras.Sequential([qlayer_d])
       # self.cls_d = tf.keras.Sequential([layers.Dense(n_features)])
        
        
    def call(self, x):
        result = self.vqc_e(x)
        encoded = self.cls_e(result)
        result2 = self.classifier(encoded)
        return result2
     #   result = self.vqc_d(encoded)
     #   decoded = self.cls_d(result)
     #   return decoded

model = Autoencoder(latent_dim)


print('Start training...')
start_time = time.time()

BATCH_SIZE = 32
batches = len(x_train) // BATCH_SIZE

epochs = 30 #for debugging
#epochs = 30

# for getting csv file
train_loss_array=np.array([]) 
test_loss_array=np.array([])
model_acc_array2=np.array([])

for epoch in range(epochs):
    # for adjusting learning rates (optional)
    #if epoch == 10:
    #model.classical_optimizer = tf.keras.optimizers.Adam(lrc)
    #model.quantum_optimizer = tf.keras.optimizers.Adam(lrq)
    
    # Train the QAE model.
    sum_loss = 0
    for batch in range(batches):
        x = x_train[BATCH_SIZE * batch:min(BATCH_SIZE * (batch + 1), len(x_train))]
        y_train_true = y_train_cross[BATCH_SIZE * batch:min(BATCH_SIZE * (batch + 1), len(y_train))]
        
        with tf.GradientTape() as t1, tf.GradientTape() as t2:
            y_pred = model(x)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_train_true, y_pred))
            
            grad_vqc = t1.gradient(loss, model.vqc_e.trainable_variables)
            model.quantum_optimizer.apply_gradients(zip(grad_vqc, model.vqc_e.trainable_variables))
            
            grad_cls = t2.gradient(loss, model.cls_e.trainable_variables)
            model.classical_optimizer.apply_gradients(zip(grad_cls, model.cls_e.trainable_variables))
        
        sum_loss += loss
        print('Batch {}/{} Loss {:.4f}'.format(batch, batches, loss), end='\r')

    avg_loss = sum_loss/batches

    # Run test samples.
    with tf.GradientTape() as t1, tf.GradientTape() as t2:
        y_pred = model(x_test)

        test_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_test_cross, y_pred))
        
        model_acc_array = tf.keras.metrics.categorical_accuracy(y_test_cross, y_pred.numpy())
        model_acc = tf.reduce_sum(model_acc_array).numpy()/len(y_test_cross)

    
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print('Elapsed {}\t Epoch {}/{} [Train Loss: {:.4f}]\t [Test Loss: {:.4f}] \t [Accuracy: {:.4f}]'.format(et, epoch+1, epochs, avg_loss.numpy(), test_loss.numpy(), model_acc))

    train_loss_array = np.append(train_loss_array, avg_loss.numpy())
    test_loss_array = np.append(test_loss_array, test_loss.numpy())
    model_acc_array2 = np.append(model_acc_array2, model_acc)

lrc_str = str(lrc)
lrq_str = str(lrq)




model.save_weights("lrc_"+lrc_str+"lrq_"+lrq_str+".h5")
with open("lrc_"+lrc_str+"lrq_"+lrq_str+".csv", "w") as csvfile:
    fieldnames = ["Epoch", "Train Loss", "Test Loss", "Accuracy"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(epochs):

        writer.writerow({"Epoch": i+1, "Train Loss": train_loss_array[i], "Test Loss":test_loss_array[i], "Accuracy":model_acc_array2[i]})
