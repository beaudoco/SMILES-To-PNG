# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import tensorflow as tf
import math
import time
import datetime
import argparse
import csv
# from PIL import Image
# import glob

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
# from tensorflow.keras.models import model_from_json

import pennylane as qml
# from pennylane.templates import RandomLayers
# from sklearn.datasets import load_digits
from openbabel import pybel
# import cv2
from pysmiles import read_smiles

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

#get list all of our first type of reactions
"""
smiles_src_test = open('USPTO-50K/src-test.txt', 'r')
content_src_test = smiles_src_test.read()
chunks_src_test = content_src_test.split('\n')
chunks_src_test.remove('')
chunks_src_test2 = content_src_test.split('\n')
chunks_src_test2.remove('')
image_src_test_list = []
idx_src_test_arr = []
for idx in range(len(chunks_src_test)):
	chunks_src_test2[idx] = chunks_src_test2[idx].replace(" ", "")
	chunks_src_test[idx] = chunks_src_test[idx].replace(" ", "").split('>',1)[1]
mols = [pybel.readstring("smi", x) for x in chunks_src_test]
smiles_src_test.close()
for idx in range(len(mols)):
	if(chunks_src_test2[idx].split('>',1)[0].replace("<RX_","") == "1"):
		idx_src_test_arr.append(idx)
		image_src_test_list.append(mols[idx].draw(False))

#get the matching reactant images
image_tgt_test_list = []
smiles_tgt_test = open('USPTO-50K/tgt-test.txt', 'r')
content_tgt_test = smiles_tgt_test.read()
chunks_tgt_test = content_tgt_test.split('\n')
chunks_tgt_test.remove('')
for idx in range(len(chunks_tgt_test)):
    chunks_tgt_test[idx] = chunks_tgt_test[idx].replace(" ", "")
smiles_tgt_test.close()
mols = [pybel.readstring("smi", x) for x in chunks_tgt_test]
for idx in range(len(mols)):
	for i in idx_src_test_arr:
		if(idx == i):
			image_tgt_test_list.append(mols[idx].draw(False))
"""
#get list all of our first type of reactions
smiles_src_train = open('USPTO-50K/src-train.txt', 'r')
content_src_train = smiles_src_train.read()
chunks_src_train = content_src_train.split('\n')
chunks_src_train.remove('')
chunks_src_train2 = content_src_train.split('\n')
chunks_src_train2.remove('')
graph_src_train_list = []
idx_src_train_arr = []
for idx in range(len(chunks_src_train)):
	chunks_src_train2[idx] = chunks_src_train2[idx].replace(" ", "")
	chunks_src_train[idx] = chunks_src_train[idx].replace(" ", "").split('>',1)[1]
# mols = [pybel.readstring("smi", x) for x in chunks_src_train]
smiles_src_train.close()
for idx in range(len(chunks_src_train)):
	if(chunks_src_train2[idx].split('>',1)[0].replace("<RX_","") == "1"):
		if(chunks_src_train[idx].find("[se]") == -1):
			idx_src_train_arr.append(idx)
			graph_src_train_list.append(read_smiles(chunks_src_train[idx]))
			# print(chunks_src_train[idx])
        # print(read_smiles(chunks_src_train[idx]))

print(graph_src_train_list[0])

#get the matching reactant images
graph_tgt_train_list = []
smiles_tgt_train = open('USPTO-50K/tgt-train.txt', 'r')
content_tgt_train = smiles_tgt_train.read()
chunks_tgt_train = content_tgt_train.split('\n')
chunks_tgt_train.remove('')
for idx in range(len(chunks_tgt_train)):
    chunks_tgt_train[idx] = chunks_tgt_train[idx].replace(" ", "")
smiles_tgt_train.close()
# mols = [pybel.readstring("smi", x) for x in chunks_tgt_train]
for idx in range(len(chunks_tgt_train)):
	for i in idx_src_train_arr:
		if(idx == i):
			# img = cv2.imread('USPTO-50K-IMAGES-TGT/mol-{0}.png', idx)
			# grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# flatten = grey_img.flatten()
			graph_tgt_train_list.append(read_smiles(chunks_tgt_train[idx]))




"""
image_src_train_list=[]
image_tgt_train_list=[]
#image = cv2.imread('USPTO-50K-IMAGES-SRC/mol-{0}.png',1)
for i in range(50000):
	img = cv2.imread('USPTO-50K-IMAGES-SRC/mol-{0}.png', i)
	if hasattr(img, 'attr_name'):	
		grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		flatten = grey_img.flatten()
		image_src_train_list.append(flatten)

for i in range(50000):
	img = cv2.imread('USPTO-50K-IMAGES-TGT/mol-{0}.png', i)
	if hasattr(img, 'attr_name'):
		grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		flatten = grey_img.flatten()
		image_tgt_train_list.append(flatten)

print('length: ',len(image_src_train_list))

"""
#image = cv2.imread('USPTO-50K-IMAGES-SRC/mol-1.png')
#print(type(image))

#x_samples = load_digits().data
#y_labels = load_digits().target


train_samples = (2*len(graph_src_train_list))//3
test_samples = len(graph_src_train_list)-train_samples
x_train = graph_src_train_list[:train_samples]
x_test = graph_src_train_list[-test_samples:]
print("####################Debugging###################")
print(len(x_train))
print("####################Debugging###################")

y_train_samples = (2*len(graph_tgt_train_list))//3
y_test_samples = len(graph_tgt_train_list) - y_train_samples
y_train = graph_tgt_train_list[:y_train_samples]
y_test = graph_tgt_train_list[-y_test_samples:]

	
#x_train = np.array(x_train)
#y_train = np.array(y_train)
#x_test = np.array(x_test)
#y_test = np.array(y_test)


print("####################Debugging###################")

print('x_train_shape:',x_train.shape)
print('x_train[0]shape : ', x_train[0].shape)
print("####################Debugging###################")

n_features = x_train.shape[1]
print('features: ', x_train.shape[1])
latent_dim = int(math.log(n_features, 2))

n_qubits = latent_dim

###########################
#Initialization for cross-entropy
#n_class = 10
#y_train_cross=np.zeros(shape = [len(y_train), 10])

#for i in range(len(y_train)):
#    y_train_cross[i][y_train[i]] = 1.0

#y_test_cross=np.zeros(shape = [len(y_test), 10])

#for j in range(len(y_test)):
#    y_test_cross[j][y_test[j]] = 1.0

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
def qnode_d(inputs, weights):
	qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
	qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
	return qml.probs(wires=[i for i in range(n_qubits)])

# Variational quantum weights in encoder and decoder.
weight_shapes_e = {"weights": (6, n_qubits, 3)} # 6 quantum layers and each qubit has 3 parameters per layer
weight_shapes_d = {"weights": (6, n_qubits, 3)} # 6 quantum layers and each qubit has 3 parameters per layer
qlayer_e = qml.qnn.KerasLayer(qnode_e, weight_shapes_e, output_dim=n_features)
#qlayer_e = qml.qnn.KerasLayer(qnode_e, weight_shapes_e, output_dim=latent_dim)
qlayer_d = qml.qnn.KerasLayer(qnode_d, weight_shapes_d, output_dim=n_features)

# Define the Quantum Autoencoder (QAE) class.

class Autoencoder(Model):
	def __init__(self, latent_dim):
		super(Autoencoder, self).__init__()
		self.latent_dim = latent_dim
		self.classical_optimizer = tf.keras.optimizers.Adam(lrc) # initial classical learning rate
		self.quantum_optimizer = tf.keras.optimizers.Adam(lrq) # initial quanutm learning rate
		self.vqc_e = tf.keras.Sequential([qlayer_e])
		self.cls_e = tf.keras.Sequential([layers.Dense(n_features)])
		#self.classifier = tf.keras.layers.Dense(n_class, activation='softmax')
		self.vqc_d = tf.keras.Sequential([qlayer_d])
		self.cls_d = tf.keras.Sequential([layers.Dense(n_features)])
	
	def call(self, x):

		encoded = self.vqc_e(x)
		result = self.vqc_d(encoded)
		decoded = self.cls_d(result)
		return decoded


model = Autoencoder(latent_dim)


print('Start training...')
start_time = time.time()

BATCH_SIZE = 128
batches = len(x_train) // BATCH_SIZE

epochs = 1 #for debugging
#epochs = 30

# for getting csv file
train_loss_array=np.array([]) 
test_loss_array=np.array([])
#model_acc_array2=np.array([])

for epoch in range(epochs):
    # for adjusting learning rates (optional)
    #if epoch == 10:
    #model.classical_optimizer = tf.keras.optimizers.Adam(lrc)
    #model.quantum_optimizer = tf.keras.optimizers.Adam(lrq)
    
    # Train the QAE model.
	sum_loss = 0
	for batch in range(batches):
		x = x_train[BATCH_SIZE*batch:min(BATCH_SIZE*(batch + 1), len(x_train))]
		y = y_train[BATCH_SIZE*batch:min(BATCH_SIZE*(batch + 1), len(y_train))]
        
		with tf.GradientTape() as t1:
			y_pred = model(x)
			loss = tf.reduce_mean(tf.square(y_pred - y))
			grad_vqc = t1.gradient(loss, model.vqc_e.trainable_variables + \
										model.vqc_d.trainable_variables)
			model.quantum_optimizer.apply_gradients(zip(grad_vqc, model.vqc_e.trainable_variables + \
																	model.vqc_d.trainable_variables))
            
			grad_cls = t1.gradient(loss, model.cls_d.trainable_variables)
			model.classical_optimizer.apply_gradients(zip(grad_cls, model.cls_d.trainable_variables))
		sum_loss += loss
		print('Batch {}/{} Loss {:.4f}'.format(batch, batches, loss), end='\r')
	avg_loss = sum_loss/batches

    # Run test samples.
	with tf.GradientTape() as t1:
		y_pred = model(x_test)
		test_loss = tf.reduce_mean(tf.square(y_pred - y_test))
    
	
	et = time.time() - start_time
	et = str(datetime.timedelta(seconds=et))[:-7]
	print('Elapsed {}\t Epoch {}/{} [Train Loss: {:.4f}]\t [Test Loss: {:.4f}]'.format(et, epoch+1, epochs, avg_loss.numpy(), test_loss.numpy()))
	
	train_loss_array = np.append(train_loss_array, avg_loss.numpy())
	test_loss_array = np.append(test_loss_array, test_loss.numpy())

lrq_str = str(lrq)




model.save_weights("lrq_"+lrq_str+".h5")
with open("lrq_"+lrq_str+".csv", "w") as csvfile:
	fieldnames = ["Epoch", "Train Loss", "Test Loss"]
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for i in range(epochs):
		writer.writerow({"Epoch": i+1, "Train Loss": train_loss_array[i], "Test Loss":test_loss_array[i]})
