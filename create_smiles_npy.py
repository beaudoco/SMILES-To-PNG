from openbabel import pybel
from numpy import asarray
import numpy as np
import os
import cv2
import glob
from tempfile import TemporaryFile

smiles = open('USPTO-50K/tgt-train.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "")
smiles.close()
mols = [pybel.readstring("smi", x) for x in chunks]
for idx in range(len(mols)):
    mols[idx].draw(False, "USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.png".format(idx))

smiles = open('USPTO-50K/src-train.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "").split('>',1)[1]
smiles.close()
mols = [pybel.readstring("smi", x) for x in chunks]
for idx in range(len(mols)):
    mols[idx].draw(False, "USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx))

smiles_src_train = open('USPTO-50K/src-train.txt', 'r')
content_src_train = smiles_src_train.read()
chunks_src_train = content_src_train.split('\n')
chunks_src_train.remove('')
idx_src_train_arr = []
for idx in range(len(chunks_src_train)):
	chunks_src_train[idx] = chunks_src_train[idx].replace(" ", "").split('>',1)[0].replace("<RX_","")
	if(chunks_src_train[idx] == "1"):
		idx_src_train_arr.append(idx)
smiles_src_train.close()

#get the list of images from our first type of reactions
image_src_train_list = []
for filename in glob.glob('USPTO-50K-IMAGES-SRC-TRAIN/*'):
    print(filename)
    for idx in idx_src_train_arr:
        print("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx))
        if(filename == "USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx)):
            img = cv2.imread(filename)
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(grey_img, (28, 28) , interpolation= cv2.INTER_AREA)
            flatten = resized.flatten()
            image_src_train_list.append(flatten)
            np.save("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.npy".format(idx), asarray(flatten))
            print("shrunk {0}".format(idx))
            # f = open("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.npy".format(idx), "w")
            # f.write(asarray(flatten))
            # f.close()
            # os.remove("USPTO-50K-IMAGES-SRC-TRAIN/mol-{0}.png".format(idx))


#get the matching reactant images
image_tgt_train_list = []
for filename in glob.glob('USPTO-50K-IMAGES-TGT-TRAIN/*'):
    for idx in idx_src_train_arr:
        if(filename == "USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.png".format(idx)):
            img = cv2.imread(filename)
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(grey_img, (28, 28) , interpolation= cv2.INTER_AREA)
            flatten = resized.flatten()
            image_tgt_train_list.append(flatten)
            np.save("USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.npy".format(idx), asarray(flatten))
            print("shrunk {0}".format(idx))
            # f = open("USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.npy".format(idx), "w")
            # f.write(asarray(flatten))
            # f.close()
            # os.remove("USPTO-50K-IMAGES-TGT-TRAIN/mol-{0}.png".format(idx))


# fps = [x.calcfp() for x in mols]
# print(fps[0].bits, fps[1].bits) 
# print(fps[0] | fps[1])