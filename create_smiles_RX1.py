from openbabel import pybel
"""
smiles = open('USPTO-50K/tgt-test.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "")
smiles.close()
mols = [pybel.readstring("smi", x) for x in chunks]
for idx in range(len(mols)):
    mols[idx].draw(False, "USPTO-50K-IMAGES-TGT/mol-{0}.png".format(idx))
# fps = [x.calcfp() for x in mols]
# print(fps[0].bits, fps[1].bits) 
# print(fps[0] | fps[1])
"""
#get list all of our first type of reactions
smiles_src_train = open('USPTO-50K/src-train.txt', 'r')
content_src_train = smiles_src_train.read()
chunks_src_train = content_src_train.split('\n')
chunks_src_train.remove('')
chunks_src_train2 = content_src_train.split('\n')
chunks_src_train2.remove('')
image_src_train_list = []
idx_src_train_arr = []
for idx in range(len(chunks_src_train)):
	chunks_src_train2[idx] = chunks_src_train2[idx].replace(" ", "")
	chunks_src_train[idx] = chunks_src_train[idx].replace(" ", "").split('>',1)[1]
mols = [pybel.readstring("smi", x) for x in chunks_src_train]
smiles_src_train.close()
for idx in range(len(mols)):
	if(chunks_src_train2[idx].split('>',1)[0].replace("<RX_","") == "1"):
		idx_src_train_arr.append(idx)
		mols[idx].draw(False, "USPTO-50K-IMAGES-SRC/mol-{0}.png".format(idx))
#		image_src_train_list.append(mols[idx].draw(show = False, filename = png))

#get the matching reactant images
image_tgt_train_list = []
smiles_tgt_train = open('USPTO-50K/tgt-train.txt', 'r')
content_tgt_train = smiles_tgt_train.read()
chunks_tgt_train = content_tgt_train.split('\n')
chunks_tgt_train.remove('')
for idx in range(len(chunks_tgt_train)):
    chunks_tgt_train[idx] = chunks_tgt_train[idx].replace(" ", "")
smiles_tgt_train.close()
mols = [pybel.readstring("smi", x) for x in chunks_tgt_train]
for idx in range(len(mols)):
	for i in idx_src_train_arr:
		if(idx == i):
			mols[idx].draw(False, "USPTO-50K-IMAGES-TGT/mol-{0}.png".format(idx))


#image_tgt_train_list.append(mols[idx].draw(show = False, filename = png))


