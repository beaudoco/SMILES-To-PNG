from openbabel import pybel
smiles = open('USPTO-50K/tgt-test.txt', 'r')
content = smiles.read()
chunks = content.split('\n')
chunks.remove('')
for idx in range(len(chunks)):
    chunks[idx] = chunks[idx].replace(" ", "")
smiles.close()
mols = [pybel.readstring("smi", x) for x in chunks]
for idx in range(len(mols)):
    mols[idx].draw(False, "USPTO-50K-IMAGES/mol-{0}.png".format(idx))
# fps = [x.calcfp() for x in mols]
# print(fps[0].bits, fps[1].bits) 
# print(fps[0] | fps[1])