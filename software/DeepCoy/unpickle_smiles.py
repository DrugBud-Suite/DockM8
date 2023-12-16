import pickle

file_name = "generated_smiles_zinc"

with open(file_name, 'rb') as in_file:
    smiles = pickle.load(in_file)

print("Number of generated SMILES: %d" % len(smiles))

with open(file_name+".smi", 'w') as out_file:
    for line in smiles:
        out_file.write(line + '\n')
