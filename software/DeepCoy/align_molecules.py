from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

# Iterative MCS alignment. Then when MCS too small, align atoms by atom types (since training set should have a near perfect match)
def align_smiles_by_MCS_it(smiles_1, smiles_2):
    try:
        mols = [Chem.MolFromSmiles(smiles_1), Chem.MolFromSmiles(smiles_2)]

        MCS_res = []
        aligned_mols = list(mols) # Deep copy
        partial_mols = list(mols) # Deep copy
        repeat = True
        count = 0
        total_atoms_aligned = 0
        while repeat:
            count +=1
            # Align mols by MCS
            res=rdFMCS.FindMCS(partial_mols)#, ringMatchesRingOnly=True)
            MCS_res.append(res)
    
            for i, mol in enumerate(aligned_mols):
                sub_idx = list(mol.GetSubstructMatches(Chem.MolFromSmarts(res.smartsString)))
                # Check match not with already aligned atoms
                if len(sub_idx) > 1:
                    for match in sub_idx:
                        if min(match) < total_atoms_aligned:
                            continue
                        else:
                            sub_idx = list(match)
                elif len(sub_idx) == 1:
                    sub_idx = list(sub_idx[0])
                size_MCS = len(sub_idx)
                # Align mols
                mol_range = list(range(mol.GetNumHeavyAtoms()))
                idx_to_add = list(set(mol_range).difference(set(sub_idx)))
                sub_idx.extend(idx_to_add)
                aligned_mols[i] = Chem.rdmolops.RenumberAtoms(mol, sub_idx)
                # Update partial mol
                partial_mols[i] = Chem.rdmolops.DeleteSubstructs(partial_mols[i], Chem.MolFromSmarts(res.smartsString))
    
            # Update number of atoms aligned
            total_atoms_aligned += size_MCS
    
            # Stop criterion
            if size_MCS < 3:
                repeat = False
            if count > 2:
                repeat = False
        
        # Align mols by atom type
        sub_idx = []
        for i in range(total_atoms_aligned, mols[0].GetNumHeavyAtoms()):
            # Get atom type of mol1
            atom_type = aligned_mols[0].GetAtomWithIdx(i).GetAtomicNum()
            # Warning if not an atom type we know is matched perfectly
            #if atom_type not in [6, 7, 8, 9, 16, 17]:
            #    print("Warning: out of guaranteed scope atom type.")
            # Find same atom type on mol2
            for j in range(total_atoms_aligned, mols[0].GetNumHeavyAtoms()):
                if aligned_mols[1].GetAtomWithIdx(j).GetAtomicNum() == atom_type and j not in sub_idx:
                    sub_idx.append(j)
                    break
        # Perform alignment
        align_idx = list(range(total_atoms_aligned))
        align_idx.extend(sub_idx)
        align_idx.extend(list(set(range(mols[0].GetNumHeavyAtoms())).difference(set(align_idx))))
        aligned_mols[1] = Chem.rdmolops.RenumberAtoms(aligned_mols[1], align_idx)
        return aligned_mols, MCS_res, []
    
    except:
        return [[], []], [], []
