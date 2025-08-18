from rdkit import Chem
from rdkit.Chem import Descriptors3D

m = Chem.MolFromSmiles("c1ccccc1")
m = Chem.AddHs(m)
from rdkit.Chem import AllChem
AllChem.EmbedMolecule(m)
AllChem.MMFFOptimizeMolecule(m)

print(Descriptors3D.NPR1(m), Descriptors3D.NPR2(m))

