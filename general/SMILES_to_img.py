from rdkit import Chem
from rdkit.Chem import Draw

smiles = "NC(=O)[C@H]1CCCN(C1)C(=O)C(=O)NCC2=CC=C(C=C2)OC(F)(F)F"  # aspiryna
mol = Chem.MolFromSmiles(smiles)
Chem.rdDepictor.Compute2DCoords(mol)

img = Draw.MolToImage(mol, size=(400, 300))
img.save("Analog_25_of_JNT-517.png")
