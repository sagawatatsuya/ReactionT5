import subprocess
from rdkit import Chem
import pandas as pd
import sys
sys.path.append('../')
from utils import canonicalize

subprocess.run("mkdir data", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1H4kOZpUayA-xSp0HNYRW4YXUcRhM9lhD'", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1DrDEjFNkU1YeubYw94k4K0Sek2xx1Sg1'", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1NPpxUgPiCd_XPC37WvD6crZ5phRxB-ie'", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1aJevXiOF8A6t9tMbfq2bmqoIsBfMyPAZ'", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1ygYs8dy1-vxD1Vx6Ux7ftrXwZctFjpV3'", shell=True)
subprocess.run("gdown 'https://drive.google.com/uc?export=download&id=1BEk2GWhNU-Azj9hm77Z2wufsPN49wN0m'", shell=True)
subprocess.run("gzip -dr .", shell=True)
subprocess.run("mv *.smi data", shell=True)
subprocess.run("mv *.txt data", shell=True)
subprocess.run("mv *.tsv data", shell=True)


lst = set([])
for i in range(4):
    suppl = Chem.SmilesMolSupplier(f'../data/16_p{i}.smi')
    # some molecules cannot be read by rdkit, so we remove them here
    for mol in suppl:
        try:
            sm = Chem.MolToSmiles(mol)
            lst.add(sm)
        except:
            pass
df = pd.DataFrame({'smiles': list(lst)})
df.to_csv('../data/ZINC-canonicalized.csv', index=False)


tx = pd.read_csv('../data/pubchem-10m.txt', names=['smiles'])
lst = set([])
for smiles in tx['smiles']:
    try:
        lst.add(canonicalize(smiles))
    except:
        pass
df = pd.DataFrame({'smiles': list(lst)})
df.to_csv('../data/pubchem-10m-canonicalized.csv', index=False)


df = pd.read_csv('../data/all_ord_reaction_uniq.tsv', sep='\t', names=['id', 'product', 'reactant'])
df['product'] = df['product'].apply(lambda x: canonicalize(x))
df['reactant'] = df['reactant'].apply(lambda x: canonicalize(x))
df.drop(['id'], inplace=True, axis=1)
df.to_csv('../data/all_ord_reaction_uniq_canonicalized.csv', index=False)