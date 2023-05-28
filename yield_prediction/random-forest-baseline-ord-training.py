import os
import gc
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import sentencepiece
import argparse
import pickle
import torch
import time
import math
from sklearn.preprocessing import MinMaxScaler
from datasets.utils.logging import disable_progress_bar
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()  

from rdkit import Chem
def canonicalize(mol):
    mol = Chem.MolToSmiles(Chem.MolFromSmiles(mol),True)
    return mol

def preprocess(df):
    df['REAGENT'] = df['REAGENT'].apply(lambda x: canonicalize(x) if x != ' ' else ' ')
    df['REACTANT'] = df['REACTANT'].apply(lambda x: canonicalize(x) if x != ' ' else ' ')
    df['PRODUCT'] = df['PRODUCT'].apply(lambda x: canonicalize(x) if x != ' ' else ' ')
    df['YIELD'] = df['YIELD'].clip(0, 100)
    df['input'] = df['REACTANT']  + '.' + df['REAGENT'] + '.' + df['PRODUCT']
    df['input'] = df['input'].str.replace('. ', '')
    df = df[['input', 'YIELD']].drop_duplicates().reset_index(drop=True)
    lens = df['input'].apply(lambda x: len(x))
    # remove data that have too long inputs
    df = df[lens <= 512].reset_index(drop=True)
    
    return df

def preprocess_ord(df):
    df['input'] = df['input'].str.replace('REAGENT:', '.')
    df['input'] = df['input'].str.replace('REACTANT:', '')
    df['input'] = df['input'].str.replace('PRODUCT:', '.')
    df['input'] = df['input'].str.replace('. ', '')
    
    return df
    
df = pd.read_csv('/data2/sagawa/regression-input-train.csv').drop_duplicates().reset_index(drop=True)
df['YIELD'] = df['YIELD']*100
train_ds = preprocess_ord(df)

df = pd.read_csv('/data2/sagawa/t5chem/data/C_N_yield/MFF_FullCV_01/test.csv').drop_duplicates().reset_index(drop=True)
valid_ds = preprocess(df)


from rdkit.Chem import AllChem
fingerprints = []
dim = 865
for idx, row in train_ds.iterrows():
    mol = Chem.MolFromSmiles(row['input'])
    fingerprints.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, dim)))
train_ds[[str(i) for i in range(dim)]] = fingerprints
fingerprints = []
for idx, row in valid_ds.iterrows():
    mol = Chem.MolFromSmiles(row['input'])
    fingerprints.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, dim)))
valid_ds[[str(i) for i in range(dim)]] = fingerprints


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
model = RandomForestRegressor(n_estimators=785, max_depth=914, random_state=42)
model.fit(train_ds[[str(i) for i in range(dim)]], train_ds['YIELD'])
valid_ds['prediction'] = model.predict(valid_ds[[str(i) for i in range(dim)]])
valid_ds['prediction'] = valid_ds['prediction'].clip(0, 100)

print('r2_score:',r2_score(valid_ds['YIELD'], valid_ds['prediction']))
print('rmse:',mean_squared_error(valid_ds['YIELD'], valid_ds['prediction'], squared=False))
print('accuracy:', sum(abs(valid_ds['YIELD'] - valid_ds['prediction']) <= 10)/len(valid_ds))