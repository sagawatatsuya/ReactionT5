import os
import random
import itertools
import warnings
import logging
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tokenizers
from transformers import AutoTokenizer, AutoConfig, AutoModel, T5EncoderModel, get_linear_schedule_with_warmup, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from datasets import load_dataset, load_metric
import sentencepiece
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from datasets.utils.logging import disable_progress_bar
import subprocess
disable_progress_bar()
import sys
sys.path.append('../')
from utils import seed_everything, canonicalize, space_clean, get_logger, AverageMeter, asMinutes, timeSince
from models import ReactionT5Yield

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help=" Data as a string or CSV file that contains an 'input' column. The format of the string or contents of the column are like 'REACTANT:{reactants of the reaction}PRODUCT:{products of the reaction}'. If there are multiple reactants, concatenate them with '.'.(ex. 'REACTANT:NCCO.O=C1COCC(=O)O1CATALYST: REAGENT: SOLVENT:c1ccncc1NoData: '"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", 
        type=str, 
        required=False,
        help="Load finetuned model weight later. So this is not necessary."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="t5", 
        required=False,
        help="Model name used for prediction. Currentry, only t5 is expected."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        help="The name of a finetuned model or path to a model which you want to use for prediction. You can use your local models or models uploaded to hugging face."
    )
    parser.add_argument(
        "--download_pretrained_model", 
        action='store_true', 
        default=False, 
        required=False,
        help="Download finetuned model from hugging face hub and use it for prediction."
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        default=False, 
        required=False,
        help="Use debug mode."
    )
    parser.add_argument(
        "--max_len",
        type=int, 
        default=512, 
        required=False,
        help="Max input token length."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=5, 
        required=False,
        help="Batch size."
    )
    parser.add_argument(
        "--fc_dropout", 
        type=float, 
        default=0.0, 
        required=False,
        help="Drop out rate after fully connected layers."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=1, 
        required=False,
        help="Number of workers used for training."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='./', 
        required=False,
        help="The directory where prediction is saved."
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=42, 
        required=False,
        help="Set seed for reproducibility."
    )

    return parser.parse_args()

CFG = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG.device = device

OUTPUT_DIR = CFG.output_dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

seed_everything(seed=CFG.seed)  

if CFG.download_pretrained_model:
    try:
        os.mkdir('tokenizer')
    except:
        print('already tokenizer exists')
    subprocess.run('wget https://huggingface.co/spaces/sagawa/predictyield-t5/resolve/main/ZINC-t5_best.pth', shell=True)
    subprocess.run('wget https://huggingface.co/spaces/sagawa/predictyield-t5/resolve/main/config.pth', shell=True)
    subprocess.run('wget https://huggingface.co/spaces/sagawa/predictyield-t5/raw/main/special_tokens_map.json -P ./tokenizer', shell=True)
    subprocess.run('wget https://huggingface.co/spaces/sagawa/predictyield-t5/raw/main/tokenizer.json -P ./tokenizer', shell=True)
    subprocess.run('wget https://huggingface.co/spaces/sagawa/predictyield-t5/raw/main/tokenizer_config.json -P ./tokenizer', shell=True)
    CFG.model_name_or_path = '.'

try: # load pretrained tokenizer from local directory
    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path+'/tokenizer', return_tensors='pt')
except: # load pretrained tokenizer from huggingface model hub
    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path, return_tensors='pt')

def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text, add_special_tokens=True, max_length=CFG.max_len, padding='max_length', return_offsets_mapping=False, truncation=True, return_attention_mask=True)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    
    return inputs

class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df['input'].values
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.inputs[item])
        
        return inputs    

def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

model = ReactionT5Yield(CFG, config_path=CFG.model_name_or_path + '/config.pth', pretrained=False)
state = torch.load(CFG.model_name_or_path + '/ZINC-t5_best.pth', map_location=torch.device('cpu'))
model.load_state_dict(state)


if 'csv' in CFG.data:
    from rdkit import Chem
    def canonicalize(mol):
        mol = Chem.MolToSmiles(Chem.MolFromSmiles(mol),True)
        return mol
    
    def preprocess(df):
        df['REAGENT'] = df['REAGENT'].apply(lambda x: canonicalize(x) if x != ' ' else ' ')
        df['REACTANT'] = df['REACTANT'].apply(lambda x: canonicalize(x) if x != ' ' else ' ')
        df['PRODUCT'] = df['PRODUCT'].apply(lambda x: canonicalize(x) if x != ' ' else ' ')
        df['YIELD'] = df['YIELD'].clip(0, 100)/100
        df['input'] = 'REACTANT:' + df['REACTANT']  + 'REAGENT:' + df['REAGENT'] + 'PRODUCT:' + df['PRODUCT']
        df = df[['input', 'YIELD']].drop_duplicates().reset_index(drop=True)
        lens = df['input'].apply(lambda x: len(x))
        # remove data that have too long inputs
        df = df[lens <= 512].reset_index(drop=True)

        return df
    
    test_ds = pd.read_csv(CFG.data)
    
    if 'input' not in test_ds.columns:
        test_ds = preprocess(test_ds)
    
    if CFG.debug:
        test_ds = test_ds[:500]
        
    test_dataset = TestDataset(CFG, test_ds)
    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)


    prediction = inference_fn(test_loader, model, device)

    test_ds['prediction'] = prediction*100
    test_ds['prediction'] = test_ds['prediction'].clip(0, 100)
    test_ds.to_csv(CFG.output_dir + 'yield_prediction_output.csv', index=False)
    
else:
    
    test_ds = pd.DataFrame.from_dict({'input': CFG.data}, orient='index').T
    test_dataset = TestDataset(CFG, test_ds)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)


    prediction = inference_fn(test_loader, model, device)
    prediction = max(min(prediction[0][0]*100, 100), 0)
    print('yiled: ', prediction)
 