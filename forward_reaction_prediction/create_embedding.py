import os
import gc
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import tokenizers
import transformers
from transformers import AutoTokenizer, EncoderDecoderModel, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig, AutoModel, T5EncoderModel
import datasets
from datasets import load_dataset, load_metric
import sentencepiece
import argparse
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import rdkit
disable_progress_bar()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=False)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action='store_true', default=False, required=False)
    parser.add_argument("--num_beams", type=int, default=5, required=False)
    parser.add_argument("--num_return_sequences", type=int, default=5, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)

    return parser.parse_args()
    
# CFG = parse_args()

class CFG:
    model = 't5'
    dataset_path = '/data2/sagawa/tcrp-multiinput-archive/12-3-agg-into-reagent-w-nodata/multi-input-test.csv'
    model_name_or_path = '/data2/sagawa/tcrp-multiinput-archive/12-3-agg-into-reagent-w-nodata/t5/checkpoint-1986462'
    num_beams = 5
    num_return_sequences = 5
    debug = True
    seed = 42
    max_len = 512
    num_workers = 1
    batch_size=3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=CFG.seed)  
    
def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text, add_special_tokens=True, max_length=cfg.max_len, padding='max_length', truncation=True)
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
    
def create_embedding(dataloader, model, device):
    outputs = []
    outputs_cls = []
    model.eval()
    model.to(device)
    tk0 = tqdm(dataloader, total=len(dataloader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            output = model(**inputs)
        outputs.append(output[0].detach().cpu().numpy())
        outputs_cls.append(output[0][:, 0, :].detach().cpu().numpy())
    
    return outputs, outputs_cls
        

CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path, return_tensors='pt')
model = T5EncoderModel.from_pretrained(CFG.model_name_or_path)

    
df = pd.read_csv(CFG.dataset_path)[:100]
dataset = TestDataset(CFG, df)
dataloader = DataLoader(dataset,
                        batch_size=CFG.batch_size,
                        shuffle=False,
                        num_workers=CFG.num_workers,
                        pin_memory=True,
                        drop_last=False
                       )

outputs, outputs_cls = create_embedding(dataloader, model, device)

outputs = np.concatenate(outputs, axis=0)
outputs_cls = np.concatenate(outputs_cls, axis=0)

np.save('embedding_5000.npy', outputs)
np.save('embedding_cls_5000.npy', outputs_cls)