import os
import gc
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tokenizers
from transformers import AutoTokenizer, AutoConfig, AutoModel, T5EncoderModel, get_linear_schedule_with_warmup, T5ForConditionalGeneration
from datasets import load_dataset, load_metric
import sentencepiece
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
import time
import math
from sklearn.preprocessing import MinMaxScaler
from datasets.utils.logging import disable_progress_bar
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
disable_progress_bar()
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys
sys.path.append('../')
from utils import seed_everything, canonicalize, space_clean, get_logger, AverageMeter, asMinutes, timeSince, get_optimizer_params
from models import ReactionT5Yield

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="The path to data used for training. CSV file that contains ['CATALYST', 'REACTANT', 'REAGENT', 'SOLVENT', 'INTERNAL_STANDARD', 'NoData','PRODUCT'] columns is expected. If there are missing values, please fill them with ' '."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="t5", 
        required=False,
        help="Model name used for training. Currentry, only t5 is expected."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", 
        type=str, 
        default="sagawa/CompoundT5",
        required=False,
        help="The name of a pretrained model or path to a model which you want to use for training. You can use your local models or models uploaded to hugging face."
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        default=False, 
        required=False,
        help="Use debug mode."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=5, 
        required=False,
        help="Number of epochs for training."
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=10, 
        required=False,
        help="Early stopping patience."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=5e-4, 
        required=False,
        help="Learning rate."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=5, 
        required=False,
        help="Batch size."
    )
    parser.add_argument(
        "--max_len",
        type=int, 
        default=400, 
        required=False,
        help="Max input token length."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4, 
        required=False,
        help="Number of workers used for training."
    )
    parser.add_argument(
        "--fc_dropout", 
        type=float, 
        default=0.0, 
        required=False,
        help="Drop out rate after fully connected layers."
    )
    parser.add_argument(
        "--eps", 
        type=float, 
        default=1e-6, 
        required=False,
        help="Eps of Adam optimizer."
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.05, 
        required=False,
        help="weight_decay used for optimizer"
    )
    parser.add_argument(
        "--max_grad_norm", 
        type=int, 
        default=1000, 
        required=False,
        help="max_grad_norm used for clip_grad_norm_"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1, 
        required=False,
        help="Number of epochs to accumulate gradient."
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=0, 
        required=False,
        help="num_warmup_steps"
    )
    parser.add_argument(
        "--batch_scheduler", 
        action='store_true', 
        default=False, 
        required=False,
        help="Use batch_scheduler"
    )
    parser.add_argument(
        "--print_freq", 
        type=int, 
        default=100, 
        required=False,
        help="Logging frequency."
    )
    parser.add_argument(
        "--use_apex", 
        action='store_true',
        default=False, 
        required=False,
        help="Use apex."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='./', 
        required=False,
        help="The directory where trained model is saved."
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
CFG.batch_scheduler = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG.device = device

OUTPUT_DIR = CFG.output_dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

seed_everything(seed=CFG.seed)  
    

df = pd.read_csv(CFG.data_path).drop_duplicates().reset_index(drop=True)
df = df[~df['YIELD'].isna()].reset_index(drop=True)
df = df[~(df['YIELD']>100)].reset_index(drop=True)
df['YIELD'] = df['YIELD']/100
df = df[~(df['REACTANT'].isna() | df['PRODUCT'].isna())]
for col in ['CATALYST', 'REACTANT', 'REAGENT', 'SOLVENT', 'INTERNAL_STANDARD', 'NoData','PRODUCT']:
    df[col] = df[col].fillna(' ')
    

df['REAGENT'] = df['CATALYST'] + '.' + df['REAGENT']
df['REAGENT'] = df['REAGENT'].apply(lambda x: space_clean(x))
df['REACTANT'] = df['REACTANT'].apply(lambda x: canonicalize(x) if x != ' ' else ' ')
df['REAGENT'] = df['REAGENT'].apply(lambda x: canonicalize(x) if x != ' ' else ' ')
df['PRODUCT'] = df['PRODUCT'].apply(lambda x: canonicalize(x) if x != ' ' else ' ')


df['input'] = 'REACTANT:' + df['REACTANT']  + 'REAGENT:' + df['REAGENT'] + 'PRODUCT:' + df['PRODUCT']
df = df[['input', 'YIELD']].drop_duplicates().reset_index(drop=True)


lens = df['input'].apply(lambda x: len(x))
df = df[lens <= 512].reset_index(drop=True)

train_ds, test_ds = train_test_split(df, test_size=int(len(df)*0.1))
train_ds, valid_ds = train_test_split(train_ds, test_size=int(len(df)*0.1))
train_ds.to_csv('regression-input-train.csv', index=False)
valid_ds.to_csv('regression-input-valid.csv', index=False)
test_ds.to_csv('regression-input-test.csv', index=False)

if CFG.debug:
    train_ds = train_ds[:int(len(train_ds)/4)].reset_index(drop=True)
    valid_ds = valid_ds[:int(len(valid_ds)/4)].reset_index(drop=True)
        

LOGGER = get_logger(OUTPUT_DIR+'train')

#load tokenizer
try: # load pretrained tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(CFG.pretrained_model_name_or_path), return_tensors='pt')
except: # load pretrained tokenizer from huggingface model hub
    tokenizer = AutoTokenizer.from_pretrained(CFG.pretrained_model_name_or_path, return_tensors='pt')
tokenizer.add_tokens(['.', 'P', '>', '<','Pd'])

tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + ['REACTANT:', 'PRODUCT:', 'REAGENT:']})
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text, add_special_tokens=True, max_length=CFG.max_len, padding='max_length', return_offsets_mapping=False, truncation=True, return_attention_mask=True)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df['input'].values
        self.labels = df['YIELD'].values
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.inputs[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        
        return inputs, label
    


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.use_apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        if CFG.gradient_accumulation_steps > 1:
            loss = loss/CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]), flush=True)
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    start = end = time.time()
    label_list = []
    pred_list = []
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        label_list += labels.tolist()
        pred_list += y_preds.tolist()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'RMSE Loss: {loss:.4f} '
                  'r2 score: {r2_score:.4f} '
                  .format(step, len(valid_loader),
                          loss=mean_squared_error(label_list, pred_list, squared=False),
                          remain=timeSince(start, float(step+1)/len(valid_loader)),
                          r2_score=r2_score(label_list, pred_list)))
    return mean_squared_error(label_list, pred_list), r2_score(label_list, pred_list)
    
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


def train_loop(train_ds, valid_ds):
    
    train_dataset = TrainDataset(CFG, train_ds)
    valid_dataset = TrainDataset(CFG, valid_ds)
    valid_labels = valid_ds['YIELD'].values
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    model = ReactionT5Yield(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    
    optimizer_parameters = get_optimizer_params(model, encoder_lr=CFG.lr, decoder_lr=CFG.lr, weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.lr, eps=CFG.eps, betas=(0.9, 0.999))
    
    num_train_steps = int(len(train_ds)/CFG.batch_size*CFG.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_train_steps)
    
    criterion = nn.MSELoss(reduction='mean')
    best_loss = float('inf')
    es_count = 0
    
    for epoch in range(CFG.epochs):
        start_time = time.time()

        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        val_loss, val_r2_score = valid_fn(valid_loader, model, criterion, device)
        
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  val_rmse_loss: {val_loss:.4f}  val_r2_score: {val_r2_score:.4f}  time: {elapsed:.0f}s')
    
        if val_loss < best_loss:
            es_count = 0
            best_loss = val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Lowest Loss: {best_loss:.4f} Model')
            torch.save(model.state_dict(), OUTPUT_DIR+f"{CFG.pretrained_model_name_or_path.split('/')[-1]}_best.pth")
            
        else:
            es_count += 1
            if es_count >= CFG.patience:
                print('early_stopping')
                break
    
    torch.cuda.empty_cache()
    gc.collect()

            
if __name__ == '__main__':
    train_loop(train_ds, valid_ds)
        
 