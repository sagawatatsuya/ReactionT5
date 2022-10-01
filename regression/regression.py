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
from transformers import AutoTokenizer, AutoConfig, AutoModel, T5EncoderModel, get_linear_schedule_with_warmup
import datasets
from datasets import load_dataset, load_metric
import sentencepiece
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
import time
import math
from sklearn.preprocessing import MinMaxScaler
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train", action='store_true', default=False, required=False)
#     parser.add_argument("--data_path", type=str, required=False)
#     parser.add_argument("--dataset_name", type=str, required=False)
#     parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
#     parser.add_argument("--model", type=str, required=True)
#     parser.add_argument("--debug", action='store_true', default=False, required=False)
#     parser.add_argument("--epochs", type=int, default=3, required=False)
#     parser.add_argument("--lr", type=float, default=2e-5, required=False)
#     parser.add_argument("--batch_size", type=int, default=16, required=False)
#     parser.add_argument("--max_len", type=int, default=128, required=False)
#     parser.add_argument("--weight_decay", type=float, default=0.01, required=False)
#     parser.add_argument("--evaluation_strategy", type=str, default="epoch", required=False)
#     parser.add_argument("--save_strategy", type=str, default="epoch", required=False)
#     parser.add_argument("--logging_strategy", type=str, default="epoch", required=False)
#     parser.add_argument("--save_total_limit", type=int, default=3, required=False)
#     parser.add_argument("--fp16", action='store_true', default=False, required=False)
#     parser.add_argument("--disable_tqdm", action="store_true", default=False, required=False)
#     parser.add_argument("--multitask", action="store_true", default=False, required=False)
#     parser.add_argument("--shuffle_augmentation", type=int, default=0, required=False)
#     parser.add_argument("--noncanonical_augmentation", type=int, default=0, required=False)
#     parser.add_argument("--seed", type=int, default=42, required=False)

#     return parser.parse_args()

class CFG():
    data='../../all_ord_reaction_uniq_with_attr_v3.tsv'
#     pretrained_model_name_or_path = 'sagawa/ZINC-t5'
    model = 'sagawa/ZINC-t5'
    debug = True
    epochs = 5
    lr = 2e-5
    batch_size = 5 #max_lenを大きくしたらoom下から15から5に
    max_len = 512
    weight_decay = 0.01
    evaluation_strategy = 'epoch'
    save_strategy = 'epoch'
    save_total_limit = 1
    seed = 42
    num_workers = 4
    fc_dropout = 0.2
    eps = 1e-6
    max_grad_norm=1000
    betas=(0.9, 0.999)
    gradient_accumulation_steps=3
    num_warmup_steps=0
    batch_scheduler=True
    print_freq=100
    apex=False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=CFG.seed)  
    

df = pd.read_csv(CFG.data)
# if CFG.debug:
#     df = df[:1000]
# df = df.dropna().reset_index(drop=True)
df = df[~df['YIELD'].isna()]
for col in ['CATALYST', 'REACTANT', 'REAGENT', 'SOLVENT', 'INTERNAL_STANDARD', 'NoData','PRODUCT', 'YIELD', 'TEMP']:
    df[col] = df[col].fillna(' ')
# df['input'] = 'REACTANT:' + df['REACTANT'] + 'PRODUCT:' + df['PRODUCT'] + 'CATALYST:' + df['CATALYST'] + 'REAGENT:' + df['REAGENT'] + 'SOLVENT:' + df['SOLVENT'] + 'NoData:' + df['NoData']
df['input'] = 'REACTANT:' + df['REACTANT'] + 'PRODUCT:' + df['PRODUCT'] + 'CATALYST:' + df['CATALYST']

lens = df['input'].apply(lambda x: len(x))
df = df[lens <= 512]
train_ds, test_ds = train_test_split(df, test_size=int(len(all)*0.1))
train_ds, valid_ds = train_test_split(train_ds, test_size=int(len(all)*0.1))
train_ds.to_csv('regression-input-train.csv', index=False)
valid_ds.to_csv('regression-input-valid.csv', index=False)
test_ds.to_csv('regression-input-test.csv', index=False)

train_ds = train_ds[train_ds['YIELD']<df['YIELD'].quantile(0.999)].reset_index(drop=True)
valid_ds = valid_ds.reset_index(drop=True)

scaler = MinMaxScaler()
train_ds['YIELD'] = scaler.fit_transform(train_ds['YIELD'].values.reshape(-1, 1))
valid_ds['YIELD'] = scaler.transform(valid_ds['YIELD'].values.reshape(-1, 1))


OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + ['CATALYST:', 'REACTANT:', 'REAGENT:', 'SOLVENT:', 'INTERNAL_STANDARD:', 'NoData:','PRODUCT:']})
CFG.tokenizer = tokenizer
def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text, add_special_tokens=True, max_length=CFG.max_len, padding='max_length', return_offsets_mapping=False, truncation=True)
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
    
class RegressionModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            if 't5' in cfg.model:
                self.model = T5EncoderModel.from_pretrained(CFG.model)
            else:
                self.model = AutoModel.from_pretrained(CFG.model)
        else:
            self.model = AutoModel.from_config(self.config)
        self.model.resize_token_embeddings(len(cfg.tokenizer))
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
#         print(last_hidden_states.shape)
        output = self.fc(self.fc_dropout(last_hidden_states)[:, 0, :].view(-1, self.config.hidden_size))
#         print(output.shape)
        return output
    
class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count
        

def asMinutes(s):
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s/(percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
#         print(y_preds.shape)
#         print(labels)
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
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        if CFG.gradient_accumulation_steps > 1:
            loss = loss/CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    return losses.avg
    
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
    
    model = RegressionModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)], 'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if 'model' not in n], 'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters
    
    optimizer_parameters = get_optimizer_params(model, encoder_lr=CFG.lr, decoder_lr=CFG.lr, weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.lr, eps=CFG.eps, betas=CFG.betas)
    
    num_train_steps = int(len(train_ds)/CFG.batch_size*CFG.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_train_steps)
    
    criterion = nn.MSELoss(reduction='mean')
    best_loss = 0
    
    for epoch in range(CFG.epochs):
        start_time = time.time()

        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        
        # prediction削除
        avg_val_loss = valid_fn(valid_loader, model, criterion, device)
        
#         score = get_score(valid_labels, prediction)
        
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
#         LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
    
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Lowest Loss: {best_loss:.4f} Model')
            torch.save(model.state_dict(), OUTPUT_DIR+f"{CFG.model}_best.pth")
    
    torch.cuda.empty_cache()
    gc.collect()

            
if __name__ == '__main__':
    train_loop(train_ds, valid_ds)
        
 