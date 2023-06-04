import os
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel, T5EncoderModel, get_linear_schedule_with_warmup, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import datasets
from datasets import load_dataset, load_metric
import sentencepiece
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import pickle
from datasets.utils.logging import disable_progress_bar
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
disable_progress_bar()
import sys
sys.path.append('../../')
from utils import seed_everything, canonicalize, space_clean, get_logger, AverageMeter, asMinutes, timeSince

class CFG():
    data_path = 'nodata.csv'
    pretrained_model_name_or_path='./tokenizer'
    debug = True
    model = 't5'
    epochs = 5
    lr = 2e-5
    batch_size = 64
    max_len = 128
    weight_decay = 0.01
    seed = 42
    num_workers = 4
    fc_dropout = 0.1
    eps = 1e-6
    max_grad_norm=1000
    gradient_accumulation_steps=1
    num_warmup_steps=0
    batch_scheduler=True
    print_freq=100
    use_apex=False
    output_file = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_everything(seed=CFG.seed)  


#load tokenizer
try: # load pretrained tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(CFG.pretrained_model_name_or_path), return_tensors='pt')
except: # load pretrained tokenizer from huggingface model hub
    tokenizer = AutoTokenizer.from_pretrained(CFG.pretrained_model_name_or_path, return_tensors='pt')

CFG.tokenizer = tokenizer
def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text, add_special_tokens=True, max_length=CFG.max_len, padding='max_length', return_offsets_mapping=False, truncation=True, return_attention_mask=True)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    
    return inputs

    
class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df['smiles'].values
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.inputs[item])
        
        return inputs
    
       
class RegressionModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            if 't5' in cfg.model:
                self.model = T5ForConditionalGeneration.from_pretrained(CFG.pretrained_model_name_or_path)
            else:
                self.model = AutoModel.from_pretrained(CFG.pretrained_model_name_or_path)
        else:
            if 't5' in cfg.model:
                self.model = T5ForConditionalGeneration.from_pretrained('sagawa/ZINC-t5')
            else:
                self.model = AutoModel.from_config(self.config)
        self.model.resize_token_embeddings(len(cfg.tokenizer))
        self.fc_dropout1 = nn.Dropout(cfg.fc_dropout)
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.hidden_size//2)
        self.fc_dropout2 = nn.Dropout(cfg.fc_dropout)
        
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.hidden_size//2)
        self.fc3 = nn.Linear(self.config.hidden_size//2*2, self.config.hidden_size)
        self.fc4 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc5 = nn.Linear(self.config.hidden_size, 2)

        self._init_weights(self.fc1)
        self._init_weights(self.fc2)
        self._init_weights(self.fc3)
        self._init_weights(self.fc4)
        self._init_weights(self.fc5)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, inputs):
        encoder_outputs = self.model.encoder(**inputs)
        encoder_hidden_states = encoder_outputs[0]
        outputs = self.model.decoder(input_ids=torch.full((inputs['input_ids'].size(0),1),
                                            self.config.decoder_start_token_id,
                                            dtype=torch.long,
                                            device=device), encoder_hidden_states=encoder_hidden_states)
        last_hidden_states = outputs[0]
        output1 = self.fc1(self.fc_dropout1(last_hidden_states).view(-1, self.config.hidden_size))
        output2 = self.fc2(encoder_hidden_states[:, 0, :].view(-1, self.config.hidden_size))
        output = self.fc3(self.fc_dropout2(torch.hstack((output1, output2))))
        output = self.fc4(output)
        output = self.fc5(output)
        return F.softmax(output)

    
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


valid_ds = pd.read_csv(CFG.data_path)
CFG.model_name_or_path = 't5'
valid_dataset = TestDataset(CFG, valid_ds)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)
model = RegressionModel(CFG, config_path='./config.pth', pretrained=False)
state = torch.load('./ZINC-t5_best.pth', map_location=torch.device('cpu'))
model.load_state_dict(state)

model.to(device)
preds = []
for inputs in valid_loader:
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        y_preds = model(inputs)
        preds.append(y_preds.tolist())

with open('pred.pkl', 'bw') as f:
    pickle.dump(preds, f)
    
    
lis = []
for p in pred:
    lis += p
lis = np.array(lis)
pred = lis[:, 1] > 0.97
valid_ds['pred'] = pred
valid_ds.to_csv('nodata-prediction.csv', index=False)