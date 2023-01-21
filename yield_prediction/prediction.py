import os
import gc
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
import time
from sklearn.preprocessing import MinMaxScaler
from datasets.utils.logging import disable_progress_bar
from sklearn.metrics import mean_squared_error, r2_score
import subprocess
disable_progress_bar()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=False)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="sagawa/ZINC-t5", required=False)
    parser.add_argument("--model", type=str, default="t5", required=False)
    parser.add_argument("--model_name_or_path", type=str, required=False)
    parser.add_argument("--download_pretrained_model", action='store_true', default=False, required=False)
    parser.add_argument("--debug", action='store_true', default=False, required=False)
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--batch_size", type=int, default=5, required=False)
    parser.add_argument("--fc_dropout", type=float, default=0.1, required=False)
    parser.add_argument("--num_workers", type=int, default=1, required=False)
    parser.add_argument("--output_dir", type=str, default='./', required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)

    return parser.parse_args()

CFG = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = CFG.output_dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=CFG.seed)  

if CFG.download_pretrained_model:
    os.mkdir('tokenizer')
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
        self.fc5 = nn.Linear(self.config.hidden_size, 1)

        self._init_weights(self.fc1)
        self._init_weights(self.fc2)
        self._init_weights(self.fc3)
        self._init_weights(self.fc4)
        
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
        return output
    

    
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

model = RegressionModel(CFG, config_path=CFG.model_name_or_path + '/config.pth', pretrained=False)
state = torch.load(CFG.model_name_or_path + '/ZINC-t5_best.pth', map_location=torch.device('cpu'))
model.load_state_dict(state)


if 'csv' in CFG.data:
    test_ds = pd.read_csv(CFG.data)
    
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
 