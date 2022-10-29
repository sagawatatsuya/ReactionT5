import os
import gc
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import tokenizers
import transformers
from transformers import AutoTokenizer, EncoderDecoderModel, AutoModelForSeq2SeqLM
import sentencepiece
import argparse
from rdkit import Chem
import rdkit



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=False)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_beams", type=int, default=5, required=False)
    parser.add_argument("--num_return_sequences", type=int, default=5, required=False)
    parser.add_argument("--debug", action='store_true', default=False, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)

    return parser.parse_args()
    
CFG = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=CFG.seed)  


tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path, return_tensors='pt')
    
if CFG.model == 't5':
    model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name_or_path).to(device)
elif CFG.model == 'deberta':
    model = EncoderDecoderModel.from_pretrained(CFG.model_name_or_path).to(device)
    

if 'csv' not in CFG.input_data:
    input_compound = CFG.input_data
    min_length = min(input_compound.find('CATALYST') - input_compound.find(':') - 10, 0)
    inp = tokenizer(input_compound, return_tensors='pt').to(device)
    output = model.generate(**inp, min_length=min_length, max_length=min_length+50, num_beams=CFG.num_beams, num_return_sequences=CFG.num_return_sequences, return_dict_in_generate=True, output_scores=True)
    scores = output['sequences_scores'].tolist()
    output = [tokenizer.decode(i, skip_special_tokens=True).replace('. ', '.').rstrip('.') for i in output['sequences']]
    for ith, out in enumerate(output):
        mol = Chem.MolFromSmiles(out.rstrip('.'))
        if type(mol) == rdkit.Chem.rdchem.Mol:
            output.append(out.rstrip('.'))
            scores.append(scores[ith])
            break
    if type(mol) == None:
        output.append(None)
        scores.append(None)
    output += scores
    output = [input_compound] + output
    output_df = pd.DataFrame(np.array(output).reshape(1, -1), columns=['input'] + [f'{i}th' for i in range(CFG.num_beams)] + ['valid compound'] + [f'{i}th score' for i in range(CFG.num_beams)] + ['valid compound score'])
    output_df.to_csv('output.csv', index=False)

elif 'csv' in CFG.input_data:
    input_data = pd.read_csv(CFG.input_data)
    if CFG.debug:
        input_data = input_data[:10]
    outputs = []
    for idx, row in input_data.iterrows():
        input_compound = row['input']
        min_length = min(input_compound.find('CATALYST') - input_compound.find(':') - 10, 0)################# min_length, max_lengthの調節は要注意
        inp = tokenizer(input_compound, return_tensors='pt').to(device)
        output = model.generate(**inp, min_length=min_length, max_length=min_length+50, num_beams=CFG.num_beams, num_return_sequences=CFG.num_return_sequences, return_dict_in_generate=True, output_scores=True)
        scores = output['sequences_scores'].tolist()
        output = [tokenizer.decode(i, skip_special_tokens=True).replace('. ', '.').rstrip('.') for i in output['sequences']]
        for ith, out in enumerate(output):
            mol = Chem.MolFromSmiles(out.rstrip('.'))
            if type(mol) == rdkit.Chem.rdchem.Mol:
                output.append(out.rstrip('.'))
                scores.append(scores[ith])
                break
        if type(mol) == None:
            output.append(None)
            scores.append(None)
        output += scores
        output = [input_compound] + output
        outputs.append(output)
    output_df = pd.DataFrame(outputs, columns=['input'] + [f'{i}th' for i in range(CFG.num_beams)] + ['valid compound'] + [f'{i}th score' for i in range(CFG.num_beams)] + ['valid compound score'])
    output_df.to_csv('multiinput_prediction_output.csv', index=False)

