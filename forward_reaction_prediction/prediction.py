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
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import rdkit
import sys
sys.path.append('../')
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data", 
        type=str, 
        required=True, 
        help="The path to data used for training. CSV file that contains 'input' column is expected."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="t5", 
        required=False,
        help="Model name used for prediction."
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="sagawa/ReactionT5-product-prediction",
        required=False,
        help="The name of a finetuned model or path to a model which you want to use for prediction. You can use your local models or models uploaded to hugging face."
    )
    parser.add_argument(
        "--num_beams", 
        type=int, 
        default=5, 
        required=False,
        help="Number of beams used for beam_search for targets."
    )
    parser.add_argument(
        "--num_return_sequences", 
        type=int, 
        default=5, 
        required=False,
        help="Number of predictions that is returned as an output. This must be smaller than or equal to num_beams."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=5, 
        required=False,
        help="Batch size"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='./', 
        required=False,
        help="The directory where prediction is saved."
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        default=False, 
        required=False,
        help="Use debug mode."
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

OUTPUT_DIR = CFG.output_dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

seed_everything(seed=CFG.seed)  

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path, return_tensors='pt')
CFG.tokenizer = tokenizer    

if CFG.model == 't5':
    model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name_or_path).to(device)
elif CFG.model == 'deberta':
    model = EncoderDecoderModel.from_pretrained(CFG.model_name_or_path).to(device)


if 'csv' not in CFG.input_data:
    input_compound = CFG.input_data
    length = input_compound.find('REAGENT') - input_compound.find(':')
    inp = tokenizer(input_compound, return_tensors='pt').to(device)
    output = model.generate(**inp, min_length=max(length - 20, 0), max_length=length+20, num_beams=CFG.num_beams, num_return_sequences=CFG.num_return_sequences, return_dict_in_generate=True, output_scores=True)
    if CFG.num_beams > 1:
        scores = output['sequences_scores'].tolist()
        output = [tokenizer.decode(i, skip_special_tokens=True).replace(' ', '').rstrip('.') for i in output['sequences']]
        # get the most probable valid prediction
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
        
    else:
        output = [tokenizer.decode(output['sequences'][0], skip_special_tokens=True).replace(' ', '').rstrip('.')]
        mol = Chem.MolFromSmiles(output[0])
        if type(mol) == rdkit.Chem.rdchem.Mol:
            output.append(output[0])
        else:
            output.append(None)
        
        output_df = pd.DataFrame(np.array([input_compound]+output).reshape(1, -1), columns=['input', '0th', 'valid compound'])

elif 'csv' in CFG.input_data:
    input_data = pd.read_csv(CFG.input_data)
    if CFG.debug:
        input_data = input_data[:10]

    
    def prepare_input(cfg, text):
        inputs = cfg.tokenizer(text, return_tensors='pt', max_length=400, padding='max_length', truncation=True)
        dic = {'input_ids': [], 'attention_mask': []}
        for k, v in inputs.items():
            dic[k].append(torch.tensor(v[0], dtype=torch.long))
        return dic

    class ProductDataset(Dataset):
        def __init__(self, cfg, df):
            self.cfg = cfg
            self.inputs = df['input'].values

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, item):
            inputs = prepare_input(self.cfg, self.inputs[item])

            return inputs

    dataset = ProductDataset(CFG, input_data)
    dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    outputs = []
    for inputs in dataloader:
        for k, v in inputs.items():
            inputs[k] = v[0].to(device)
        output = model.generate(**inputs, min_length=6, max_length=109, num_beams=CFG.num_beams, num_return_sequences=CFG.num_return_sequences, return_dict_in_generate=True, output_scores=True)
        outputs.append(output)

    if CFG.num_beams > 1:
        sequences = []
        sequences_scores = []
        for ith, output in enumerate(outputs):
            seqs = [CFG.tokenizer.decode(i, skip_special_tokens=True).replace(' ', '').rstrip('.') for i in output['sequences']]
            scores = output['sequences_scores'].tolist()
            sequences += seqs
            sequences_scores += scores

        output_list = []
        for i in range(0, len(sequences), CFG.num_return_sequences):
            output_list.append([input_data.loc[i//CFG.num_return_sequences, 'input']]+sequences[i:i+CFG.num_return_sequences] + sequences_scores[i:i+CFG.num_return_sequences])
        output_df = pd.DataFrame(output_list, columns=['input'] + [f'{i}th' for i in range(CFG.num_return_sequences)] + [f'{i}th score' for i in range(CFG.num_return_sequences)])
    else:
        sequences = []
        for ith, output in enumerate(outputs):
            seqs = [CFG.tokenizer.decode(i, skip_special_tokens=True).replace(' ', '').rstrip('.') for i in output['sequences']]
            sequences += seqs

        output_list = []
        for i in range(0, len(sequences), CFG.num_return_sequences):
            output_list.append([input_data.loc[i//CFG.num_return_sequences, 'input']]+sequences[i:i+CFG.num_return_sequences])
        output_df = pd.DataFrame(output_list, columns=['input'] + [f'{i}th' for i in range(CFG.num_return_sequences)])

output_df.to_csv(CFG.output_dir + 'forward_reaction_prediction_output.csv', index=False)