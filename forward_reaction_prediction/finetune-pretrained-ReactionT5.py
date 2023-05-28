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
from transformers import AutoTokenizer, EncoderDecoderModel, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import sentencepiece
import argparse
from sklearn.model_selection import train_test_split
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path", 
        type=str, 
        required=True, 
        help="The path to data used for training. CSV file that contains ['REACTANT', 'REAGENT', 'PRODUCT'] columns is expected."
    )
    parser.add_argument(
        "--valid_data_path", 
        type=str, 
        required=True, 
        help="The path to data used for validation. CSV file that contains ['REACTANT', 'REAGENT', 'PRODUCT'] columns is expected."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="t5", 
        required=False,
        help="Model name used for training. Currentry, only t5 is expected."
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        required=True,
        help="The name of a pretrained model or path to a model which you want to finetune on your dataset. You can use your local models or models uploaded to hugging face."
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
        default=3, 
        required=False,
        help="Number of epochs for training."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=2e-5, 
        required=False,
        help="Learning rate."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        required=False,
        help="Batch size."
    )
    parser.add_argument(
        "--input_max_len",
        type=int, 
        default=128, 
        required=False,
        help="Max input token length."
    )
    parser.add_argument(
        "--target_max_len",
        type=int, 
        default=128, 
        required=False,
        help="Max target token length."
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01, 
        required=False,
        help="weight_decay used for trainer"
    )
    parser.add_argument(
        "--evaluation_strategy", 
        type=str, 
        default="epoch", 
        required=False,
        help="Evaluation strategy used during training. Select from 'no', 'steps', or 'epoch'. If you select 'steps', also give --eval_steps."
    )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        required=False,
        help="Number of update steps between two evaluations"
    )
    parser.add_argument(
        "--save_strategy", 
        type=str, 
        default="epoch", 
        required=False,
        help="Save strategy used during training. Select from 'no', 'steps', or 'epoch'. If you select 'steps', also give --save_steps."
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        required=False,
        default="500",
        help="Number of steps between two saving"
    )
    parser.add_argument(
        "--logging_strategy", 
        type=str, 
        default="epoch", 
        required=False,
        help="Logging strategy used during training. Select from 'no', 'steps', or 'epoch'. If you select 'steps', also give --logging_steps."
    )
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        required=False,
        default="500",
        help="Number of steps between two logging"
    )
    parser.add_argument(
        "--save_total_limit", 
        type=int, 
        default=2, 
        required=False,
        help="Limit of the number of saved checkpoints. If limit is reached, the oldest checkpoint will be deleted."
    )
    parser.add_argument(
        "--fp16", 
        action='store_true', 
        default=False, 
        required=False,
        help="Use fp16 during training"
    )
    parser.add_argument(
        "--disable_tqdm", 
        action="store_true", 
        default=False, 
        required=False,
        help="Disable tqdm during training"
    )
#     parser.add_argument("--multitask", action="store_true", default=False, required=False)
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

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=CFG.seed)  


train = pd.read_csv(CFG.train_data_path)
valid = pd.read_csv(CFG.valid_data_path)


for col in ['REACTANT', 'REAGENT']:
    train[col] = train[col].fillna(' ')
    valid[col] = valid[col].fillna(' ')
train['input'] = 'REACTANT:' + train['REACTANT'] + 'REAGENT:' + train['REAGENT']
valid['input'] = 'REACTANT:' + valid['REACTANT'] + 'REAGENT:' + valid['REAGENT']


if CFG.debug:
    train = train[:int(len(train)/40)].reset_index(drop=True)
    valid = valid[:int(len(valid)/40)].reset_index(drop=True)
    

dataset = DatasetDict({'train': Dataset.from_pandas(train[['input', 'PRODUCT']]), 'validation': Dataset.from_pandas(valid[['input', 'PRODUCT']])})


def preprocess_function(examples):
    inputs = examples['input']
    targets = examples['PRODUCT']
    model_inputs = tokenizer(inputs, max_length=CFG.input_max_len, truncation=True)
    labels = tokenizer(targets, max_length=CFG.target_max_len, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def compute_metrics(eval_preds):
    metric = load_metric('sacrebleu')
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {'bleu': result['score']}


#load tokenizer
try: # load pretrained tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(CFG.model_name_or_path), return_tensors='pt')
except: # load pretrained tokenizer from huggingface model hub
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path, return_tensors='pt')

model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name_or_path).to(device)
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    load_from_cache_file=False
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    CFG.model,
    evaluation_strategy=CFG.evaluation_strategy,
    save_strategy=CFG.save_strategy,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size,
    weight_decay=CFG.weight_decay,
    save_total_limit=CFG.save_total_limit,
    num_train_epochs=CFG.epochs,
    predict_with_generate=True,
    fp16=CFG.fp16,
    disable_tqdm=CFG.disable_tqdm,
    push_to_hub=False,
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('./best_model')
