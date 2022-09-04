import os
import json
import shutil
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import tokenizers
import transformers
from transformers import AutoTokenizer, T5Config
import datasets
from datasets import load_dataset
import sentencepiece
import argparse

# https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/t5_tokenizer_model.py
#!/usr/bin/env python3
import json
from typing import Iterator, List, Union
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing


class SentencePieceUnigramTokenizer(BaseTokenizer):
    """
    This class is a copy of `DeDLOC's tokenizer implementation <https://github.com/yandex-research/DeDLOC/blob/main/sahajbert/tokenizer/tokenizer_model.py>`__ .

    Custom SentencePiece Unigram Tokenizer with NMT, NKFC, spaces and lower-casing characters normalization
    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(
        self,
        replacement: str = "▁",
        add_prefix_space: bool = True,
        unk_token: Union[str, AddedToken] = "<unk>",
        eos_token: Union[str, AddedToken] = "</s>",
        pad_token: Union[str, AddedToken] = "<pad>",
    ):
        self.special_tokens = {
            "pad": {"id": 0, "token": pad_token},
            "eos": {"id": 1, "token": eos_token},
            "unk": {"id": 2, "token": unk_token},
        }

        self.special_tokens_list = [None] * len(self.special_tokens)
        for token_dict in self.special_tokens.values():
            self.special_tokens_list[token_dict["id"]] = token_dict["token"]

        tokenizer = Tokenizer(Unigram())

        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Nmt(),
                normalizers.NFKC(),
                normalizers.Replace(Regex(" {2,}"), " "),
#                 normalizers.Lowercase(),
            ]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.Punctuation(),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

        tokenizer.post_processor = TemplateProcessing(
            single=f"$A {self.special_tokens['eos']['token']}",
            special_tokens=[(self.special_tokens["eos"]["token"], self.special_tokens["eos"]["id"])],
        )

        parameters = {
            "model": "SentencePieceUnigram",
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given files"""

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )

        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

        self.add_unk_id()

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given iterator"""

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )

        self._tokenizer.train_from_iterator(iterator, trainer=trainer)

        self.add_unk_id()

    def add_unk_id(self):
        tokenizer_json = json.loads(self._tokenizer.to_str())

        tokenizer_json["model"]["unk_id"] = self.special_tokens["unk"]["id"]

        self._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dataset", action="store_true", default=False, required=False)
    
    return parser.parse_args()

CFG = parse_args()


if CFG.load_dataset:
    
    # PubChem10m data

    # Initialize a dataset
    dataset = load_dataset('sagawa/pubchem-10m-canonicalized')
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation']])
    training_corpus = (
        dataset[i : i + 1000]['smiles']
        for i in range(0, len(dataset), 1000)
    )

    # Train tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 1000)

    # Save files to disk
    tokenizer.save_pretrained('./pretraining/PubChem10m-deberta/PubChem10m-deberta-base')


    # ZINC data

    # Initialize a dataset
    dataset = load_dataset('sagawa/ZINC-canonicalized')
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation']])
    training_corpus = (
        dataset[i : i + 1000]['smiles']
        for i in range(0, len(dataset), 1000)
    )
    
    # Train tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 1000)

    # Save files to disk
    tokenizer.save_pretrained('./pretraining/ZINC-deberta/ZINC-deberta-base')


    # PubChem10m data

    vocab_size = 32_000
    input_sentence_size = None

    # Initialize a dataset
    dataset = load_dataset('sagawa/pubchem-10m-canonicalized')
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation']])

    tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    # Build an iterator over this dataset
    def batch_iterator(input_sentence_size=None):
        if input_sentence_size is None:
            input_sentence_size = len(dataset)
        batch_length = 1000
        for i in range(0, input_sentence_size, batch_length):
            yield dataset[i: i + batch_length]['smiles']

    # Train tokenizer
    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=input_sentence_size),
        vocab_size=vocab_size,
        show_progress=True,
    )

    # Save files to disk
    tokenizer.save('./pretraining/PubChem10m-t5/PubChem10m-t5-base/tokenizer.json')

    config = T5Config.from_pretrained('google/t5-v1_1-base', vocab_size=tokenizer.get_vocab_size())
    config.save_pretrained('./pretraining/PubChem10m-t5/PubChem10m-t5-base/')


    # ZINC data

    vocab_size = 32_000
    input_sentence_size = None

    # Initialize a dataset
    dataset = load_dataset('sagawa/ZINC-canonicalized')
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation']])

    tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    # Build an iterator over this dataset
    def batch_iterator(input_sentence_size=None):
        if input_sentence_size is None:
            input_sentence_size = len(dataset)
        batch_length = 1000
        for i in range(0, input_sentence_size, batch_length):
            yield dataset[i: i + batch_length]['smiles']

    # Train tokenizer
    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=input_sentence_size),
        vocab_size=vocab_size,
        show_progress=True,
    )

    # Save files to disk
    tokenizer.save('./pretraining/ZINC-t5/ZINC-t5-base/tokenizer.json')

    config = T5Config.from_pretrained('google/t5-v1_1-base', vocab_size=tokenizer.get_vocab_size())
    config.save_pretrained('./pretraining/ZINC-t5/ZINC-t5-base/')
    
else:

    # ORD data
    # Prepare train, validation and test data
    all = pd.read_csv('./data/all_ord_reaction_uniq_canonicalized.csv')

    train, test = train_test_split(all, test_size=int(len(all)*0.1))
    train, val = train_test_split(train, test_size=int(len(all)*0.1))

    train.to_csv('./data/all_ord_reaction_uniq_canonicalized-train.csv', index=False)
    val.to_csv('./data/all_ord_reaction_uniq_canonicalized-valid.csv', index=False)
    test.to_csv('./data/all_ord_reaction_uniq_canonicalized-test.csv', index=False)


    # PubChem10m data

    # Initialize a dataset
    dataset = load_dataset('csv',data_files='./data/pubchem-10m-canonicalized.csv')
    training_corpus = (
        dataset['train'][i : i + 1000]['smiles']
        for i in range(0, len(dataset['train']), 1000)
    )

    # Train tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 1000)

    # Save files to disk
    tokenizer.save_pretrained('./pretraining/PubChem10m-deberta/PubChem10m-deberta-base')

    # Split into train and validation data
    all = pd.read_csv('./data/pubchem-10m-canonicalized.csv')
    train, valid = train_test_split(all, test_size=0.1)

    # Save train and validation data
    train.to_csv('./data/pubchem-10m-canonicalized-train.csv', index=False)
    valid.to_csv('./data/pubchem-10m-canonicalized-valid.csv', index=False)

    with open('./data/pubchem-10m-train.json', 'w') as fp:
        for smiles in train.smiles:
            fp.write(json.dumps({'text':smiles})+'\n')

    with open('./data/pubchem-10m-valid.json', 'w') as fp:
        for smiles in valid.smiles:
            fp.write(json.dumps({'text':smiles})+'\n')


    # ZINC data

    ZINC = pd.read_csv('./data/ZINC-canonicalized.csv')
    ZINC.columns = ['text']
    ZINC.to_csv('./data/ZINC-canonicalized.csv')
    # Initialize a dataset
    dataset = load_dataset('csv',data_files='./data/ZINC-canonicalized.csv')
    training_corpus = (
        dataset["train"][i : i + 1000]['text']
        for i in range(0, len(dataset["train"]), 1000)
    )

    # Train tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 1000)

    # Save files to disk
    tokenizer.save_pretrained('./pretraining/ZINC-deberta/ZINC-deberta-base')

    # Split into train and validation data
    all = pd.read_csv('./data/ZINC-canonicalized.csv')
    train, valid = train_test_split(all, test_size=0.1)

    # Save train and validation data
    train.to_csv('./data/ZINC-canonicalized-train.csv', index=False)
    valid.to_csv('./data/ZINC-canonicalized-valid.csv', index=False)

    with open("./data/ZINC-train.json", "w") as fp:
        for smiles in train.text:
            fp.write(json.dumps({"text":smiles})+'\n')

    with open("./data/ZINC-valid.json", "w") as fp:
        for smiles in valid.text:
            fp.write(json.dumps({"text":smiles})+'\n')

    # PubChem10m data

    vocab_size = 32_000
    input_sentence_size = None

    # Initialize a dataset
    dataset = load_dataset('csv',data_files='./data/pubchem-10m-canonicalized.csv')

    tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    # Build an iterator over this dataset
    def batch_iterator(input_sentence_size=None):
        if input_sentence_size is None:
            input_sentence_size = len(dataset)
        batch_length = 1000
        for i in range(0, input_sentence_size, batch_length):
            yield dataset["train"][i: i + batch_length]['smiles']

    # Train tokenizer
    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=input_sentence_size),
        vocab_size=vocab_size,
        show_progress=True,
    )

    # Save files to disk
    tokenizer.save('./pretraining/PubChem10m-t5/PubChem10m-t5-base/tokenizer.json')

    config = T5Config.from_pretrained('google/t5-v1_1-base', vocab_size=tokenizer.get_vocab_size())
    config.save_pretrained('./pretraining/PubChem10m-t5/PubChem10m-t5-base/')


    # ZINC data

    vocab_size = 32_000
    input_sentence_size = None

    # Initialize a dataset
    dataset = load_dataset('csv',data_files='./data/ZINC-canonicalized.csv')

    tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    # Build an iterator over this dataset
    def batch_iterator(input_sentence_size=None):
        if input_sentence_size is None:
            input_sentence_size = len(dataset)
        batch_length = 1000
        for i in range(0, input_sentence_size, batch_length):
            yield dataset["train"][i: i + batch_length]['text']

    # Train tokenizer
    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=input_sentence_size),
        vocab_size=vocab_size,
        show_progress=True,
    )

    # Save files to disk
    tokenizer.save('./pretraining/ZINC-t5/ZINC-t5-base/tokenizer.json')

    config = T5Config.from_pretrained('google/t5-v1_1-base', vocab_size=tokenizer.get_vocab_size())
    config.save_pretrained('./pretraining/ZINC-t5/ZINC-t5-base/')

