# transformer chemical reaction prediciton
We trained T5 and DeBERTa models on SMILES from ZINC and PubChem-10m for the task of masked-language modeling (MLM). These models can be used for predicting molecules' properties, reactions, or interactions with proteins by fine-tuning them in different ways. You can download these pre-trained models [here](https://huggingface.co/sagawa). Using these pre-trained models, we created yield prediction and product prediction models. You can access them on [Hugging Face space](https://huggingface.co/sagawa).

In this repository, we will explain how to perform pre-training and fine-tuning. Then, we will explain how to use the pre-trained models and fine-tuned models.

# How to start with
To get started, you will first need to install the necessary libraries. You can use the requirements.yaml file for this purpose. If the versions of torch and jax do not match your environment, you can change and run the following command:
```
conda install -c conda-forge rdkit
conda install -c conda-forge gdown
conda install scikit-learn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tokenizers==0.12.1
pip install transformers==4.21.0
pip install datasets
pip install sentencepiece==0.1.96
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax
conda install -c conda-forge optuna
```
This will install all the necessary libraries for the project.

The original data used for pre-training and fine-tuning is uploaded to Google Drive and can be found at the following links:
・[ZINC](https://drive.google.com/drive/folders/1lSPCqh31zxTVEhuiPde7W3rZG8kPgp-z)  
・[PubChem-10m](https://drive.google.com/file/d/1ygYs8dy1-vxD1Vx6Ux7ftrXwZctFjpV3/view)  
・[ORD](https://drive.google.com/file/d/1BEk2GWhNU-Azj9hm77Z2wufsPN49wN0m/view)  
The pre-processed data is also available on [Hugging Face Hub](https://huggingface.co/sagawa) and can be used directly. 

To download the data, you can run the following command:
```
python data-download-and-preprocess.py
```
To complete the preparation for model pre-training and fine-tuning, you can run the following command:
```
python data-split-and-model-preparation.py
```
In our experiments, character-level tokenizers have shown to perform better than normal tokenizers. If you want to use character-level tokenizers, you can use the following command with the flag --use_character_level_tokenizer
```
python data-split-and-model-preparation.py --use_character_level_tokenizer
```

# Model pre-training
If you want to re-do the model pre-training yourself, you can easily do so by using the dataset uploaded to Hugging Face Hub.  
For example, to pre-train a T5 model on the PubChem-10m dataset, you can use the following command:
PubChem10m-t5
```
cd pretraining/PubChem10m-t5
python ./new_run_t5_mlm_flax.py \
    --output_dir="./PubChem10m-t5-base-output" \
    --model_type="t5" \
    --config_name="./PubChem10m-t5-base" \
    --tokenizer_name="./PubChem10m-t5-base" \
    --dataset_name "sagawa/pubchem-10m-canonicalized" \
    --max_seq_length="512" \
    --per_device_train_batch_size="5" \
    --per_device_eval_batch_size="5" \
    --adafactor \
    --learning_rate="0.005" \
    --weight_decay="0.001" \
    --warmup_steps="2000" \
    --overwrite_output_dir \
    --logging_steps="500" \
    --save_steps="100000" \
    --num_train_epochs="10" \
    --do_train \
    --do_eval \
    --eval_steps="100000"
```
Please note that if your GPU memory size is small, you may encounter an out-of-memory error during T5 pre-training. If this occurs, you can try reducing the batch size or you can try putting XLA_PYTHON_CLIENT_MEM_FRACTION=.8 before python ./new_run_t5_mlm_flax.py this reduces GPU memory preallocation.

ZINC-t5
```
cd pretraining/ZINC-t5
python ./new_run_t5_mlm_flax.py \
    --output_dir="./ZINC-t5-base-output" \
    --model_type="t5" \
    --config_name="./ZINC-t5-base" \
    --tokenizer_name="./ZINC-t5-base" \
    --dataset_name "sagawa/ZINC-canonicalized" \
    --max_seq_length="512" \
    --per_device_train_batch_size="5" \
    --per_device_eval_batch_size="5" \
    --adafactor \
    --learning_rate="0.005" \
    --weight_decay="0.001" \
    --warmup_steps="2000" \
    --overwrite_output_dir \
    --logging_steps="500" \
    --save_steps="100000" \
    --num_train_epochs="10" \
    --do_train \
    --do_eval \
    --eval_steps="100000"
```

PubChem10m-deberta
```
cd pretraining/PubChem10m-deberta
python ./run_mlm.py \
    --model_name_or_path "microsoft/deberta-base" \
    --tokenizer_name "./PubChem10m-deberta-base" \
    --num_train_epochs 10 \
    --dataset_name "sagawa/pubchem-10m-canonicalized" \
    --per_device_train_batch_size 5 \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --output_dir "./PubChem10m-deberta-base-output" \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --eval_steps 100000 \
    --save_strategy "steps" \
    --save_steps="100000" \
    --save_total_limit=2 \
    --logging_steps 500 \
    --learning_rate 0.00005 \
    --report_to "none" \
    --use_new_implementation \
    --load_best_model_at_end True \
    --disable_tqdm True
```

ZINC-deberta
```
cd pretraining/ZINC-deberta
python ./run_mlm.py \
    --model_name_or_path "microsoft/deberta-base" \
    --tokenizer_name "./ZINC-deberta-base" \
    --num_train_epochs 10 \
    --dataset_name "sagawa/ZINC-canonicalized" \
    --per_device_train_batch_size 5 \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --output_dir "./ZINC-deberta-base-output" \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --eval_steps 100000 \
    --save_strategy "steps" \
    --save_steps="100000" \
    --save_total_limit=2 \
    --logging_steps 500 \
    --learning_rate 0.00005 \
    --report_to "none" \
    --use_new_implementation \
    --load_best_model_at_end True \
    --disable_tqdm True
```

By changing **--dataset_name dataset_name** to **--train_file=train_file_path --validation_file=validation_file_path**, you can use your data for model training.

PubChem10m-t5
```
cd pretraining/PubChem10m-t5
python ./new_run_t5_mlm_flax.py \
    --output_dir="./PubChem10m-t5-base-output" \
    --model_type="t5" \
    --config_name="./PubChem10m-t5-base" \
    --tokenizer_name="./PubChem10m-t5-base" \
    --train_file="../../data/pubchem-10m-canonicalized-train.csv" \
    --validation_file="../../data/pubchem-10m-canonicalized-valid.csv" \
    --max_seq_length="512" \
    --per_device_train_batch_size="5" \
    --per_device_eval_batch_size="5" \
    --adafactor \
    --learning_rate="0.005" \
    --weight_decay="0.001" \
    --warmup_steps="2000" \
    --overwrite_output_dir \
    --logging_steps="500" \
    --save_steps="100000" \
    --num_train_epochs="10" \
    --do_train \
    --do_eval \
    --eval_steps="100000"
```

ZINC-t5
```
cd pretraining/ZINC-t5
python ./new_run_t5_mlm_flax.py \
    --output_dir="./ZINC-t5-base-output" \
    --model_type="t5" \
    --config_name="./ZINC-t5-base" \
    --tokenizer_name="./ZINC-t5-base" \
    --train_file="../../data/ZINC-canonicalized-train.csv" \
    --validation_file="../../data/ZINC-canonicalized-valid.csv" \
    --max_seq_length="512" \
    --per_device_train_batch_size="5" \
    --per_device_eval_batch_size="5" \
    --adafactor \
    --learning_rate="0.005" \
    --weight_decay="0.001" \
    --warmup_steps="2000" \
    --overwrite_output_dir \
    --logging_steps="500" \
    --save_steps="100000" \
    --num_train_epochs="10" \
    --do_train \
    --do_eval \
    --eval_steps="100000"
```

PubChem10m-deberta
```
cd pretraining/PubChem10m-deberta
python ./run_mlm.py \
    --model_name_or_path "microsoft/deberta-base" \
    --tokenizer_name "./PubChem10m-deberta-base" \
    --num_train_epochs 10 \
    --train_file "../../data/pubchem-10m-canonicalized-train.csv" \
    --validation_file "../../data/pubchem-10m-canonicalized-valid.csv" \
    --per_device_train_batch_size 5 \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --output_dir "./PubChem10m-deberta-base-output" \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --eval_steps 100000 \
    --save_strategy "steps" \
    --save_steps="100000" \
    --logging_steps 500 \
    --learning_rate 0.00005 \
    --report_to "none" \
    --use_new_implementation \
    --load_best_model_at_end True \
    --disable_tqdm True
```

ZINC-deberta
```
cd pretraining/ZINC-deberta
python ./run_mlm.py \
    --model_name_or_path "microsoft/deberta-base" \
    --tokenizer_name "./ZINC-deberta-base" \
    --num_train_epochs 10 \
    --train_file "../../data/ZINC-canonicalized-train.csv" \
    --validation_file "../../data/ZINC-canonicalized-valid.csv" \
    --per_device_train_batch_size 5 \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --output_dir "./ZINC-deberta-base-output" \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --eval_steps 100000 \
    --save_strategy "steps" \
    --save_steps="100000" \
    --logging_steps 500 \
    --learning_rate 0.00005 \
    --report_to "none" \
    --use_new_implementation \
    --load_best_model_at_end True \
    --disable_tqdm True
```

# Model finetuning

### Yield prediction
You can predict yields of chemical reactions from their inputs (reactants, products, and catalysts). By executing following command, you can finetune yield prediction 
```
cd yield_prediction/
python train.py     
    --data_path='all_ord_reaction_uniq_with_attr_v3.tsv'
    --pretrained_model_name_or_path='sagawa/ZINC-t5'
    --model='t5'
    --epochs=100
    --batch_size=50
    --max_len=400
    --num_workers=4
    --weight_decay=0.05
    --gradient_accumulation_steps=1
    --batch_scheduler
    --print_freq=100
    --output_dir='./'
```

### Product prediction
You can predict products of reactions only from reactants. However, we found inputting reactants, catalysts, reagents, solvents, and NoData(their classification is unknown) can achive better results. By execting following command, you can do multi-input product prediction.
```
cd forward_reaction_prediction/
python train.py \
    --model='deberta' \
    --epochs=20 \
    --lr=1e-7 \
    --batch_size=16 \
    --max_len=512 \
    --weight_decay=0.01 \
    --evaluation_strategy='epoch' \
    --save_strategy='epoch' \
    --logging_strategy='epoch' \
    --save_total_limit=1 \
    --data_path='all_ord_reaction_uniq_with_attr_v3.tsv' \
    --disable_tqdm \
    --pretrained_model_name_or_path='sagawa/ZINC-deberta'
```

### Retrosynthesis
Use dataset uplaoded to Hugging Face Hub.
```
cd retrosynthesis/
python train.py \
    --model='t5' \
    --epochs=10 \
    --lr=2e-5 \
    --batch_size=4 \
    --max_len=256 \
    --weight_decay=0.01 \
    --evaluation_strategy='epoch' \
    --save_strategy='epoch' \
    --logging_strategy='epoch' \
    --save_total_limit=3 \
    --train \
    --dataset_name='sagawa/ord-uniq-canonicalized' \
    --disable_tqdm \
    --pretrained_model_name_or_path='sagawa/ZINC-t5' \
    --multitask \
    --shuffle_augmentation=1 \
    --noncanonical_augmentation=1
```
Use local dataset.
```
cd retrosynthesis/
python train.py \
    --model='t5' \
    --epochs=10 \
    --lr=2e-5 \
    --batch_size=4 \
    --max_len=256 \
    --weight_decay=0.01 \
    --evaluation_strategy='epoch' \
    --save_strategy='epoch' \
    --logging_strategy='epoch' \
    --save_total_limit=3 \
    --train \
    --data_path='../data/' \
    --disable_tqdm \
    --pretrained_model_name_or_path='sagawa/ZINC-t5' \
    --multitask \
    --shuffle_augmentation=1 \
    --noncanonical_augmentation=1
```



# Use finetuned models
We finetuned pretrained-models and uploaded to Hugging Face Hub. So you can immediately use them for predicting products or yields against your data. 

### Product prediction
To predict the products of reactions from their inputs, you can use the following command. The code expects input_data as a string or CSV file that contains an 'input' column. The format of the string or contents of the column are like "REACTANT:{reactants of the reaction}CATALYST:{catalysts of the reaction}REAGENT:{reagents of the reaction}SOLVENT:{solvent of the reaction}NoData:{uncategorized compounds of the reaction}". If there are no catalyst or reagent, fill the blank with a space. And if there are multiple reactants, concatenate them with ".".(ex. "REACTANT:NCCO.O=C1COCC(=O)O1CATALYST: REAGENT: SOLVENT:c1ccncc1NoData: ")

```
cd forward_reaction_prediction/
python prediction.py \
    --input_data="demo_input.csv" \
    --model_name_or_path="sagawa/ZINC-t5-productpredicition" \
    --model="t5" \
    --num_beams=5 \
    --num_return_sequences=5
```

### Yield prediction
To predict the yields of reactions from their inputs, you can use the following command. The code expects input_data as a string or CSV file that contains an 'input' column. The format of the string or contents of the column are like "REACTANT:{reactants of the reaction}PRODUCT:{products of the reaction}". If there are multiple reactants, concatenate them with ".".(ex. "REACTANT:NCCO.O=C1COCC(=O)O1CATALYST: REAGENT: SOLVENT:c1ccncc1NoData: ")
```
cd yield_prediction/
python prediction.py \
    --data="demo_input.csv" \
    --model="t5" \
    --batch_size=10 \
    --output_dir="./" \
    --download_pretrained_model
```

