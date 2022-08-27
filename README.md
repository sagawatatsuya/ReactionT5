# transformer-chemical-reaction-prediciton
We trained T5 and DeBERTa on SMILES from ZINC and PubChem-10m using the task of masked-language modeling (MLM). These models can be used for the prediction of molecules' property, reaction, or interaction with proteins by changing the way of finetuning. Here, we conducted a chemical reaction prediction, where if models were given a product, they generate reactants which is nessesary for the reaction.
# How to start with
The data used for pretraining and finetuning are uploaded to google drive, and you can find them at following links. ([ZINC](https://drive.google.com/drive/folders/1lSPCqh31zxTVEhuiPde7W3rZG8kPgp-z), [PubChem-10m](https://drive.google.com/file/d/1ygYs8dy1-vxD1Vx6Ux7ftrXwZctFjpV3/view), [ORD](https://drive.google.com/file/d/1BEk2GWhNU-Azj9hm77Z2wufsPN49wN0m/view))
By executing the following command, you can download necessary data and preprocess them in one go.
```
python data-download-and-preprocess.py
```
Then, by executing the following command, you can complete preparation for model pretraining and finetuning.
```
python data-split-and-model-preparation.py
```
# Model pretraining
If you want to redo model pretraining by yourself, you can do easily.

PubChem10m-t5
```
cd /pretraining/PubChem10m-t5
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
    --save_steps="10000" \
    --num_train_epochs="30" \
    --do_train \
    --do_eval \
    --eval_steps="25000"
```

ZINC-t5
```
cd /pretraining/ZINC-t5
XLA_PYTHON_CLIENT_MEM_FRACTION=.8 python ./new_run_t5_mlm_flax.py \
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
    --save_steps="10000" \
    --num_train_epochs="30" \
    --do_train \
    --do_eval \
    --eval_steps="25000"
```

PubChem10m-deberta
```
cd /pretraining/PubChem10m-deberta
python ./run_mlm.py \
    --model_name_or_path "microsoft/deberta-base" \
    --tokenizer_name "./PubChem10m-deberta-base" \
    --num_train_epochs 30 \
    --train_file "../../data/pubchem-10m-train.json" \
    --validation_file "../../data/pubchem-10m-valid.json" \
    --per_device_train_batch_size 5 \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --output_dir "./PubChem10m-deberta-base-output" \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --eval_steps 25000 \
    --save_strategy "no" \
    --logging_steps 200 \
    --learning_rate 0.005 \
    --report_to "none" \
    --use_new_implementation \
    --load_best_model_at_end True \
    --disable_tqdm True
```


```
cd /pretraining/ZINC-deberta
python ./run_mlm.py \
    --model_name_or_path "microsoft/deberta-base" \
    --tokenizer_name "./ZINC-deberta-base" \
    --num_train_epochs 30 \
    --train_file "../../data/ZINC-train.json" \
    --validation_file "../../data/ZINC-valid.json" \
    --per_device_train_batch_size 5 \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --output_dir "./ZINC-deberta-base-output" \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --eval_steps 25000 \
    --save_strategy "no" \
    --logging_steps 200 \
    --learning_rate 0.005 \
    --report_to "none" \
    --use_new_implementation \
    --load_best_model_at_end True \
    --disable_tqdm True
```