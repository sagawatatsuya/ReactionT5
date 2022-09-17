# transformer-chemical-reaction-prediciton
We trained T5 and DeBERTa on SMILES from ZINC and PubChem-10m using the task of masked-language modeling (MLM). These models can be used for the prediction of moelcules' properties, reactions, or interactions with proteins by changing the way of finetuning. You can download these pretrained models [here](https://huggingface.co/sagawa). Using these pretrained models, we conducted a chemical reaction prediction, where if models were given a product, they generate reactants which is necessary for the reaction.
# How to start with
The original data used for pretraining and finetuning are uploaded to google drive, and you can find them at following links. ([ZINC](https://drive.google.com/drive/folders/1lSPCqh31zxTVEhuiPde7W3rZG8kPgp-z), [PubChem-10m](https://drive.google.com/file/d/1ygYs8dy1-vxD1Vx6Ux7ftrXwZctFjpV3/view), [ORD](https://drive.google.com/file/d/1BEk2GWhNU-Azj9hm77Z2wufsPN49wN0m/view))
The preprocessed data is uploaded to [Hugging Face Hub](https://huggingface.co/sagawa) and we can use them, but if you want to download the data to you local directory, you can do by executing the following command, .
```
python data-download-and-preprocess.py
```
Then, by executing the following command, you can complete preparation for model pretraining and finetuning. 
```
python data-split-and-model-preparation.py --load_dataset
```
If you want to use your local data, remove **--load_dataset**.
```
python data-split-and-model-preparation.py
```

# Model pretraining
If you want to redo model pretraining by yourself, you can do easily.
(If your GPU memory size is small, you may hit out of memory error during t5 pretraining. And if you can't solve it by reducing batch_size, try putting **XLA_PYTHON_CLIENT_MEM_FRACTION=.8** before **python ./new_run_t5_mlm_flax.py**. This reduces GPU memory preallocation.)

Use dataset uploaded to Hugging Face Hub
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
    --save_steps="10000" \
    --num_train_epochs="30" \
    --do_train \
    --do_eval \
    --eval_steps="25000"
```

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
    --save_steps="10000" \
    --num_train_epochs="30" \
    --do_train \
    --do_eval \
    --eval_steps="25000"
```

PubChem10m-deberta
```
cd pretraining/PubChem10m-deberta
python ./run_mlm.py \
    --model_name_or_path "microsoft/deberta-base" \
    --tokenizer_name "./PubChem10m-deberta-base" \
    --num_train_epochs 30 \
    --dataset_name "sagawa/pubchem-10m-canonicalized" \
    --per_device_train_batch_size 5 \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --output_dir "./PubChem10m-deberta-base-output" \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --eval_steps 25000 \
    --save_strategy "steps" \
    --save_steps="25000" \
    --logging_steps 500 \
    --learning_rate 0.005 \
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
    --num_train_epochs 30 \
    --dataset_name "sagawa/ZINC-canonicalized" \
    --per_device_train_batch_size 5 \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --output_dir "./ZINC-deberta-base-output" \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --eval_steps 25000 \
    --save_strategy "steps" \
    --save_steps="25000" \
    --logging_steps 500 \
    --learning_rate 0.005 \
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
    --save_steps="10000" \
    --num_train_epochs="30" \
    --do_train \
    --do_eval \
    --eval_steps="25000"
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
    --save_steps="10000" \
    --num_train_epochs="30" \
    --do_train \
    --do_eval \
    --eval_steps="25000"
```

PubChem10m-deberta
```
cd pretraining/PubChem10m-deberta
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
    --save_strategy "steps" \
    --save_steps="25000" \
    --logging_steps 500 \
    --learning_rate 0.005 \
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
    --save_strategy "steps" \
    --save_steps="25000" \
    --logging_steps 500 \
    --learning_rate 0.005 \
    --report_to "none" \
    --use_new_implementation \
    --load_best_model_at_end True \
    --disable_tqdm True
```

# Hyperparameter tuning for model finetuning
Before finetuning, you can tune hyperparameters(ex. learning rate, weight decay) that affect final results.
You can use local pretrained models or pretrained models at Hugging Face Hub by specifying the **pretrained_model_name_or_path**.

Use dataset uplaoded to Hugging Face Hub.
```
cd hyperparameter-tuning/
python hp_tuning.py \
    --model='t5' \
    --output_dir='output' \
    --batch_size=4 \
    --max_len=256 \
    --epochs=3 \
    --n_trials=10 \
    --evaluation_strategy='epoch' \
    --logging_strategy='epoch' \
    --dataset_name='sagawa/ord-uniq-canonicalized' \
    --pretrained_model_name_or_path='sagawa/ZINC-t5' \
    --multitask \
    --shuffle_augmentation=1 \
    --noncanonical_augmentation=1
```
Use local dataset.
```
cd hyperparameter-tuning/
python hp_tuning.py \
    --model='t5' \
    --output_dir='output' \
    --batch_size=4 \
    --max_len=256 \
    --epochs=3 \
    --n_trials=10 \
    --evaluation_strategy='epoch' \
    --logging_strategy='epoch' \
    --data_path='../data/' \
    --pretrained_model_name_or_path='sagawa/ZINC-t5' \
    --multitask \
    --shuffle_augmentation=1 \
    --noncanonical_augmentation=1
```


# Model finetuning
In this finetuning, we load pretrained model's weight to a Seq2Seq model and train it to generate reactants from given products.
T5 is encoder-decoder model, so we can use them as Seq2Seq model directly. However, DeBERTa is encoder model, and its decoder is not available, so we use RoBERTa's decoder instead.

Use dataset uplaoded to Hugging Face Hub.
```
cd finetuning/
python finetuning.py \
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
cd finetuning/
python finetuning.py \
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