# ReactionT5
![training_procedure_image](https://github.com/sagawatatsuya/ReactionT5/blob/main/study_reproduction/training-procedure.png)
We will explain how to reproduce our paper's results(compound pretraining, reaction pretraining, and restoration of uncategorized data ). 


# Installation
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

# Compound pretraining
Run the following command to conduct compound pretraining. In compound pretraining, T5 is trained on the ZINC dataset using span-masked language modeling. The pretraine model (CompoundT5) is uploaded to [Hugging Face Hub](https://huggingface.co/sagawa/CompoundT5).
```
cd compound_pretraining/CompoundT5
sh run.sh
```
Please note that if your GPU memory size is small, you may encounter an out-of-memory error during T5 pre-training. If this occurs, you can try reducing the batch size or you can try putting XLA_PYTHON_CLIENT_MEM_FRACTION=.8 before python ./new_run_t5_mlm_flax.py this reduces GPU memory preallocation.


# Restore uncategorized data
Many reactions in ORD involve uncategorized compounds, indicating the presence of compounds with unidentified roles in these reactions. To fully utilize ORD's data, we trained ClassificationT5 which classifies uncategorized compounds as reactants or reagetns.

create train data from the ORD dataset in the following notebook
```
cd compound_classification
create_dataset.ipynb
```

train ClassificationT5
```
python train.py
```

predict and restore uncategorized data
```
python nodata-prediction.py
python create-file-from-prediction.py
```


# Reaction pretraining

### Yield prediction
```
cd ../yield_prediction/
python train.py     
    --data_path='all_ord_reaction_uniq_with_attr_v3.tsv'
    --pretrained_model_name_or_path='sagawa/CompoundT5'
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
cd ../forward_reaction_prediction/
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
