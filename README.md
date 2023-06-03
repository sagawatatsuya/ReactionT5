# ReactionT5
ReactionT5 is a T5 model pretrained on large amount of chemical reactions in the Open Reaction Database (ORD). Other currently available models for chemical reaction is mostly trained on small and biased dataset (ex. patent dataset or high throughput reaction dataset created by a single reaction). These models may show high performance at benchmark dataset, but don't suit for practical usage. Reaction has high generalizability and provides good performance in forward reaction prediction (product prediction) and yield prediction against unseen reactions.

![model image](https://github.com/sagawatatsuya/ReactionT5/blob/main/model-image.png)


In this repository, we will show how to perform product prediction and yield prediction against your dataset with ReactionT5. The pretrained models, datasets, and demo is available at [Hugging Face Hub](https://huggingface.co/sagawa)


# Installation
Reaction T5 is based on transformers libraly. So you first need to install related libraries. In addition, RDKit is necessary for checking predictions' invalidity. 
```
conda install -c conda-forge rdkit
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tokenizers==0.12.1
pip install transformers==4.21.0
pip install datasets
pip install sentencepiece==0.1.96
```


# Use ReactionT5


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

