# ReactionT5
ReactionT5 is a T5 model pretrained on large amount of chemical reactions in the Open Reaction Database (ORD). Other currently available models for chemical reaction is mostly trained on small and biased dataset (ex. patent dataset or high throughput reaction dataset created by a single reaction). These models may show high performance at benchmark dataset, but don't suit for practical usage. ORD's vast dataset allows ReactionT5 to predict both the products and yields of unseen chemical reactions with high accuracy. Whether you're working in zero-shot or few-shot scenarios, ReactionT5 provides a robust solution to your chemical reaction prediction needs.

![model image](https://github.com/sagawatatsuya/ReactionT5/blob/main/model-image.png)


In this repository, we will show how to perform product prediction and yield prediction against your dataset with ReactionT5. The pretrained models, datasets, and demo is available at [Hugging Face Hub](https://huggingface.co/sagawa).


# Installation
Reaction T5 relies on the transformers library and RDKit for the validity check of the predicted products. To install these and other necessary libraries, use the following commands:
```
pip install rdkit
pip install pytorch
pip install tokenizers==0.12.1
pip install transformers==4.21.0
pip install datasets
pip install sentencepiece==0.1.96
```


# Use ReactionT5
You can use ReactionT5 to predict the products and yields of chemical reactions.

### Product prediction
To predict the products of reactions from their inputs, use the following command. The code expects input_data as a string or CSV file that contains an 'input' column. The format of the string or contents of the column should follow this template: "REACTANT:{SMILES of reactants}REAGENT:{SMILES of reagents, catalysts, or solvents}". If there are no catalyst, reagent, or solvents, fill the blank with a space. And if there are multiple compounds as reactants or reagents, concatenate them with ".".(ex. "REACTANT:COC(=O)C1=CCCN(C)C1.O.\[Al+3].\[H-].\[Li+].\[Na+].\[OH-]REAGENT:C1CCOC1")
```
cd forward_reaction_prediction/
python prediction.py \
    --input_data="../data/forward_reaction_prediction_demo_input.csv" \
    --num_beams=5 \
    --num_return_sequences=5 \
    --batch_size=2 \
    --output_dir="./"
```

### Yield prediction
To predict the yields of reactions from their inputs, use the following command. The code expects input_data as a string or CSV file that contains an 'input' column. The format of the string or contents of the column should follow this template: "REACTANT:{SMILES of reactants}REAGENT:{SMILES of reagents, catalysts, or solvents}PRODUCT:{SMILES of products}". If there are multiple compounds, concatenate them with ".".(ex. "REACTANT:CC(C)n1ncnc1-c1cn2c(n1)-c1cnc(O)cc1OCC2.CCN(C(C)C)C(C)C.Cl.NC(=O)\[C@@H]1C\[C@H](F)CN1REAGENT: PRODUCT:O=C(NNC(=O)C(F)(F)F)C(F)(F)F")
When you run the command for the first time, you need to add the 'download_pretrained_model' argument.
```
cd yield_prediction/
python prediction.py \
    --data="../data/yield_prediction_demo_input.csv" \
    --batch_size=10 \
    --output_dir="./" \
    --download_pretrained_model
```


# Fine-tuning
You can also conduct fine-tuning of ReactionT5 on your dataset. 

### Product prediction
Specify your training and validation data used for fine-tuning and run the following command. We expect these data to contain columns named 'REACTANT', 'REAGENT', and 'PRODUCT'; each has SMILES information. If there is no reagent information, fill in the blank with ' '.
```
cd forward_reaction_prediction/
pip install sacrebleu
python finetune-pretrained-ReactionT5.py \
    --epochs=50 \
    --batch_size=32 \
    --train_data_path='your_train.csv' \
    --valid_data_path='your_validation.csv'
```

### Yield prediction

```
python finetuning.py \
    --train_data_path='/data2/sagawa/t5chem/data/C_N_yield/MFF_FullCV_01/test.csv' \
    --valid_data_path='/data2/sagawa/t5chem/data/C_N_yield/MFF_FullCV_01/train.csv' \
    --download_pretrained_model \
    --epochs=200 \
    --batch_size=6 \
    --output_dir='output/'
```

