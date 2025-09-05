# GELM
GELM: Graph-based Tanimoto Similarity Grouping Pretraining and Entropy-Guided Conformer Selection Finetuning for Large Language Models

# ModuLM

# Requirements

See `modulm.yml`. Run the following command to create a new anaconda environment `modulm`: 

```bash
conda env create -f environment.yml
```

# Dataset and Backbone

* **Dataset**

* The datasets used in this project can be downloaded from the [MolTC](https://github.com/MangoKiller/MolTC/).  
All data should be placed in the `/data` folder.

* **LLM Backbone Models**

* The backbones of different Large Language Models (LLMs) can be downloaded from [Hugging Face](https://huggingface.co/).  
Please make sure the downloaded LLMs are stored in the `backbone` folder.

# Usage of GELM
* In this section, we will explain how to specifically use our ModuLM framework for training.

## Data Process
* The data processing method of GTSG in the paper has been provided in dataprocess.
* We provide dataset processing methods in the dataproces folder, including 2D molecular graph processing and 3D molecular conformation processing. You can choose different processing approaches based on your specific needs e.g..

```bash
python ZhangDDI.py
python ChChMiner.py
python ZhangDDI_3d.py
python ChChMiner_3d.py
python CombiSolv.py
python CombiSolv_3d.py
```

## Pretraining stage. Run the following script for pretraining stage on the pretrain_data dataset:
```bash
python mystage2_3d.py --root '/home/cz/consel/MolTC-main/data/Pretraining/train/' --valid_root '/home/cz/consel/MolTC-main/data/Pretraining/valid/'    --stage2_path "all_checkpoints/stage2/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 80 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 2 --save_every_n_epochs 15 --batch_size 10 --DDI True --use_3d True --caption_eval_epoch 4  --num_query_token 10  --max_len 40  --init_checkpoint  "all_checkpoints/pretrain1/last.ckpt" --devices '2,3,4,5'
```
## Fine-tune stage. Run the following script for Fine-tune stage
```bash
python mystage2_3d.py --root '/home/cz/consel/MolTC-main/data/ddi_data/Zhangddi_data_3d/train/' --valid_root '/home/cz/consel/MolTC-main/data/ddi_data/Zhangddi_data_3d/valid/'   --filename "/checkpoints/galactica/Zhang_10" --stage2_path "all_checkpoints/stage2/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 80 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 2 --save_every_n_epochs 15 --batch_size 10 --DDI True --use_3d True --caption_eval_epoch 4  --num_query_token 10 --min_len 7 --max_len 12  --init_checkpoint  "all_checkpoints/pretrain1/last.ckpt" --devices '2,3,4,5'
```




