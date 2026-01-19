#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
get_ipython().run_line_magic('pwd', '')


# In[5]:


os.chdir("../")


# In[6]:


get_ipython().run_line_magic('pwd', '')


# In[7]:


from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelTrainerConfig:
    ## Here we have to read all the params in config.yaml(for model trainer) and in params.yaml for model training parameters, hence we have to make those many variables
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int


# In[8]:


from src.textSummarizer.constants import *
from src.textSummarizer.utils.common import read_yaml, create_directories


# In[9]:


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer  ## In utils/common the function has return type as ConfigBox, when u have such return type u can read everything directly by using keys like : .
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_ckpt = config.model_ckpt,
            num_train_epochs = params.num_train_epochs,
            warmup_steps = params.warmup_steps,
            per_device_train_batch_size = params.per_device_train_batch_size,
            weight_decay = params.weight_decay,
            logging_steps = params.logging_steps,
            evaluation_strategy = params.evaluation_strategy,
            eval_steps = params.save_steps,
            gradient_accumulation_steps = params.gradient_accumulation_steps
        )
        return model_trainer_config


# In[ ]:


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk


# In[ ]:


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)   ## Instead of directly giving the model name i will use self.config.model_ckpt
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model = model_pegasus)

        ## Loading the Data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir = self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16 
        ) ## This is the hardcoded way, if u don't want to use this u can also use : self.config

        trainer = Trainer(model=model_pegasus, args=trainer_args,
                          tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                          train_dataset=dataset_samsum_pt["test"],
                          eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        ## Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))


# - !pip install --upgrade accelerate
# - !pip uninstall -y transformers accelerate
# - !pip install transformers accelerate
# 
# - Do this if it throws an error
# - Then restart the kernel, execute every code again from starting to beginning.

# In[ ]:


config = ConfigurationManager()
model_trainer_config = config.get_model_trainer_config()
model_trainer_config = ModelTrainer(config=model_trainer_config)
model_trainer_config.train()

