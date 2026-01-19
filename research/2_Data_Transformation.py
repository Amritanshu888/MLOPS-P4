#!/usr/bin/env python
# coding: utf-8

# ### Data Transformation(This can also be said as a feature engineering tasks)

# In[1]:


import os
get_ipython().run_line_magic('pwd', '')


# In[2]:


os.chdir("../") ## Going back to my parent folder because this is where my project is basically starting


# In[3]:


get_ipython().run_line_magic('pwd', '')


# In[4]:


from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path


# In[5]:


from src.textSummarizer.constants import *
from src.textSummarizer.utils.common import read_yaml, create_directories


# In[6]:


class ConfigurationManager:
    def __init__(self,
                 config_path=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_transformation_config(self)->DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            tokenizer_name = config.tokenizer_name
        )
        return data_transformation_config             


# In[ ]:


import os
from src.textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk


# ### Data Transformation Component

# In[ ]:


class DataTransformation:
    ## This is just like the constructor
    def __init__(self,config:DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation = True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length=1024, truncation = True)

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }  

    def convert(self):
        ## Bcoz i need to apply this to my entire dataset, first of all i should load my dataset
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
        ## Now we will go ahead and save this to the disk
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir,"samsum_dataset"))


# In[ ]:


config = ConfigurationManager()
data_transformation_config = config.get_data_transformation_config()
data_transformation = DataTransformation(config = data_transformation_config) ## Initializing my component over here
data_transformation.convert()

