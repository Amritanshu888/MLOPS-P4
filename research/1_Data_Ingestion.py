#!/usr/bin/env python
# coding: utf-8

# # Data Ingestion Module

# In[2]:


import os
get_ipython().run_line_magic('pwd', ' ## See present working directory')


# In[3]:


os.chdir("../") ## Going to my parent directory
get_ipython().run_line_magic('pwd', '## Now if u execute this u will get to see ur present directory as MLOPS-P4 --> Where ur project is getting created')


# ### Basic Configuration

# In[4]:


from dataclasses import dataclass
from pathlib import Path

## The reason why we are creating dataclasses ?? --> Bcoz output fields(data ingestion output) in artifacts are already defined in the config.yaml file

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path
## I have created a dataclass with all these particular variables    


# In[5]:


## There are some functionality that we need to write for every component that we define ---> For this we use Configuration Manager
from src.textSummarizer.constants import *
from src.textSummarizer.utils.common import read_yaml, create_directories


# ## Configuration Updates

# In[7]:


class ConfigurationManager:
    def __init__(self,
                 config_path=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        return data_ingestion_config   


# In[8]:


import os
import urllib.request as request
import zipfile
from src.textSummarizer.logging import logger


# ## Components

# In[12]:


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"File is downloaded")
        else:
            logger.info(f"File already exists")

    ## Another functionality is extracting the zip file
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)              


# In[13]:


config = ConfigurationManager()
data_ingestion_config = config.get_data_ingestion_config()
data_ingestion = DataIngestion(config = data_ingestion_config) ## Initializing our component

data_ingestion.download_file()
data_ingestion.extract_zip_file()
## Note: params.yaml file should not be empty, so initially enter : "key":"value", later we will update this.

