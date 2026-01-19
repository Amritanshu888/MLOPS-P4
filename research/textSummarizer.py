#!/usr/bin/env python
# coding: utf-8

# - nvidia-smi --> library needs to be installed
# - Model which we are using : google/pegasus-cnn_dailymail --> for my text summarization task and we will be using samsun data(Samsung data that we have) on huggingface.
# - This data has 3 important fields : 1. Dialogue: Text of dialogue 2. Summary: human written summary of the dialogue 3. id: unique id of an example

# - Whenever we talk abt this google/pegasus-cnn_dailymail model --> This is based on a sequence to sequence model, so for sequence to sequence model u specifically require a tokenizer. When u click Use this model(in black available on right hand side) -> u will get the codes u need to execute this model.

# - rogue_score will be the performance metrics to calculate for this particular model
# - Other libraries required: accelerate, transformers accelerate(These libraries are provided by huggingface itself).
# - We specifically use the above libraries to make sure that we assign all the jobs in a better way to the GPUs when our training specifically happens. That is the reason why we specifically use accelerate.
# 
# ## Purpose of accelerate:
# 1. Ease of Multi-Device Training: Whether you're using multiple GPUs or TPUs, accelerate makes it easier to scale your model across devices with minimal code changes.
# 2. Mixed Precision: It allows models to be trained using mixed precision, which can speed up training and reduce memory usage.
# 3. Zero Redundancy Optimizer (ZeRO): Helps manage large models by efficiently splitting the model across multiple devices.
# 4. Offload to CPU/SSD: Useful for large models that may not fit entirely into GPU memory, by allowing parts of the model or optimizer to be offloaded to CPU or even SSD.

# In[1]:


from transformers import pipeline, set_seed
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
# from datasets import load_dataset, load_metric

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer ## AutoTokenizer is a library which is specifically used to convert ur text into tokens(So for every model that is available in huggingface there will be an AutoTokenizer which will be compatible to that model and its responsibility is to basically convert ur text into token).
## AutoModelForSeq2SeqLM ---> This is basically used just to load the particular model u are specifically using
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
import torch



# ## Basic Functionality of Huggingface Model

# In[2]:


from transformers import AutoTokenizer, PegasusForConditionalGeneration

model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum") ## Loading the model  1st we are loading the model
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")  ## Converting words into tokens    2nd we are loading the tokenizer

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousands customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tommorow."
) ## This is my article
inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt") ## Applying tokenizer to article summary, we are returning the tensors in pt format
## Above we are converting the articles into tokens
# Generate Summary
summary_ids = model.generate(inputs["input_ids"])  ## Generating the summarizer for the text above, it will generate something called as "input_ids", generating based on the tokens
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]  ## Converting back from my ids to my text(Converting back from tokens to text)
## skip_special_tokens=True will remove all the unnecessary tokens that are present inside like the space token or the clear token


# In[3]:


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu" ## If torch.cuda is available or not --> it will display whether we are going to use GPU or not
device


# # Fine Tuning

# ## This will give us the idea that how can we do finetuning on any kind of model and for any kind of dataset that we have
# 

# ## Our main aim here is to finetune with the custom data

# In[15]:


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers


# Load model with explicit configuration
model_name = "google/pegasus-cnn_dailymail"

# Try loading with different parameters
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # Force slow tokenizer --> ## Main work is to convert the text into tokens
    model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device) ## Here we are basically loading the model
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
## This code is used just to load the model and the tokenizer that is used for this particular model that is "pegasus-cnn_dailymail"    


# In[15]:


## download & unzip data
import requests
import zipfile
import os
from tqdm import tqdm
import io

# Download the file
url = "https://github.com/krishnaik06/datasets/raw/refs/heads/main/summarizer-data.zip"
print("Downloading dataset...")

# Download with progress bar
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open("summarizer-data.zip", "wb") as file:
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            pbar.update(len(data))

print("Download complete!")

# Extract the zip file
print("Extracting files...")
with zipfile.ZipFile("summarizer-data.zip", 'r') as zip_ref:
    zip_ref.extractall()
    
print("Extraction complete!")


# In[5]:


dataset_samsum = load_from_disk('samsum_dataset')  ## If u use this load_from_disk function it will load that data and convert it into a dictionary
dataset_samsum


# In[6]:


split_lengths = [len(dataset_samsum[split]) for split in dataset_samsum] ## Here in this code we are trying to understand how much is the length of train data, test data and validation data.

print(f"Split lengths: {split_lengths}")
print(f"Features: {dataset_samsum['train'].column_names}")
print("\nDialogue:")
## There are 3 features : dialogue, id and summary in both train and test
print(dataset_samsum["test"][1]["dialogue"])

print("\nSummary")

print(dataset_samsum["test"][1]["summary"])


# In[7]:


## See the same for next test data
split_lengths = [len(dataset_samsum[split]) for split in dataset_samsum] ## Here in this code we are trying to understand how much is the length of train data, test data and validation data.

print(f"Split lengths: {split_lengths}")
print(f"Features: {dataset_samsum['train'].column_names}")
print("\nDialogue:")
## There are 3 features : dialogue, id and summary in both train and test
print(dataset_samsum["test"][2]["dialogue"]) ## For next test data

print("\nSummary")

print(dataset_samsum["test"][2]["summary"]) # For next test data


# ## Preparing Data For Training For Sequence To Sequence Model
# - {
#     'dialogue': "Hi! How are you?",
#     'summary': "The speaker is asking how the other person is."
# }
# - This above data needs to be converted into 3 main fields which are given below.
# - Here u have input_ids(which is token ids for the dialogue), whatever input tokenizer we are using it will convert ur words into tokens. -> This is required for the training of any sequence to sequence model.
# - Then we have attention mask which is used to apply some special characters within this word in the form of tokens -> which will be useful for the sequence to sequence model to implement or to do the prediction and finetune the model.
# - Third parameter that we have is labels : It is basically Token ID for the summar('target' feature). For 'summary'(target feature) we will apply the tokenizer and convert these words into this particular tokens.
# - This has to be done before i pass my data to Sequence to Sequence Model.
# 
# - {
#     'input_ids': [123,456,789, ...], # Token IDs for the dialogue
#     'attention_mask': [1,1,1, ...], # Attention mask for the input
#     'labels': [321,654,987, ...], # Token IDs for the summary (target)
# }

# In[16]:


def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['dialogue'], max_length = 1024, truncation = True) ## To convert into input encodings i will be using a tokenizer

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['summary'], max_length = 128, truncation = True) ## Also converting our summary into tokens, but here we have to use this tokenizer as a target tokenizer for that we use the function as_target_tokenizer()

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    } ## Here we are basically returning our 'input_ids', 'attention_mask' and 'labels'


# In[9]:


## Now the above particular function i can apply to my entire dataset of samsum which is in the form of dictionary
dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched = True)


# In[ ]:


dataset_samsum_pt['train'] ## Now here u can see that in features i have more additional fields : 'input_ids', 'attention_mask', 'labels'
## These have also been added to my dataset. -> The new features are really important for training purpose


# In[11]:


dataset_samsum_pt['test']


# - DataCollatorForSeq2Seq is a special data collator designed for sequence-to-sequence models(e.g., Pegasus,T5,BART) that helps in preparing batches of data for training.

# In[ ]:


## Training

from transformers import DataCollatorForSeq2Seq  ## When we are training a sequence to sequence model we need to use this DataCollator
## What it does is that : This make sures that whatever data we specifically have it tries to convert that into batch so that it can be provided to the model for the training purpose

seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)


# In[18]:


from transformers import TrainingArguments, Trainer
## In order to finetune our data: Initially we prepared it by adding additional features above
trainer_args = TrainingArguments(
    output_dir = 'pegasus-samsum', num_train_epochs=1, warmup_steps=500,  ## Reason why number of epochs we have kept 1 is that it is a very huge dataset
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    weight_decay=0.01, logging_steps=10,
    evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
    gradient_accumulation_steps=16
)


# In[ ]:


trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["test"], ## Instead of givng train_dataset i m giving test_dataset as its smaller that the train dataset
                  eval_dataset=dataset_samsum_pt["validation"]) ## Evaluation Dataset i will set to validation


# In[ ]:


trainer.train()


# In[ ]:


## Evaluation
### 1st[1,2,3,4,5,6] -> [1,2,3][4,5,6]
def generate_batch_sized_chunks(list_of_elements, batch_size):
    """split the dataset into smaller batches that we can process simultaneously
    Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i: i+batch_size]

def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                                batch_size=16, device=device,
                                column_text="article",
                                column_summary="highlights"):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches), total = len(article_batches)):

        inputs = tokenizer(article_batch, max_length=1024, truncation=True,
                           padding="max_length", return_tensors="pt")
        
        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                   attention_mask=inputs["attention_mask"].to(device),
                                   length_penalty=0.8, num_beams=8, max_length=128)
        '''parameter for length penalty ensures that the model does not generate sequences that are too long.'''

        # Finally, we decode the generated texts,
        # replace the token, and add the decoded texts with the reference to the metric.
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization=True) for s in summaries]
        decoded_summaries = [d.replace(""," ") for d in decoded_summaries]

        metric.add_batch(predictions = decoded_summaries, references=target_batch)  ## Whatever decoded summaries we have and our target batch will get compared

    # Finally compute and return the ROGUE scores.
    score = metric.compute()
    return score    


# In[ ]:


## !pip install evaluate


# In[ ]:


import evaluate

rouge_metric = evaluate.load('rouge')
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
# rogue_metric = load_metric('rogue')


# In[ ]:


rouge_metric


# In[ ]:


score = calculate_metric_on_test_ds(
    dataset_samsum['test'][0:10], rouge_metric, trainer.model, tokenizer, batch_size = 2, column_text = 'dialogue', column_summary = 'summary'
) ## Calculating for top 10 data itself

## Directly use the scores without accessing fmeasure or mid
rouge_dict = {rn: score[rn] for rn in rouge_names}

# Convert the dictionary to a DataFrame for easy visualization
import pandas as pd
pd.DataFrame(rouge_dict, index=[f'pegasus'])


# ## Interpreting Good Vs Bad ROUGE Scores:
# 
# 1. Scores close to 1: This indicates a strong overlap between the generated summary and the reference summary, which is desirable in summarization tasks. For example, an F1-score of 0.7 or higher across metrics is generally considered good.
# 2. Scores between 0.5 and 0.7: Indicates moderate overlap. The summary might be capturing key points but is likely missing some structure or important information.
# 3. Scores below 0.5: Suggest a poor match between the generated and reference summaries. The model might be generating irrelevant or incomplete summaries that don't capture the ideas well.

# In[ ]:


## Save model
model_pegasus.save_pretrained("pegasus-samsum-model")


# In[ ]:


## Save tokenizer
tokenizer.save_pretrained("tokenizer")


# In[ ]:


# Load
tokenizer = AutoTokenizer.from_pretrained("tokenizer")


# In[ ]:


gen_kwargs = {"length_penalty":0.8, "num_beams":8, "max_length":128}

sample_text = dataset_samsum["test"][0]["dialogue"]
reference = dataset_samsum["test"][0]["summary"]

pipe = pipeline("summarization", model="pegasus-samsum-model", tokenizer=tokenizer)

##
print("Dialogue:")
print(sample_text)

print("\nReference Summary:")
print(reference)  ## This is my true summary

print("\nModel Summary:")
print(pipe(sample_text, **gen_kwargs)[0]["summary_text"]) ## This is my generated summary from the model. 

