#!/usr/bin/env python
# coding: utf-8

# ## Importance of using the ConfigBox

# In[1]:


dict_info = {"name": "Shubham", "lastname":"Bhardwaj"}

dict_info["name"] ## Here u will be able to get the info


# In[ ]:


## But if i try to retrieve this information by writing the way below
dict_info.name ## Here it will throw an error

## Most of the yaml files will be in the form of key-value pairs and its great that instead of writing key in this way: dict_info["name"]
## we should write in this way: dict_info.name and get the output


# In[2]:


from box import ConfigBox
dict_info = ConfigBox({"name": "Shubham", "lastname":"Bhardwaj"})
dict_info.name ## Now if i execute this it will display the name 


# ## Ensure Annotations

# In[3]:


def multiplication(x:int,y:int) -> int:
    return x*y


# In[4]:


multiplication(2,3)


# In[ ]:


multiplication(2,"3") ## This is giving a output this should not happen, here we should get an error as we are passing a string and not a integer as it is in the function defined above


# - To prevent the above scenario we use something called as ensure_annotations

# In[6]:


from ensure import ensure_annotations

@ensure_annotations
def multiplication(x:int,y:int) -> int:
    return x*y


# In[ ]:


multiplication(2,"3")  ## This time if i execute this code it will throw me a error
## This is why we use ensure annotations : So that whatever inputs u are giving it should be of same datatype otherwise it will be an error.

