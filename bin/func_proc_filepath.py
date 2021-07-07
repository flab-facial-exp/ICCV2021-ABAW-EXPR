#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import os


# In[21]:


# eg. valid_name = ["_Cam1", "_Cam2"]
# eg. invalid_name = ["_Cam3", "_Cam4"]
# eg. ext = ".txt"
#
def search_files(dir_root, valid_names, invalid_names, ext=None, recursive=False):
    if dir_root.endswith("\\") == False:
        dir_root = dir_root + "\\"
    if recursive == True:
        if ext == None:
            file_data = dir_root + "**\\*"
        else:
            if ext.startswith(".") == False:
                ext = "." + ext
            file_data = dir_root + "**\\*" + ext
    else:
        if ext == None:
            file_data = dir_root + "*"
        else:
            if ext.startswith(".") == False:
                ext = "." + ext
            file_data = dir_root + "*" + ext
            
    files_list = [
        filename for filename in sorted(glob.glob(file_data, recursive=recursive))
        if any([(x in filename) for x in valid_names])
        if all([(y not in filename) for y in invalid_names])
    ]
    return files_list


# In[48]:


def get_name_list(files_list):
    names = np.zeros(1)
    for fp in files_list:
        name = os.path.splitext(os.path.basename(fp))[0]
        names = np.append(names, name)
    names = np.delete(names, 0, 0)
    return names


# In[62]:


class FileInfo:
   def __init__(self):
       self.name = ""
       self.ext = ""
       self.dir_path = ""
       self.dir_name = ""


# In[63]:


def get_file_info(filepath):
    fileinfo = FileInfo()
    
    fileinfo.name = os.path.splitext(os.path.basename(filepath))[0]
    fileinfo.ext = os.path.splitext(os.path.basename(filepath))[1]
    fileinfo.dir_path = os.path.dirname(filepath)
    fileinfo.dir_name = os.path.basename(os.path.dirname(filepath))
    
    return fileinfo


# In[ ]:




