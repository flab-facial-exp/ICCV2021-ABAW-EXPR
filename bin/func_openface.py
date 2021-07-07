#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import os
import subprocess
import time
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


"""
FeatureExtraction.exe -f "C:\my videos\video.avi"
FeatureExtraction.exe -fdir "C:\my videos\sequence1"
FaceLandmarkImg.exe -f "C:\my images\img.jpg"
 -aus -pose -gaze
"""


# In[ ]:


"""
predict au, head-pose, gaze using openface
dir_ofe: folder including model folder, FeatureExtraction.exe
dir_out: output folder
filepath_video: file path of video
"""
def func_video_to_au(dir_ofe, dir_out, filepath_video):

    start = time.time() #開始時刻
    # openface.exe for video convert
    file_ofe = dir_ofe + "\\FeatureExtraction.exe"
    # video name without folder path and ext
    video_name = os.path.splitext(os.path.basename(filepath_video))[0]
    log = "** FILE INFO **\n  name\t: {0}".format(video_name)
    print(log)
    
    # generate command
    str_ofe = '"' + file_ofe + '" '
    str_src = '-f "' + filepath_video + '" '
    str_out = '-out_dir "' + dir_out + '" '
    #str_opt = "-pose -2Dfp -3Dfp -aus -gaze"
    str_opt = "-pose -aus -gaze"
    cmd = str_ofe + str_src + str_out + str_opt
    log = "** CONVERT: convert video to data **"
    print(log)
    # run openface using command prompt
    returncode = subprocess.call(cmd)
    log = "** CONVERT: finished **"
    print(log)
    # output processing time
    end = time.time() #終了時刻
    proc_time = end-start
    log = "processing time: {} sec".format(proc_time)
    print(log)
    
    out_path = dir_out + "\\" + video_name + ".csv"
    
    return out_path


# In[ ]:


"""
predict au, head-pose, gaze using openface
dir_ofe: folder including model folder, FeatureExtraction.exe
dir_out: output folder
dirpath_images: folder path of images
"""
def func_images_to_au(dir_ofe, dir_out, dirpath_images):

    start = time.time() #開始時刻
    # openface.exe for video convert
    file_ofe = dir_ofe + "\\FeatureExtraction.exe"
    # video name without folder path and ext
    video_name = os.path.basename(dirpath_images)
    log = "** FILE INFO **\n  name\t: {0}".format(video_name)
    print(log)
    
    # generate command
    str_ofe = '"' + file_ofe + '" '
    str_src = '-fdir "' + dirpath_images + '" '
    str_out = '-out_dir "' + dir_out + '" '
    #str_opt = "-pose -2Dfp -3Dfp -aus -gaze"
    str_opt = "-pose -aus -gaze"
    cmd = str_ofe + str_src + str_out + str_opt
    log = "** CONVERT: convert images to data **"
    print(log)
    # run openface using command prompt
    returncode = subprocess.call(cmd)
    log = "** CONVERT: finished **"
    print(log)
    # output processing time
    end = time.time() #終了時刻
    proc_time = end-start
    log = "processing time: {} sec".format(proc_time)
    print(log)
    
    out_path = dir_out + "\\" + video_name + ".csv"
    
    return out_path


# In[ ]:


"""
predict au, head-pose, gaze using openface
dir_ofe: folder including model folder, FeatureExtraction.exe
dir_out: output folder
filepath_image: file path of image
"""
def func_image_to_au(dir_ofe, dir_out, filepath_image):

    start = time.time() #開始時刻
    # openface.exe for video convert
    file_ofe = dir_ofe + "\\FaceLandmarkImg.exe"
    # video name without folder path and ext
    video_name = os.path.splitext(os.path.basename(filepath_image))[0]
    log = "** FILE INFO **\n  name\t: {0}".format(video_name)
    print(log)
    
    # generate command
    str_ofe = '"' + file_ofe + '" '
    str_src = '-f "' + filepath_image + '" '
    str_out = '-out_dir "' + dir_out + '" '
    #str_opt = "-pose -2Dfp -3Dfp -aus -gaze"
    str_opt = "-pose -aus -gaze"
    cmd = str_ofe + str_src + str_out + str_opt
    log = "** CONVERT: convert image to data **"
    print(log)
    # run openface using command prompt
    returncode = subprocess.call(cmd)
    log = "** CONVERT: finished **"
    print(log)
    # output processing time
    end = time.time() #終了時刻
    proc_time = end-start
    log = "processing time: {} sec".format(proc_time)
    print(log)
    
    out_path = dir_out + "\\" + video_name + ".csv"
    
    return out_path

