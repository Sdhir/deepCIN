# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:53:28 2019

@author: ssbw5

Generate 704x64 images and save in each specified class folder
"""

import numpy as np
#from utilities import get_files_path
import os
import cv2


# set the segments folder path for the 3 datasets we have
folderspath = 'C:\\Users\\ssbw5\\Documents\\dataSets_hist_matched\\vertical_patches\\org_vs\\sz32Segs_'
resize = True
#%%
# read the segments information and calculate the aspect ratio for the segments

# Get segments information as a list 

working_dir = folderspath

if resize:
    height = 704
    width = 64

out_dir = os.path.join("\\".join(working_dir.split('\\')[:-1]),'sz32Segs_corrected')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


for dirpath, dirnames, filenames in os.walk(working_dir):
    for filename in [f for f in filenames if f.endswith(".tif")]:
        print(filename)
        im = cv2.imread(os.path.join(dirpath, filename))
        rows = im.shape[0]
        cols = im.shape[1]
        if rows%2 == 1:
            im = im[:-1,:,:]
            
        if cols%2 == 1:
            im = im[:,:-1,:]
            
        rows = im.shape[0]
        cols = im.shape[1]
            
        if resize:
            im = cv2.resize(im,(width,height), interpolation = cv2.INTER_CUBIC)
        
        im_folder = dirpath[len(working_dir)+1:] #dirpath.split("\\")[-1]
        
        out_dir_file = os.path.join(out_dir,im_folder)
        if not os.path.exists(out_dir_file):
            os.mkdir(out_dir_file)
        cv2.imwrite(os.path.join(out_dir_file,filename),im)
#        io.imshow(im)
        