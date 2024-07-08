# -*- coding: utf-8 -*-
import os
import shutil

out_dir = 'Triplet_type/'
os.makedirs(out_dir)

# set the paths of the source and destination directories
source_dir = '/work/pooryakh/9k_secondary_structure/Dataset/lexiographic'
dest_dir = '/work/pooryakh/9k_secondary_structure/Triplet_type'

# get a list of all the files in the source directory
file_list = os.listdir(source_dir)

# iterate through the list of files and copy only the files containing "1Dkeys_dist26" in their names
for filename in file_list:
    if "3Dkeys_theta30_maxdist35" in filename:
        # construct the full path of the source and destination files
        src_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)

        # use shutil.copy() function to copy the file from the source to destination directory
        shutil.copy(src_file, dest_file)
