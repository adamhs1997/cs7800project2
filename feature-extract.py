# Adam Horvath-Smith
# CS 7800 Project 2
# Keke Chen

"""
This file extracts the features needed to classify/cluster the mini newsgroups
dataset. The following steps are used:
    1. Build inverted index over entire dataset
    2. Assigns a feture id number to each indexed term
    3. Maps each newsgroups set to one of 6 classes
    4. Generates a libsvm file with TF-IDF of each term
"""

from sys import argv
from index import InvertedIndex, index_newsgroups
from os import listdir

# Get input args
newsgroups_root_dir = argv[1]
feat_def_path = argv[2]
class_def_path = argv[3]
training_data_path = argv[4]

# Generate index
#index_newsgroups(newsgroups_root_dir, "idx_save.pkl")
ii = InvertedIndex()
ii.load("idx_save.pkl")

# Write out feature/term pairs to feat_def_path
feature_id = 0
with open(feat_def_path, 'w') as outf:
    for item in ii.items:
        outf.write(str(feature_id) + "," + str(item) + "\n")
        feature_id += 1
        
# Map the different newsgroups to a given class
# This is fairly manual...
with open(class_def_path, 'w') as outf:
    for dir in listdir(newsgroups_root_dir):
        if dir.startswith("comp"):
            outf.write("computing," + dir + "\n")
        elif dir.startswith("rec"):
            outf.write("recreation," + dir + "\n")
        elif dir.startswith("sci"):
            outf.write("science," + dir + "\n")
        elif dir.startswith("misc"):
            outf.write("miscellaneous," + dir + "\n")
        elif dir.startswith("talk.politics"):
            outf.write("politics," + dir + "\n")
        else:
            outf.write("religion," + dir + "\n")



