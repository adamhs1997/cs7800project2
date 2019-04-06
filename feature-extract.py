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


