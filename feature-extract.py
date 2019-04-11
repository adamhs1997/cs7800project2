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
from os import listdir, walk, sep
from math import log10
import util

def main():

    #########
    # SETUP #
    #########
    
    # Get input args
    newsgroups_root_dir = argv[1]
    feat_def_path = argv[2]
    class_def_path = argv[3]
    training_data_path = argv[4]
    feature_value_type = int(argv[5])

    # Generate index
    #index_newsgroups(newsgroups_root_dir, "idx_save.pkl")
    ii = InvertedIndex()
    ii.load("idx_save.pkl")

    # Write out feature/term pairs to feat_def_path
    feature_id = 0
    with open(feat_def_path, 'w') as outf:
        for item in ii.items:
            outf.write(str(feature_id) + " " + str(item) + "\n")
            feature_id += 1
    
    # Read back in the feature/term pairs for later
    with open(feat_def_path, 'r') as inf:
        ft_pairs = inf.readlines()
        
    # Put the ft_pairs into a dictionary for quick lookup
    ft_dict = {}
    for pair in ft_pairs:
        ft_dict[pair.split()[1].strip()] = pair.split()[0]
            
    # Map the different newsgroups to a given class
    # This is fairly manual...
    with open(class_def_path, 'w') as outf:
        for dir in listdir(newsgroups_root_dir):
            outf.write(class_def_helper(dir) + " " + dir + "\n")

    ############################
    # TRAINING DATA GENERATION #
    ############################
            
    # Create the training data
    # For each document:
        # Find its containing folder, and extract class from class def
        # For each term in document
            # Compute tfidf, tf or idf
    current_file_id = 1
    with open(training_data_path, 'w') as outf:
        if feature_value_type == 0:
            # Compute tf-idf
            # Go through each document in newsgroups dir
            for root, _, files in walk(newsgroups_root_dir):
                # Find and write out the class label
                local_dir = root.split(sep)[-1]
                
                # For each file...
                for file in files:
                    outf.write(class_def_helper(local_dir) + " ")
                    print(root, file)
                    
                    # Get the words from the doc
                    stemmed_token_list = preprocess_doc(root + sep + file)
                    
                    # Put all the info into a set (for uniqueness)
                    data_set = set()
                    
                    # Now that we've re-done all that, find idfs
                    for word in stemmed_token_list:
                        # Skip blank stopwords
                        if word == "": continue
                        
                        # Get the term ID
                        #outf.write(ft_dict[word] + ":")

                        # Calculate and write out TF-IDF
                        # Note current_file_id is our doc_id
                        tf = ii.find(word).posting[current_file_id].term_freq()
                        idf = ii.idf(word)
                        #outf.write(str(log10(1 + tf) * idf) + " ")
                        data_set.add(ft_dict[word] + ":" + str(log10(1 + tf) * idf))
                        
                    # Write newline to signify end of file
                    #outf.write("\n")
                    outf.write(" ".join(sorted(data_set, key=lambda x: int(x.split(':')[0]))) + "\n")
                    outf.flush()
                    
                    # Increment our current doc
                    current_file_id += 1
                    
        elif feature_value_type == 1:
            # Compute tf
            # Go through each document in newsgroups dir
            for root, _, files in walk(newsgroups_root_dir):
                # Find and write out the class label
                local_dir = root.split(sep)[-1]
                
                # For each file...
                for file in files:
                    outf.write(class_def_helper(local_dir) + " ")
                    print(root, file)
                    
                    # Get the words from the doc
                    stemmed_token_list = preprocess_doc(root + sep + file)
                    
                    # Put all the info into a set (for uniqueness)
                    data_set = set()
                    
                    # Now that we've re-done all that, find idfs
                    for word in stemmed_token_list:
                        # Skip blank stopwords
                        if word == "": continue
                        
                        # Get the term ID
                        #outf.write(ft_dict[word] + ":")

                        # Write the TF
                        # Note current_file_id is our doc_id
                        # outf.write(str(ii.find(word).posting[
                            # current_file_id].term_freq()) + " ")
                        data_set.add(ft_dict[word] + ":" + str(ii.find(word).posting[
                            current_file_id].term_freq()))
                        
                    # Write newline to signify end of file
                    # outf.write("\n")
                    outf.write(" ".join(sorted(data_set, key=lambda x: int(x.split(':')[0]))) + "\n")
                    # outf.flush()
                    
                    # Increment our current doc
                    current_file_id += 1
                    
        elif feature_value_type == 2:
            # Compute idf
            # Go through each document in newsgroups dir
            for root, _, files in walk(newsgroups_root_dir):
                # Find and write out the class label
                local_dir = root.split(sep)[-1]
                
                # For each file...
                for file in files:
                    outf.write(class_def_helper(local_dir) + " ")
                    print(root, file)
                    
                    # Get the words from the doc
                    stemmed_token_list = preprocess_doc(root + sep + file)
                    
                    # Put all the info into a set (for uniqueness)
                    data_set = set()
                    
                    # Now that we've re-done all that, find idfs
                    for word in stemmed_token_list:
                        # Skip blank stopwords
                        if word == "": continue
                        
                        # Get the term ID
                        #outf.write(ft_dict[word] + ":" + str(ii.idf(word))
                        #    + " ") 
                        data_set.add(ft_dict[word] + ":" + str(ii.idf(word)))
                        
                    # Write newline to signify end of file
                    outf.write(" ".join(sorted(data_set, key=lambda x: int(x.split(':')[0]))) + "\n")
                    #outf.flush()
                    
        else:
            print("Invalid feature type! Abort.")
            exit(1)
            
            
def class_def_helper(dir):
    """This function returns the class given a dir name"""
    if dir.startswith("comp"):
        return "0"
    elif dir.startswith("rec"):
        return "1"
    elif dir.startswith("sci"):
        return "2"
    elif dir.startswith("misc"):
        return "3"
    elif dir.startswith("talk.politics"):
        return "4"
    else:
        return "5"

        
def preprocess_doc(doc):
    """Get the words back out of the file"""
    # Read in doc, only get subject and body of the document
    with open(doc) as f:
        doc_lines = f.readlines()
        
    sub_body_lines = []
    for l in doc_lines:
        if l.startswith("Subject:"):
            sub_body_lines.append(l[9:].strip())
        if l.startswith("Lines:"):
            num_lines = int(l[7:])
    
    for l in doc_lines[len(doc_lines) - num_lines:]:
        if l is not "\n":
            sub_body_lines.append(l.strip())
            
    # Process all the words again
    # Get doc string
    doc_string = " ".join(sub_body_lines)
    
    # Tokenize and lowercase doc into list form
    token_list = util.tokenize_doc(doc_string)
        
    # Helper function to replace stopwords with empty string
    def remove_stop_word(tok):
        return "" if util.isStopWord(tok) else tok
        
    # Remove the stopwords from both positional list and token list
    token_list_no_stopword = list(map(remove_stop_word, token_list))
    
    # Stem the words
    stemmed_token_list = list(map(
        lambda tok: util.stemming(tok), token_list_no_stopword))
        
    return stemmed_token_list
        
        
if __name__ == "__main__":
    main()
