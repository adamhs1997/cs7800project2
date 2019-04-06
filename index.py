

'''

Index structure:

    The Index class contains a list of IndexItems, stored in a dictionary type for easier access

    each IndexItem contains the term and a set of PostingItems

    each PostingItem contains a document ID and a list of positions that the term occurs

'''

import util
from pickle import dump, load
from math import log10, sqrt
from os import walk


class Posting:
    def __init__(self, docID):
        self.docID = docID
        self.positions = []

    def append(self, pos):
        self.positions.append(pos)

    def sort(self):
        ''' sort positions'''
        self.positions.sort()

    def merge(self, positions):
        self.positions.extend(positions)

    def term_freq(self):
        ''' return the term frequency in the document'''
        
        # The number of times a term is in a document corresponds to 
        #   the length of the position list
        return len(self.positions)
       
    # For testing purposes...
    def __repr__(self):
        return str(self.positions)


class IndexItem:
    def __init__(self, term):
        self.term = term
        self.posting = {} #postings are stored in a python dict for easier index building
        self.sorted_postings= [] # may sort them by docID for easier query processing

    def add(self, docid, pos):
        ''' add a posting'''
        if not docid in self.posting:
            self.posting[docid] = Posting(docid)
        self.posting[docid].append(pos)

    def sort(self):
        ''' sort by document ID for more efficient merging. For each document also sort the positions'''
        # ToDo
        
        # We already have the postings in posting dict. Store the sorted docID keys
        #   in sorted_postings list, for easier reference in postings dict.
        self.sorted_postings = sorted(self.posting)
        
        # We sort the positions of each posting in place
        for doc in self.posting:
            self.posting[doc].sort()


class InvertedIndex:

    def __init__(self):
        self.items = {} # list of IndexItems
        #self.doc_tfidf = {} # tf-idf of every term in every doc
        self.nDocs = 0  # the number of indexed documents


    def indexDoc(self, doc): # indexing a Document object
        ''' indexing a docuemnt, using the simple SPIMI algorithm, but no need to store blocks due to the small collection we are handling. Using save/load the whole index instead'''
        
        # Using the SPIMI algorithm as defined at
        # https://nlp.stanford.edu/IR-book/html/htmledition/single-pass-in-memory-indexing-1.html
        
        # Each term in a doc has its own index item!!!
        
        # Preprocess first...
        # Call tokenize_doc to convert doc title and body into tokenized list, in lowercase
        # Do remove stopwords and stemming as expected

        # ToDo: indexing only title and body; use some functions defined in util.py
        # (1) convert to lower cases,
        # (2) remove stopwords,
        # (3) stemming
        
        # Then go term-by-term and create the index. Use algorithm to track which terms already in index, add new ones if not. If we create a new index item, add it to the self.items dict!!!
        
        # ---
        
        # Increment number of documents indexed
        self.nDocs += 1
        
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
        stemmed_token_list = list(map(lambda tok: util.stemming(tok), token_list_no_stopword))
        
        # Note that the stemmed tokens are now our terms
        for pos, term in enumerate(stemmed_token_list):
            # Skip over stopwords, now replaced by ""
            if term == "": continue
            
            # If this term has already appeared, update the existing posting
            if not term in self.items:
                self.items[term] = IndexItem(term)
            # Use self.nDocs as docID since there is no ID with newsgroups
            self.items[term].add(self.nDocs, pos)


    def sort(self):
        ''' sort all posting lists by docID'''
        #ToDo
        
        # The actual sort is implemented in IndexItem. Just call it here.
        for item in self.items:
            self.items[item].sort()

    def find(self, term):
        return self.items[term] if term in self.items else None

    def save(self, filename):
        ''' save to disk'''
        # ToDo: using your preferred method to serialize/deserialize the index
        
        # Combine items dict and nDocs into a list so they can be pickled together
        to_pickle = [self.items, self.nDocs]#, self.doc_tfidf]
        
        # Use Pickle to dump the index to a file
        with open(filename, 'wb') as out:
            dump(to_pickle, out)

    def load(self, filename):
        ''' load from disk'''
        # ToDo
        
        # Load data back from pickled file
        with open(filename, 'rb') as inf:
            file_read = load(inf)
            self.items = file_read[0]
            self.nDocs = file_read[1]
            #self.doc_tfidf = file_read[2]

    def idf(self, term):
        ''' compute the inverted document frequency for a given term'''
        #ToDo: return the IDF of the term
        
        # IDF of term t is log(total # of docs / # docs with t in it)
        return log10(self.nDocs / len(self.items[term].posting)) \
            if term in self.items else 0

    def compute_tfidf(self):
        """ pre-compute tf-idf vectors for each word in each doc """
        
        # Compute tf-idf vector for every other doc
        for iter in range(self.nDocs):
            doc = iter + 1
            word_vector = {}
            
            for word in self.items:
                # Get tf
                try:
                    tf = self.find(word).posting[doc].term_freq()
                except KeyError: # if not in doc
                    tf = 0
                
                # Get idf
                idf = self.idf(word)
                
                # Calculate tf-idf; add to current dict
                word_vector[word] = log10(1 + tf) * idf
                
            # Normalize the word vector
            accum = 0
            for word in word_vector:
                accum += word_vector[word]**2
            accum = sqrt(accum)
                
            for word in word_vector:
                word_vector[word] /= accum
                
            self.doc_tfidf[doc] = word_vector
    
    
def index_newsgroups(root_newsgroup_dir, save_location):
    print("Building index...")

    # Index each doc in the dataset
    ii = InvertedIndex()
    for root, _, files in walk(root_newsgroup_dir):
        for file in files:
            ii.indexDoc(root + "\\" + file)
            
    # Sort and compute TF-IDF
    ii.sort()
    #ii.compute_tfidf()
    
    # Save off index
    ii.save(save_location)
    print("Index saved to", save_location + "!")
    

if __name__ == '__main__':
    # ii = InvertedIndex()
    # ii.indexDoc(r"D:\CS 7800 Project 2\mini_newsgroups\comp.sys.mac.hardware\50457")
    index_newsgroups(r"D:\CS 7800 Project 2\mini_newsgroups", "idx_save.pkl")
