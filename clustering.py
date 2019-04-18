# Adam Horvath-Smith
# CS 7800 Project 2
# Keke Chen

"""
For each n_clusters in range(2,25), try to do the fit.
If we have time, do feat sel first.
Go through graphing once we are able to generate the data.
"""

from sklearn.datasets import load_svmlight_file
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from warnings import filterwarnings

# Suppress future warnings
filterwarnings("ignore")

# Import the libSVM data file
# --Testing only done on the TF-IDF data
x, y = load_svmlight_file(r"training_data.txt.TFIDF")

# For AC, we need x to be dense array
ac_x = x.toarray()

# Run thorugh n values from 2 to 25:
for n in range(2, 6):
    print("N:", n)
    # Initialize our clusterers
    kmeans_model = KMeans(n_clusters=n).fit(x)
    single_linkage_model = AgglomerativeClustering(
            n_clusters=n, linkage='ward').fit(ac_x)
            
    # Get the scores
    clustering_labels = kmeans_model.labels_
    print("KM SI", metrics.silhouette_score(
        x, clustering_labels, metric='euclidean'))
    print("KM NMI", metrics.normalized_mutual_info_score(y, clustering_labels))

    clustering_labels = single_linkage_model.labels_
    print("AC SI", metrics.silhouette_score(
        ac_x, clustering_labels, metric='euclidean'))
    print("KM NMI", metrics.normalized_mutual_info_score(y, clustering_labels))
    print()
        

