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
import numpy as np
import matplotlib.pyplot as plt

# Suppress future warnings
filterwarnings("ignore")

def main():
    # Import the libSVM data file
    # --Testing only done on the TF-IDF data
    x, y = load_svmlight_file(r"training_data.txt.TFIDF")

    # For AC, we need x to be dense array
    ac_x = x.toarray()

    # Hold values in lists to plot later
    km_sc_scores = []
    km_nmi_scores = []
    ac_sc_scores = []
    ac_nmi_scores = []

    # Run thorugh n values from 2 to 25:
    for n in range(2, 26):
        print("N:", n)
        # Initialize our clusterers
        kmeans_model = KMeans(n_clusters=n).fit(x)
        single_linkage_model = AgglomerativeClustering(
                n_clusters=n, linkage='ward').fit(ac_x)
                
        # Get the scores
        clustering_labels = kmeans_model.labels_
        score = metrics.silhouette_score(
            x, clustering_labels, metric='euclidean')
        km_sc_scores.append((n, score))
        print("KM SC", score)
        score = metrics.normalized_mutual_info_score(y, clustering_labels)
        km_nmi_scores.append((n, score))
        print("KM NMI", score)

        clustering_labels = single_linkage_model.labels_
        score = metrics.silhouette_score(
            ac_x, clustering_labels, metric='euclidean')
        ac_sc_scores.append((n, score))
        print("AC SC", score)
        score = metrics.normalized_mutual_info_score(y, clustering_labels)
        ac_nmi_scores.append((n, score))
        print("AC NMI", score)
        print()
        
    # Plot accumulated scores
    plot_scores(km_sc_scores, ac_sc_scores, "Silhouette Coefficient Scores", 
        "sc_scores.png")
    plot_scores(km_nmi_scores, ac_nmi_scores, "NMI Scores",
        "nmi_scores.png")
        
        
def plot_scores(km_scores_raw, ac_scores_raw, title, outfile):
    # Set up the plot
    plt.figure()
    plt.title(title)
    plt.xlabel("N (Number of Clusters)")
    plt.ylabel("Score")
    km_scores = np.array(km_scores_raw)
    ac_scores = np.array(ac_scores_raw)
    plt.grid()

    # Chart the points
    plt.plot(km_scores[:,0], km_scores[:,1], 'o-', color="b",
             label="KMeans")
    plt.plot(ac_scores[:,0], ac_scores[:,1], 'o-', color="g",
             label="Agglomerative Clustering")

    # Save off the figure    
    plt.legend(loc="best")
    plt.savefig(outfile)
    print("Saved figure", outfile)
    
    
if __name__ == "__main__":
    main()
