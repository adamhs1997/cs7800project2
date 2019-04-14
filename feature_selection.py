# Adam Horvath-Smith
# CS 7800 Project 2
# Keke Chen

"""
The idea here is to iterate through several k-values for both
the chi-squared and MI feature selection algorithms. For each
k, track and chart the selected k features against the performance
of each of the 4 classifiers.
"""

from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from warnings import filterwarnings
import numpy as np
import matplotlib.pyplot as plt

# Suppress the SVM future warning
filterwarnings("ignore")


def main():

    # Import the libSVM data file
    # --Testing only done on the TF-IDF data
    x, y = load_svmlight_file(r"training_data.txt.TFIDF")

    # Select a few k-values to test
    k_values = [100, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

    # Hold our score results here
    mnbc_x2_scores = []
    mnbc_mi_scores = []
    bnbc_x2_scores = []
    bnbc_mi_scores = []
    knn_x2_scores = []
    knn_mi_scores = []
    svm_x2_scores = []
    svm_mi_scores = []

    # Test our classifiers
    for k in k_values:
        # Extract k features with chi-squared and MI
        x_new1 = SelectKBest(chi2, k=k).fit_transform(x, y)
        x_new2 = SelectKBest(mutual_info_classif, k=k).fit_transform(x, y)
        
        # Report F1 value for each k-value from each classifier
        print("--Multinomial NBC, K-Value {}, Chi-Squared--".format(k))
        clf = MultinomialNB()
        scores = cross_val_score(clf, x_new1, y, cv=5, scoring='f1_macro')
        print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        mnbc_x2_scores.append((k, scores.mean(), scores.std()))
        print()
        
        print("--Multinomial NBC, K-Value {}, MI--".format(k))
        scores = cross_val_score(clf, x_new2, y, cv=5, scoring='f1_macro')
        print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        mnbc_mi_scores.append((k, scores.mean(), scores.std()))
        print()
        
        print("--Bernoulli NBC, K-Value {}, Chi-Squared--".format(k))
        clf = BernoulliNB()
        scores = cross_val_score(clf, x_new1, y, cv=5, scoring='f1_macro')
        print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        bnbc_x2_scores.append((k, scores.mean(), scores.std()))
        print()
        
        print("--Bernoulli NBC, K-Value {}, MI--".format(k))
        scores = cross_val_score(clf, x_new2, y, cv=5, scoring='f1_macro')
        print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        bnbc_mi_scores.append((k, scores.mean(), scores.std()))
        print()
        
        print("--KNN, K-Value {}, Chi-Squared--".format(k))
        clf = KNeighborsClassifier()
        scores = cross_val_score(clf, x_new1, y, cv=5, scoring='f1_macro')
        print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        knn_x2_scores.append((k, scores.mean(), scores.std()))
        print()
        
        print("--KNN, K-Value {}, MI--".format(k))
        scores = cross_val_score(clf, x_new2, y, cv=5, scoring='f1_macro')
        print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        knn_mi_scores.append((k, scores.mean(), scores.std()))
        print()
        
        print("--SVM, K-Value {}, Chi-Squared--".format(k))
        clf = SVC()
        scores = cross_val_score(clf, x_new1, y, cv=5, scoring='f1_macro')
        print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        svm_x2_scores.append((k, scores.mean(), scores.std()))
        print()
        
        print("--SVM, K-Value {}, MI--".format(k))
        scores = cross_val_score(clf, x_new2, y, cv=5, scoring='f1_macro')
        print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        svm_mi_scores.append((k, scores.mean(), scores.std()))
        print()

    plot_scores(mnbc_x2_scores, mnbc_mi_scores, "Multinomial NBC", "mnbc_scores.png")
    plot_scores(bnbc_x2_scores, bnbc_mi_scores, "Bernoulli NBC", "bnbc_scores.png")
    plot_scores(knn_x2_scores, knn_mi_scores, "KNN", "knn_scores.png")
    plot_scores(svm_x2_scores, svm_mi_scores, "SVM", "svm_scores.png")
  
    
def plot_scores(x2_scores_raw, mi_scores_raw, title, outfile):
    # Set up the plot
    plt.figure()
    plt.title(title)
    plt.xlabel("K Value")
    plt.ylabel("F1 Score")
    x2_scores = np.array(x2_scores_raw)
    mi_scores = np.array(mi_scores_raw)
    plt.grid()

    # Chart the standard deviations and points
    plt.fill_between(x2_scores[:,0], x2_scores[:,1] - x2_scores[:,2],
                     x2_scores[:,1] + x2_scores[:,2], alpha=0.1, color="b")
    plt.fill_between(mi_scores[:,0], mi_scores[:,1] - mi_scores[:,2],
                     mi_scores[:,1] + mi_scores[:,2], alpha=0.1, color="g")
    plt.plot(x2_scores[:,0], x2_scores[:,1], 'o-', color="b",
             label="Chi-Squared Scores")
    plt.plot(mi_scores[:,0], mi_scores[:,1], 'o-', color="g",
             label="MI Scores")

    # Save off the figure    
    plt.legend(loc="best")
    plt.savefig(outfile)
    print("Saved figure", outfile)

    
if __name__ == "__main__":
    main()
    