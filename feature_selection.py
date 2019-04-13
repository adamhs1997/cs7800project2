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

# Import the libSVM data file
# --Testing only done on the TF-IDF data
x, y = load_svmlight_file(r"training_data.txt.TFIDF")

# Select a few k-values to test
k_values = [100, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

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
    print()
    
    print("--Multinomial NBC, K-Value {}, MI--".format(k))
    scores = cross_val_score(clf, x_new2, y, cv=5, scoring='f1_macro')
    print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
    
    print("--Bernoulli NBC, K-Value {}, Chi-Squared--".format(k))
    clf = BernoulliNB()
    scores = cross_val_score(clf, x_new1, y, cv=5, scoring='f1_macro')
    print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
    
    print("--Bernoulli NBC, K-Value {}, MI--".format(k))
    scores = cross_val_score(clf, x_new2, y, cv=5, scoring='f1_macro')
    print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
    
    print("--KNN, K-Value {}, Chi-Squared--".format(k))
    clf = KNeighborsClassifier()
    scores = cross_val_score(clf, x_new1, y, cv=5, scoring='f1_macro')
    print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
    
    print("--KNN, K-Value {}, MI--".format(k))
    scores = cross_val_score(clf, x_new2, y, cv=5, scoring='f1_macro')
    print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
    
    print("--SVM, K-Value {}, Chi-Squared--".format(k))
    clf = SVC()
    scores = cross_val_score(clf, x_new1, y, cv=5, scoring='f1_macro')
    print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()
    
    print("--SVM, K-Value {}, MI--".format(k))
    scores = cross_val_score(clf, x_new2, y, cv=5, scoring='f1_macro')
    print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()

