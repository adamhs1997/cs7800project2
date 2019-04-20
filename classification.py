# Adam Horvath-Smith
# CS 7800 Project 2
# Keke Chen

from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from warnings import filterwarnings

# The cross-validation code seems to split classes in such a way that some
#   of the five newsgroups classes are not trained on, but they are tested.
#   This yields some warnings about ill-defined classes, though the system
#   still reports scores. We suppress these here.
filterwarnings("ignore")

# Import the libSVM data file
# --Testing only done on the TF-IDF data
feature_vectors, targets = load_svmlight_file(r"training_data.txt.TFIDF")

# Evaluate multinomial NBC
print("--Multinomial NBC--")
clf = MultinomialNB()
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='precision_macro')
print("Precision Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='recall_macro')
print("Recall Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()

# Evaluate Bernoulli NBC
print("--Bernoulli NBC--")
clf = BernoulliNB()
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='precision_macro')
print("Precision Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='recall_macro')
print("Recall Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()

# Evaluate KNN
print("--KNN--")
clf = KNeighborsClassifier()
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='precision_macro')
print("Precision Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='recall_macro')
print("Recall Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()

# Evaluate SVM
print("--SVM (default kernel)--")
clf = SVC()
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='precision_macro')
print("Precision Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='recall_macro')
print("Recall Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()

print("--SVM (linear kernel)--")
clf = SVC(kernel="linear")
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='f1_macro')
print("F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='precision_macro')
print("Precision Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, feature_vectors, targets, cv=5, scoring='recall_macro')
print("Recall Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()
