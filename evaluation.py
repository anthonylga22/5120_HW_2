# 1. import any required library to load dataset, open files (os), print confusion matrix and accuracy score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle


# 2. Create test set if you like to do the 80:20 split programmatically or if you have not already split the data at this point
X_train, X_test, y_train, y_test = train_test_split(test_size=0.2) 

# 3. Load your saved model for pulsar classifier that you saved in pulsar_classification.py via Pikcle
pickled_model  = pickle.load(open('pulsasr_classification', "rb"))

# 4. Make predictions on test_set created from step 2


# 5. use predictions and test_set (X_test) classifications to print the following:
#    1. confution matrix, 2. accuracy score, 3. precision, 4. recall, 5. specificity
#    You can easily find the formulae for Precision, Recall, and Specificity online.

# Get and print confusion matrix
cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

# Create train and test splits for model development using 80/20 split ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Train SVM
classifier = SVC(kernel='rbf', C=10.0, gamma=0.3)
classifier.fit(X_train, y_train)

# Test developed SVC on unseen pulsar dataset samples
y_pred = classifier.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("The Accuracy Score: ", accuracy)

# Save SVC model
filename = 'pulsarClassifier.sav'
pickle.dump(classifier, open(filename, "wb"))




# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

# Metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# Compute Precision and use the following line to print it
precision = TP / (TP+FP)
print('Precision : {0:0.3f}'.format(precision))

# Compute Recall and use the following line to print it
recall = TP / (TP+FN)
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = TN / (TN+FP)
print('Specificity : {0:0.3f}'.format(specificity))