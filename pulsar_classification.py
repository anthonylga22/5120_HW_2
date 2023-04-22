# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

# Load dataset
dataset = pd.read_csv('pulsar_stars.csv')

print("Data Size: ", dataset.shape) # Dataset size (dimensions)
print("Column Names: ", list(dataset.columns)) # Column names that exist for the attributes in the dataset
print("Distribution of the target class: ", dataset['target_class'].value_counts()) # 3. Distribution of target_class column
print("Percentage Distribution of class: ", dataset['target_class'].value_counts(normalize=True) * 100) # 4. Percentage distribution of target_class column

# Separate predictor variables from the target variable
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

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