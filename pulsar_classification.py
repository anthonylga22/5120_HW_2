# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# In this section, you can use a search engine to look for the functions that will help you implement the following steps

# Load dataset and show basic statistics
# 1. Show dataset size (dimensions)
# 2. Show what column names exist for the 9 attributes in the dataset
# 3. Show the distribution of target_class column
# 4. Show the percentage distribution of target_class column

# Load Dataset
dataset = pd.read_csv('pulsar_stars.csv')



# Separate predictor variables from the target variable (X and y as we did in the class)
X = dataset.iloc[:,:-1] # Gets all the rows (except for the last one because it's the output)
y = dataset.iloc[:,-1] # Gets all the columns





# Create train and test splits for model development. Use the 80% and 20% split ratio
# Name them as X_train, X_test, y_train, and y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=0) 

#X_train = [] # Remove this line after implementing train test split
#X_test = [] # Remove this line after implementing train test split





# Standardize the features (Import StandardScaler here)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Below is the code to convert X_train and X_test into data frames for the next steps
cols = X_train.columns
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])






# Train SVM with the following parameters.
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3
classifier = SVC(kernel='rbf', C=10.0, gamma=0.3)
classifier.fit(X_train, y_train)




# Test the above developed SVC on unseen pulsar dataset samples
y_pred = classifier.predict(X_test)


# compute and print accuracy score
print(classifier.predict(sc.transform([[30, 87000]])))





# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment









# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]



# Compute Precision and use the following line to print it
precision = 0 # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
recall = 0 # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = 0 # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))

