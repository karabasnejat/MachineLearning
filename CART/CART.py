# -*- coding: utf-8 -*-
"""CART.ipynb

***Oğuzhan Nejat Karabaş***


# Setting up the Required Environment
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from graphviz import Digraph

"""# Loading and Preprocessing the Dataset"""

# Loading and Preprocessing the Dataset
train_data = pd.read_csv('trainSet.csv')
test_data = pd.read_csv('testSet.csv')

# Let's see our datasets
print(train_data)

print(test_data)

"""# Numerical Conversion of Categorical Variables
- Categorical variables are non-numeric values present in our datasets.
To make our model more accurate, we will use LabelEncoder() to convert these variables to numeric values.
"""

cat_cols = ['A1', 'A2', 'A3', 'A4',
            'A5', 'A6', 'A7', 'A8',
            'A9', 'A10', 'A11', 'A12', 
            'A13', 'A14', 'A15', 'class']
le = LabelEncoder()
for col in train_data.columns:
    if train_data[col].dtype == 'object': # Identify categorical columns
        le.fit(train_data[col].values) 
        train_data[col] = le.transform(train_data[col]) # Numerically convert columns in the training set
        test_data[col] = le.transform(test_data[col]) # Numerically convert columns in the test set

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

"""# Building the Decision Tree Model"""

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            y_pred[i] = self._traverse_tree(x, self.tree)
        return y_pred
    
    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth == self.max_depth or 
            num_labels == 1 or 
            num_samples < self.min_samples_split):
            return np.argmax(np.bincount(y))
        
        # Finding the best split
        best_feature, best_threshold = self._find_best_split(X, y, num_samples, num_features)
        
        # Splitting the data
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth+1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth+1)
        
        # Creating a node
        return {'feature': best_feature, 'threshold': best_threshold, 
                'left': left_tree, 'right': right_tree}
    
    def _find_best_split(self, X, y, num_samples, num_features):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        # Calculating Gini impurity for each feature and threshold
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices
                if (np.sum(left_indices) < self.min_samples_leaf or 
                    np.sum(right_indices) < self.min_samples_leaf):
                    continue
                left_labels = y[left_indices]
                right_labels = y[right_indices]
                gini = (len(left_labels)/num_samples)*self._gini_impurity(left_labels) + \
                       (len(right_labels)/num_samples)*self._gini_impurity(right_labels)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _gini_impurity(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return 1 - np.sum(probabilities**2)
    
    def _traverse_tree(self, x, tree):
        if type(tree) != dict:
            return tree
        feature_value = x[tree['feature']]
        if feature_value < tree['threshold']:
            return self._traverse_tree(x, tree['left'])
        else:
            return self._traverse_tree(x, tree['right'])

# Training our model
dtc = DecisionTreeClassifier()
dtc = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5)
dtc.fit(X_train, y_train)

# Evaluating the model on training data
y_train_pred = dtc.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_train_pred).ravel()
tpr_train = tp_train / (tp_train + fn_train)
tnr_train = tn_train / (tn_train + fp_train)

# Evaluating the model on test data
y_test_pred = dtc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
tpr_test = tp_test / (tp_test + fn_test)
tnr_test = tn_test / (tn_test + fp_test)

"""# Printing and Saving the Results to a .txt File"""

with open("performance_metrics.txt", "w") as f:
    f.write("Training Results:\n")
    f.write(f"Accuracy: {train_accuracy}\n")
    f.write(f"True Positive Rate: {tpr_train}\n")
    f.write(f"True Negative Rate: {tnr_train}\n")
    f.write(f"True Positive Count: {tp_train}\n")
    f.write(f"True Negative Count: {tn_train}\n")
    f.write("\nTest Results:\n")
    f.write(f"Accuracy: {test_accuracy}\n")
    f.write(f"True Positive Rate: {tpr_test}\n")
    f.write(f"True Negative Rate: {tnr_test}\n")
    f.write(f"True Positive Count: {tp_test}\n")
    f.write(f"True Negative Count: {tn_test}\n")
# Printing training and test results to the console
    print("Training Results:")
    print(f"Accuracy: {train_accuracy}")
    print(f"True Positive Rate: {tpr_train}")
    print(f"True Negative Rate: {tnr_train}")
    print(f"True Positive Count: {tp_train}")
    print(f"True Negative Count: {tn_train}\n")

    print("Test Results:")
    print(f"Accuracy: {test_accuracy}")
    print(f"True Positive Rate: {tpr_test}")
    print(f"True Negative Rate: {tnr_test}")
    print(f"True Positive Count: {tp_test}")
    print(f"True Negative Count: {tn_test}\n")

"""# Drawing the Decision Tree Model and Saving it as a .png File"""

def draw_tree(tree, feature_names):
    from graphviz import Digraph

    g = Digraph('G', filename='decision_tree.gv', format='png')
    draw_node(g, '0', tree, feature_names)
    return g

def draw_node(g, name, tree, feature_names):
    if type(tree) != dict:
        g.node(name, label=str(tree), shape='ellipse')
    else:
        g.node(name, label=feature_names[tree['feature']] + '\n' + str(tree['threshold']))
        left_name = name + '0'
        right_name = name + '1'
        draw_node(g, left_name, tree['left'], feature_names)
        draw_node(g, right_name, tree['right'], feature_names)
        g.edge(name, left_name, label='<')
        g.edge(name, right_name, label='>=')

feature_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']
tree_graph = draw_tree(dtc.tree, feature_names)

tree_graph.render()
