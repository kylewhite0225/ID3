# Author: Kyle White
# Class: CS460 - Machine Learning
# Date: 10/29/2021

import pandas as pd
import numpy as np

def getEntropy(df):
    # To make this function generic, get the class label
    className = df.keys()[-1]
    # Initialize entropy to 0
    entropy = 0
    # Gather unique values of class (yes, no, >50k, <=50k, etc)
    targetVariables = df[className].unique()
    # For each unique value
    for attr in targetVariables:
        # Number of 'attr'/total number of rows
        proportion = df[className].value_counts()[attr]/len(df[className])
        # Add -(fraction)log2(fraction)
        entropy += -proportion*np.log2(proportion)
    return entropy
"""
Helper function that returns the calculated entropy of a dataset.
:param df: a pandas dataframe object containing a row of features and rows of values.
:return: the calculated entropy value.
"""

def getInfoGain (df, attr):
    # To make this function generic, get the class label
    className = df.keys()[-1]
    # targetVariables are the yes/no or >50k/<=50k values
    targetVariables = df[className].unique()

    # Subset of the dataframe with 'Yes' values
    subset0 = df.where(df[className] == targetVariables[0]).dropna()
    # Subset of the dataframe with 'No' values
    subset1 = df.where(df[className] == targetVariables[1]).dropna()

    # Number of true values
    true = len(subset0.index)
    # Number of false values
    false = len(subset1.index)
    # Total rows
    total = true + false

    # Keeping track of the entropy summation
    summation = 0
    # For each unique value in the 'attr' column in the dataframe
    for value in df[attr].unique():
        entropy = 0
        # Number of 'yes' values matching the 'value' level of 'attr'
        num1 = len(subset0[attr].where(subset0[attr] == value).dropna())
        # Number of 'no' values matching the 'value' level of 'attr'
        num2 = len(subset1[attr].where(subset1[attr] == value).dropna())
        # Total number of 'value' levels of 'attr'
        denom = df[attr].value_counts()[value]

        # If statement to filter out zero value numerators (not compatible with log2)
        if num1 == 0:
            trueEntropy = 0
        else:
            # Entropy representing the true values in the 'attr' column
            trueEntropy = (num1/denom)*np.log2(num1/denom)
        # If statement to filter out zero value numerators (not compatible with log2)
        if num2 == 0:
            falseEntropy = 0
        else:
            # Entropy representing the false values in the 'attr' column
            falseEntropy = (num2/denom)*np.log2(num2/denom)
        # Subtract the two entropy values
        entropy -= trueEntropy
        entropy -= falseEntropy
        # Multiply by true or false over total and multiply by entropy, then add to summation.
        summation += (denom / total) * entropy

    # Information gain
    return getEntropy(df)-summation
"""
Helper function that returns the information gain of an attribute in a dataframe.
:param df: a pandas dataframe object containing a row of features and rows of values.
:param attr: the attribute that is being queried for information gain as a string.
:return: the calculated information gain value.
"""

def bestFeature(df, features):
    # Initialize a dictionary for info gain from each feature
    infoGain = {}
    # Current counter
    current = 0
    # Maximum counter
    maximum = 0
    # Maximum attribute string
    maxAttr = ""
    # Loop through feature
    for attr in features:
        # Set current to information gain from df, attr (current feature)
        current = getInfoGain(df, attr)
        # Append attribute to current index of infoGain dictionary
        infoGain[current] = attr
        # If current > maximum, set maximum to current and maxAttr to current attribute
        if current > maximum:
            maximum = current
            maxAttr = attr
    # If current is equal to maximum, return the first element because they are all the same
    # This splits the difference when coming to the case where two identical queries result in different
    # class values
    if current == maximum:
        # Possibly update to fix
        best = features[0]
        return best
    return maxAttr
"""
Helper function that returns the best feature in a particular dataset or partition of a data set
for partitioning, based on information gain.
:param df: a pandas dataframe object containing a row of features and rows of values.
:param features: a pandas index object containing the feature names.
:return: the best feature for data partitioning as a string.
"""

def mostCommonValue(df, className):
    targetValues = pd.value_counts(df[className].values.ravel())
    index = targetValues.index

    mostCommon = index[0]
    count = 0
    maxIndex = 0
    for val in targetValues:
        if val > count:
            count = val
            mostCommon = index[maxIndex]
        maxIndex += 1
    return mostCommon
"""
Helper function that returns the most common feature in a pandas dataframe. Useful
in building the decision tree using ID3.
:param df: a pandas dataframe object containing a row of features and rows of values.
:param className: the target class (last column name) in the dataframe.
:return: the most common value as a string.
"""

def buildTree(df, features, parentNode, pruning):
    # Get the target class name
    className = df.keys()[-1]

    # targetValues is a series containing each target level and their associated quantities
    targetValues = pd.value_counts(df[className].values.ravel())
    # index is the labels within target values (yes or no)
    index = targetValues.index

    # Length of the dataset
    rows = len(df)

    # Base cases:
    if index.size == 1:
        # All remaining instances in the partition have the same value (yes or no)
        # We reached the end of this tree branch. This becomes a leaf node with the class value
        return index[0]
    elif df.empty:
        # For a particular partition, there are no instances that have a feature value
        # Return the most common class value of the dataset of the parent node
        mostCommon = mostCommonValue(parentNode, className)
        return mostCommon
    elif len(features) == 0:
        # All features along this path have been tested (no more features to split on)
        mostCommon = mostCommonValue(df, className)
        return mostCommon
    elif rows <= 30 and pruning == True:
        # If the partition is <= 30 rows long, find the most common value in the data set and set that
        # as the leaf node. Pruning!
        mostCommon = mostCommonValue(df, className)
        return mostCommon
    else:
        # Recursive case:
        # Get the best feature for splitting the data set using information gain
        best = bestFeature(df, features)

        # Create a root node of this section of the tree using best feature.
        root = {best : {}}

        # Remove best from the list of features
        features = [i for i in features if i != best]

        # Get unique values of the best feature
        uniqueValues = df[best].unique()

        # Build the tree by looping through uniqueValues, partitioning, and initiating the recursive call.
        for val in uniqueValues:
            parent = df.copy()
            # Create a partition
            indexBool = df[best] == val
            partition = df[indexBool]
            #Begin the recursion
            subtree = buildTree(partition, features, parent, pruning)
            root[best][val] = subtree

        return root
"""
Tree building function that uses the ID3 algorithm to build a decision tree using the previously defined
helper functions. Function returns a nested dictionary object which can be visualized using graphviz library.
:param df: a pandas dataframe object containing a row of features and rows of values.
:param features: a pandas index object containing the feature names.
:param parentNode: a pandas dataframe object which is passed as the parent node in the decision tree.
:param pruning: a boolean value that determines whether or not to implement tree pruning.
:return: a nested dictionary object which represents a decision tree built from the input data set.
"""

def predict(tree, row):
    # If it is a leaf node
    if not isinstance(tree, dict):
        # Return value
        return tree
    else:
        # Get the root feature name from the tree
        root_node = next(iter(tree))
        # Get the value of the feature
        feature_value = row[root_node]
        # Check the feature value in the current tree node
        if feature_value in tree[root_node]:
            # Go to next feature
            return predict(tree[root_node][feature_value], row)
        else:
            return predict(tree[root_node][list(tree[root_node].keys())[0]], row)
"""
This function accepts a completed decision tree nested dictionary object, as well as
a row of a pandas dataframe (as a series object) and queries the decision tree to
determine the predicted class value.
:param tree: a dictionary object which represents a decision tree built with buildTree.
:param row: a row of a pandas dataframe object which is used to query the decision tree.
:return: a class value conclusion in the form of a string stored at a leaf node in the decision tree.
"""

def compareAgainst(df, tree):
    print("---------- TESTING STARTED ----------")
    print()
    className = df.keys()[-1]
    rows = len(df)
    correct = 0
    for i in range(rows):
        rowClassValue = df.loc[i, className]
        prediction = predict(tree, df.loc[i])

        if rowClassValue == prediction:
            correct += 1
    incorrect = rows - correct
    print("Number of testing examples: " + str(rows))
    print("Correct classificaton count: " + str(correct))
    print("Incorrect classification count: " + str(incorrect))
    accuracy = (correct/rows)*100
    print("Accuracy = " + str(accuracy) + "%")
    print()
    print("----------- TESTING ENDED -----------")
"""
This function accepts a pandas dataframe object and a completed decision tree
in order to iterate through the dataframe and compare its class values to the values
determined by the prediction function and the decision tree.
:param df: a pandas dataframe object containing a row of features and rows of values.
:param tree: a dictionary object which represents a decision tree built with buildTree.
:return: the most common value as a string.
"""

def treeBuilder(df, pruning):
    rows = len(df)
    print("---------- Training started on " + str(rows) + " examples with pruning = " + str(pruning) + ". -----------")
    attributes = df.keys()[:-1]
    parentNode = None
    tree = buildTree(df, attributes, parentNode, pruning)
    print("---------------------------- Finished training. ------------------------------")
    return tree
"""
Helper function that accepts a pandas dataframe object and a boolean value to determine whether
or not to implement tree pruning during the tree building operation. Prints useful console output information
and drives the buildTree function.
:param df: a pandas dataframe object containing a row of features and rows of values.
:param pruning: a boolean value which signals to the buildTree function whether to use pruning or not.
:return: the decision tree as a dictionary object.
"""

# Import all of the data sets.
tennis_df = pd.read_csv("assets/playtennis_dayremoved.csv")
census_df = pd.read_csv("assets/census_training.csv")
emails_df = pd.read_csv("assets/emails_noID.csv")
census_test = pd.read_csv("assets/census_training_test.csv")

# Build a tree from the census training dataset.
censusTree = treeBuilder(census_df, pruning = False)
# Build another tree from census training but with pruning this time.
censusTreePrune = treeBuilder(census_df, pruning = True)
#Build a tree from the tennis training dataset. Set pruning to false because the dataset is < 30 lines in total.
tennisTree = treeBuilder(tennis_df, pruning = False)

# Run the compareAgainst function to test the tree against the census_training_test dataset.
compareAgainst(census_test, censusTree)
print()
# Run the comparison again with pruning.
compareAgainst(census_test, censusTreePrune)
