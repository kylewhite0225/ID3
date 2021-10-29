# Author: Kyle White
# Class: CS460 - Machine Learning
# Date: 10/29/2021

import pandas as pd
import numpy as np

def getEntropy(df):
    """
    Helper function that returns the calculated entropy of a dataset.
    :param df: a pandas dataframe object containing a row of features and rows of values.
    :return: the calculated entropy value.
    """
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

def getInfoGain (df, attr):
    """
    Helper function that returns the information gain of an attribute in a dataframe.
    :param df: a pandas dataframe object containing a row of features and rows of values.
    :param attr: the attribute that is being queried for information gain as a string.
    :return: the calculated information gain value.
    """
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

def bestFeature(df, features):
    """
    Helper function that returns the best feature in a particular dataset or partition of a data set
    for partitioning, based on information gain.
    :param df: a pandas dataframe object containing a row of features and rows of values.
    :param features: a pandas index object containing the feature names.
    :return: the best feature for data partitioning as a string.
    """
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
        best = features[0]
        return best
    return maxAttr

def majority(df, className):
    """
    Helper function that returns the most common feature in a pandas dataframe. Useful
    in building the decision tree using ID3.
    :param df: a pandas dataframe object containing a row of features and rows of values.
    :param className: the target class (last column name) in the dataframe.
    :return: the most common value as a string.
    """
    # Get a series containing each target level and their associated quantities
    targetValues = pd.value_counts(df[className].values.ravel())
    # Get the labels within target values (yes or no, etc.)
    index = targetValues.index

    # Set most common to the first one
    majority = index[0]
    # Counter to keep track of current target quantity
    count = 0
    # Index of maximum value
    maxIndex = 0
    # Loop through target values to find which one has the most occurrences
    for val in targetValues:
        # If the val > count
        if val > count:
            # Set count to val
            count = val
            # Set most common to the current index
            majority = index[maxIndex]
        # Increment maxIndex
        maxIndex += 1
    return majority

def buildTree(df, features, parentNode, pruning):
    """
    Tree building function that uses the ID3 algorithm to build a decision tree using the previously defined
    helper functions. Function returns a nested dictionary object which can be visualized using graphviz library.
    :param df: a pandas dataframe object containing a row of features and rows of values.
    :param features: a pandas index object containing the feature names.
    :param parentNode: a pandas dataframe object which is passed as the parent node in the decision tree.
    :param pruning: a boolean value that determines whether or not to implement tree pruning.
    :return: a nested dictionary object which represents a decision tree built from the input data set.
    """
    # Get the target class name
    className = df.keys()[-1]

    # targetValues is a series containing each target level and their associated quantities
    targetValues = pd.value_counts(df[className].values.ravel())
    # index is the labels within target values (yes or no, etc.)
    index = targetValues.index

    # Length of the dataset (for pruning)
    rows = len(df)

    # Base cases:
    if index.size == 1:
        # All instances in the partition have the same target level (yes or no),
        # return a leaf node with the label of this target level (class value).
        return index[0]
    elif len(features) == 0:
        # If features list is empty, return a leaf node with the label of
        # the majority class value in this partition.
        return majority(df, className)
    elif df.empty:
        # If this partition is empty, return a leaf node with the label of the majority
        # target class value in the parent of this partition.
        return majority(parentNode, className)
    elif rows <= 30 and pruning == True:
        # If the partition is <= 30 rows long, find the most common class value in this
        # partition and set that as the leaf node. Pruning! Enabled via pruning boolean.
        return majority(df, className)
    else:
    # Recursive case:
        # Get the best feature for splitting the data set using information gain
        best = bestFeature(df, features)
        # Create a root node of this section of the tree using best feature.
        root = {best : {}}
        # Remove best from the list of features
        features = [feature for feature in features if feature != best]
        # Get unique values of the best feature
        uniqueValues = df[best].unique()

        # Build the tree by looping through uniqueValues, partitioning, and initiating the recursive call.
        for val in uniqueValues:
            # Create a copy of the current partition to keep as a parent node for use in the base cases.
            parent = df.copy()
            # Create a partition
            partition = df.where(df[best] == val).dropna()
            # Begin the recursion
            subtree = buildTree(partition, features, parent, pruning)
            # Assign the subtree to best,val indices of the root
            root[best][val] = subtree
        return root

def predict(tree, row):
    """
    This function accepts a completed decision tree nested dictionary object, as well as
    a row of a pandas dataframe (as a series object) and queries the decision tree to
    determine the predicted class value.
    :param tree: a dictionary object which represents a decision tree built with buildTree.
    :param row: a row of a pandas dataframe object which is used to query the decision tree.
    :return: a class value conclusion in the form of a string stored at a leaf node in the decision tree.
    """
    # If it is a leaf node
    if not isinstance(tree, dict):
        # Return value
        return tree
    else:
        # Get the root feature name from the tree
        root = next(iter(tree))
        # Get the value of the feature
        featureValue = row[root]
        # Check the feature value in the current tree node
        if featureValue in tree[root]:
            # Go to next feature
            return predict(tree[root][featureValue], row)
        else:
            return predict(tree[root][list(tree[root].keys())[0]], row)

def compareAgainst(df, tree):
    """
    This function accepts a pandas dataframe object and a completed decision tree
    in order to iterate through the dataframe and compare its class values to the values
    determined by the prediction function and the decision tree.
    :param df: a pandas dataframe object containing a row of features and rows of values.
    :param tree: a dictionary object which represents a decision tree built with buildTree.
    :return: the most common value as a string.
    """
    print("---------- TESTING STARTED ----------")
    print()
    # Obtain class label name from dataframe (last column name)
    className = df.keys()[-1]
    rows = len(df)
    # Counter to track correct predictions
    correct = 0
    # Loop through the entire dataframe
    for i in range(rows):
        # Gather the entire row as a series
        rowClassValue = df.loc[i, className]
        # Pass series into predict function to get prediction value as a string
        prediction = predict(tree, df.loc[i])
        # If the dataframe class value and prediction are equal, increment correct counter.
        if rowClassValue == prediction:
            correct += 1
    # Calculate incorrect values
    incorrect = rows - correct
    print("Number of testing examples: " + str(rows))
    print("Correct classificaton count: " + str(correct))
    print("Incorrect classification count: " + str(incorrect))
    # Calculate accuracy percentage
    accuracy = (correct/rows)*100
    print("Accuracy = " + str(accuracy) + "%")
    print()
    print("----------- TESTING ENDED -----------")

def treeBuilder(df, pruning):
    """
    Helper function that accepts a pandas dataframe object and a boolean value to determine whether
    or not to implement tree pruning during the tree building operation. Prints useful console output information
    and drives the buildTree function.
    :param df: a pandas dataframe object containing a row of features and rows of values.
    :param pruning: a boolean value which signals to the buildTree function whether to use pruning or not.
    :return: the decision tree as a dictionary object.
    """
    rows = len(df)
    print("---------- Training started on " + str(rows) + " examples with pruning = " + str(pruning) + ". -----------")
    # Create list of features
    features = df.keys()[:-1]
    # Set parent to None - no parent yet
    parentNode = None
    # Create tree object using buildTree function
    tree = buildTree(df, features, parentNode, pruning)
    print("---------------------------- Finished training. ------------------------------")
    return tree

# --------------------------------------------------------------------------------------- #

# Import all of the data sets.
tennis_df = pd.read_csv("assets/playtennis_dayremoved.csv")
census_df = pd.read_csv("assets/census_training.csv")
emails_df = pd.read_csv("assets/emails_noID.csv")
census_test = pd.read_csv("assets/census_training_test.csv")

# Build a tree from the census training dataset.
censusTree = treeBuilder(census_df, pruning = False)
# Build another tree from census training but with pruning this time.
censusTreePrune = treeBuilder(census_df, pruning = True)
# Build a tree from the tennis training dataset. Set pruning to false because the dataset is < 30 lines in total.
tennisTree = treeBuilder(tennis_df, pruning = False)

# Run the compareAgainst function to test the tree against the census_training_test dataset.
compareAgainst(census_test, censusTree)
print()
# Run the comparison again with pruning.
compareAgainst(census_test, censusTreePrune)
