### Analysis of categorization algorithms for the wine dataset

In all cases I am using F1 Score which gives a better measurement for performance  

#### Advanced #2

Tested the **KNeighborsClassifier** with the following values of n_neighbors

1- n_neighbors = 1  with Overall f1-score **0.8233364487694027**
2- n_neighbors = 5 (default) with Overall f1-score **0.7153787878787878**
3- n_neighbors = 10 with Overall f1-score **0.7402085640457733**
4- n_neighbors = 20 with Overall f1-score **0.710163145956877**

Analysis indicates that the algorithm works better when there is only one neighbor for classification
I would guess that using 1 as parameter evaluate to a linear classification

#### Reach #1
##### LogisticRegression

This algorithm gives me a Overall f1-score of **0.9343489718583492**

After that I followed the scikit-learn algorithm cheatsheet and tried LinearSVC and SVC for classification of the wine dataset 
Here are the results 

##### LinearSVC
Overall f1-score **0.7608695652173912**

**Description:**
LinearSVC does not accept keyword kernel, as this is assumed to be linear

##### SVC
Overall f1-score **0.964957264957265**

Playing with the parameters of the SVC function has always resulted in F1 values that was lower **0.7145352900069882**
Meaning that you have to know what you are doing before making changes to the default values

**description**

Obviously the SVC has a better Fitness score than LogisticRegression

SVC Overall f1-score **0.964957264957265**
LogisticRegression Overall f1-score **0.9343489718583492**

The best algorithm for the classification is SVC

#### As an extra for fun I added the dataset vizualisation
The interesting thing about it is that 2 of the classes are kind of blurred together and one is quite separate
