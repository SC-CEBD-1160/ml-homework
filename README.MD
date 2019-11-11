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

I tried using LinearSVC and SVC for classification of the wine dataset 
Here are the results 

##### LinearSVC
Overall f1-score **0.6997732426303855**

**Description:**
Implements “one-vs-the-rest” multi-class strategy, thus training n_class models. If there are only two classes, only one model is trained
implementation of Support Vector Classification for the case of a linear kernel. Note that LinearSVC does not accept keyword kernel, as this is assumed to be linear

##### SVC
Overall f1-score **0.9057659483191398**

Playing with the parameters of the SVC function has always resulted in F1 values that were lower

**description**
The fit time scales at least quadratically with the number of samples

Obviously the SVC has a better Fitness score than LinearSVC but compared to LogisticRegression which has a
Overall f1-score **0.9539161610590181**

The best algorithm for the classification is LogisticRegression

#### As an extra for fun I added the dataset vizualisation
The interesting thing about it is that 2 of the classes are kind of blurred together and one is quite separate
