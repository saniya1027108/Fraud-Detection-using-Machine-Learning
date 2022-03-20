# Fraud-Detection-using-Machine-Learning

This notebook contains Exploratory Data Analysis and Predictive Machine Learning Model for fraud detection. Fraud detection is valuable to many industries including the banking-financial sectors, insurance, law enforcement, government agencies, and many more.

In recent years we have seen a huge increase in Fraud attempts, making fraud detection important as well as challenging. Despite countless efforts and human supervision, hundreds of millions are lost due to fraud. Fraud can happen using various methods ie, stolen credit cards, misleading accounting, phishing emails, etc. Due to small cases in large population detection of fraud is important as well as challenging.

Data mining and machine learning help to foresee and rapidly distinguish fraud and make quick move to limit costs. Using data mining tools, a huge number of transactions can be looked to spot pattern and distinguish fraud transactions.


## Algorithms used:

`1. RANDOM FORESTS`

The method leverages a set of randomized decision trees and averages across their predictions to create outputs. It has multiple trees producing different values and this prevents the algorithm from overfitting to training datasets (something standard decision tree algorithms tend to do) and makes it more robust to noise.

Numerous comparative studies have been published that prove RF’s effectiveness in fraud detection relative to other models. The results of this research show that an RF-based model outperforms a support vector machine and even a neural network in terms of AP, AUC, and PrecisonRank metrics (all of the models made predictions on a real transaction data from a Belgian payment provider.)

`2. K-NEAREST NEIGHBORS`

The algorithm predicts which class an unseen instance belongs to, based on K (a predefined number) most similar data objects. The similarity is typically defined by Euclidean distance but, for specific settings, Chebyshev and Hamming distance measures can be applied too, when that’s more suitable.

So, after being given an unseen observation, KNN runs through the entire annotated dataset and computes the similarity between the new data object and all other data objects in it. When an instance has similarities with objects in different categories, the algorithm picks the class that has the most votes. If K=10, for example, and the object has 7 nearest neighbors in category a , and 3 nearest neighbors in category b – it will be assigned to a .

Though quite sensitive to noise, KNN performs well on real financial transaction data. Over the years, studies have demonstrated that KNNs can have a lower error rate than Decision Trees and Logistic Regression models, and that they can beat Support Vector Machines in terms of fraud detection rates (sensitivity) as well as Random Forests in balance classification rate.

`3. LOGISTIC REGRESSION`

An easily explainable model that enables us to predict the probability of a categorical response based on one or a few predictor variables. LR is quick to implement, which might make it seem like an attractive option. However, the empirical evidence shows it performs poorly when dealing with non-linear data and that it tends to overfit to training datasets.

This paper, for instance, describes how neural nets have a clear edge over LR-based models in solving credit card fraud detection problems. Similarly, this comparative research states LR can’t provide predictions as accurate as those produced by a deep learning model and Gradient Boosted Tree (for this experiment, the researchers had all three models making predictions on a dataset containing about 80 million transactions with 69 attributes.)

`4. SUPPORT VECTOR MACHINES`

SVMs, advanced yet simple in implementation, derive optimal hyperplanes that maximize a margin between classes. They utilize kernel functions to project input data onto high-dimensional feature spaces, wherein it’s easier to separate instances linearly. This makes SVMs particularly effective in terms of non-linear classification problems such as financial fraud detection.

In this study, the performance of an SVM in investigating a time-varying fraud problem is compared to that of a neural net. The researchers write that though the models show similar results (in terms of accuracy) during training, the neural net tends to overfit to training datasets more which makes the SVMs a superior solution in the long run.

Another research (by Lu & Ju) says that an imbalance class weighted SVM-based fraud detection model is more suitable for working with real-world credit card transactional data (which is imbalance in nature) and shows higher accuracy rates in the fraud detection problem than Naive Bayes, Decision Tree, and Back Propagation Neural Network classifiers.

It should be noted though that while SVMs work great in complicated domains that have distinct margins of separation, their performance on large data sets is generally average. If there’s noise in data, it can hamper SVM’s accuracy tremendously, so when there are many overlapping classes (and we need to count independent evidence) other supervised algorithms would probably make a better choice.


