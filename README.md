# Credit_Risk_Analysis

## Overview

Using skills in data preparation, statistical reasoning, and machine learning, we will solve a real-world challenge of credit card risk. We will need to employ different techniques to train and evaluate models with unbalanced classes. In order to build and evaluate models using resampling, imbalanced-learn and scikit-learn libraries will be important.

Using the credit card credit dataset from LendingClub, we will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, we'll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 

## Resources
Data Source: LoanStats_2019Q1.csv <br />
Software: Python, Jupyter Notebook

### Results
There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models (15 pt)
Instead of using a bulleted list to describe the balanced/imbalanced accuracy score, precision score, and recall score, I think a table would be a better visual representation. 

- Naive Random Oversampling
  - Balanced Accuracy Score: 0.66
  - Precision: 72/(72 + 6605) = 0.01
  - Recall: 72/(72 + 29) = 0.71
 
 - SMOTE Oversampling
    - Balanced Accuracy Score: 0.66
    - Precision: 64/(64 + 5291) = 0.01
    - Recall: 64/(64 + 37) = 0.63
  
 - Undersampling
   - Balanced Accuracy Score: 0.54
   - Precision: 70/(70 + 10340) = 0.01
   - Recall: 70/(70 + 31) = 0.69
  
 - SMOTEENN
   - Balanced Accuracy Score: 0.64
   - Precision: 73/(73 + 7411) = 0.01
   - Recall: 73/(73 + 28) = 0.72
  
 - Balanced Random Forest Classifier
   - Balanced Accuracy Score: 0.79
   - Precision: 71/(71 + 2153) = 0.03
   - Recall: 71/(71 + 30) = 0.70

 - Easy Ensemble AdaBoost Classifier
   - Balanced Accuracy Score: 0.93
    - Precision: 93/(93 + 983) = 0.09
    - Recall: 93/(93 + 8) = 0.92
  
## Summary 
There is a summary of the results (2 pt)
There is a recommendation on which model to use, or there is no recommendation with a justification
- Both Oversampling methods, Naive Random Oversampling and SMOTE Oversampling, had similar balanced accuracy scores of 0.66. SMOTEEN, which begins with SMOTE Oversampling also has a similar balancaed accuracy score of 0.64. It appears that using Oversampling does not result in accurate results. Undersampling proved to be worse with a balanced accuracy score of 0.54. The Easy Ensemble AdaBoost Classifier had the highest balanced accuracy score of 0.93. 
- A model that sequentially focuses on errors and repeats the process in the next model allows for the greatest accuracy. Also, the model that has the largest recall allows for the least amount of true negatives to be overlooked. In this case, I would recommend using the Easy Ensemble AdaBoost Classifier.
