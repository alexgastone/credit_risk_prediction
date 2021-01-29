# Credit Risk Predictor and Explainer

## Context

Credit cards are, in essence, models that attempt to quantify the probability that a customer will exhibit certain financial behaviors, namely default on a loan or pay on time. This process of quantification is also called credit scoring, and gives an estimate of risk that a particular customer will exhibit.

Many models of credit scoring exhist - from the simplest with linear regression, to more complex hazard models.

A lot of the variables that go into credit scoring, however, are not always very straightforward. A common misconception is that the only data used in credit scoring is past history of payments. 

**Therefore, the goal of this project is to better understand both the credit score prediction process and the effect of certain variables on that score.**

## Data

The [dataset](https://www.kaggle.com/rikdifos/credit-card-approval-prediction), available from Kaggle, includes (anonymized) personal information and data submitted by credit card applicants, labeled with the status of their defaults.

These labels, or targets credit risk categories, are : 

* No loan / paid off
* 1-29 days past due
* 30-59 days past due
* 60-89 days past due
* 90-119 days past due
* 120-149 days past due
* overdue more than 150 days

Typical data from the applicant includes whether they own a property, a car, marital status, number of children...

## Steps

1. Data cleaning
2. Data transformations (handling unbalanced classes, variable encoding, outliers...)
3. Train models:
	* Logistic Regression
	* KNeighbors
	* Support Vector Classifier
	* Decision Tree
	* Random Forest
	* XGBoost
4. Compare performance on CV folds
5. Hyperparameter tuning
6. Final training with best model parameters
7. Predict on test set and check performance
8. Create and customize [Explainer Dashboard](https://medium.com/analytics-vidhya/explainer-dashboard-build-interactive-dashboards-for-machine-learning-models-fda63e0eab9) with chosen model
9. Deploy Dashboard with Flask (gunicorn)

*Currently: Further feature engineering to increase performance on F1 score, explore other data augemtation techniques (or re-balancing of classes).*
