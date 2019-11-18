## Introduction

Traditional investing has primarily been done through an allocation of risky assets (e.g. stocks and ETFs), and relatively riskless assets (e.g. bonds and certificates of deposit). Today, historically low interest rates are the norm, and a significant amount of debt today is issued with negative nominal interest rates. Even in the US, government-issued treasuries are being offered with yields below inflation. With stock markets near all-time highs amidst an ongoing trade war and slowing global growth, it can be difficult to justify an investment in the stock market in the medium term. Fortunately for the more risk-averse investor, alternative forms of investment have emerged through companies such as LendingClub, a fintech firm that provides a platform for peer-to-peer lending and is among the largest in this space.

We tackle two problems of interest. 

1. Predict bad loans, i.e. loans that are in default or charged-off. This is a classic problem which banks usually tackle to minimize losses. 
2. Optimize the investor's portfolio for maximum returns.

Loans on LendingClub are rated based on their interest rates, though it is expected that a higher percentage of these "subprime" loans with higher rates eventually default and/or are charged-off, resulting in a loss for the investor. Hence a balance must be made between risk and reward.

## Bad loans

The entire dataset shows us that the default rate is about 12%, so a model picking loans at random will be expected to have this miss rate (the base case). By loan grade, the default rates steadily increase, as expected. Keep in mind that LendingClub no longer offers loans of grades E, F, and G.

| Loan grade | Percent of total loans (%) | Default rate (%) |
| ---- | ---- | ---- |
| A | 19.15 | 3.3 |
| B | 29.35 | 7.9 |
| C | 28.75 | 13.2 |
| D | 14.35 | 18.9 |
| E | 6.00 | 26.7 |
| F | 1.85 | 34.9 |
| G | 0.54 | 38.1 |

The data preprocessing is conducted in detail in the `BadLoanPrediction.ipynb` notebook. The most important factors to keep in consideration are summarized as follows:
* Drop features not known to the investor at the time of offering. This prevents us from "cheating" with features like the last payment date.
* Deal with the imbalanced data. Since the dataset has about 2.2 million entries, we try random undersampling to resample with a 50/50 ratio of good and bad loans.
* Properly treat categorical features. This means that we elected to interpolate the subgrades so that they are consistent with the ordinally encoded grades feature. To avoid dimensional blowup, the addresses were one-hot encoded into the four US Census designated regions.
* Deal with missing data. Employment length is the feature with the most amount of missing data (6.5%); all others have less than 0.1% missing. We reason that missing employment length data is more likely than not due to borrowers with no significant employment history. The others can reasonably be imputed using their medians.

We constructed three models to predict bad loans. As this is a binary classification, the first model to try is logistic regression. We also used the more sophisticated random forests and deep neural networks. As we want to minimize the amount of bad loans selected, the recall and miss rates will be more important to track than accuracy and precision. The miss rate, taking into account true positives of label=0, will not be fully inversely related to the recall rate. Notice that the logistic regression is already performing twice as well as picking loans at random.

| Model | Recall (%) | Miss rate (%) | 
| ---- | ---- | ---- |
| Logistic Regression | 66.1 | 6.1 |
| Random Forest Classifier | 70.3 | 5.8 |
| 6-layer 48-16-64-128-16-1 | 79.3 | 4.5 |

The feature importance visualization as extracted from the random forest confirms that the interest rate and loan subgrade are most important, in addition to DTI, revolving balance/utilization, and annual income.
