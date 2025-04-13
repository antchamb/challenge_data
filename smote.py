# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 14:30:39 2025

@author: dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

x_train = pd.read_csv(
    r'data\x_train.csv',
    index_col=0
    ).apply(pd.to_numeric, errors='coerce')

y_train = pd.read_csv(
    r'data\y_train.csv',
    index_col=0
    ).apply(pd.to_numeric, errors='coerce')


x_test = pd.read_csv(
    r'data\x_test.csv',
    index_col=0
    ).apply(pd.to_numeric, errors='coerce')



x_train_corr = x_train.corr()
x_train.boxplot()
plt.show()
# S2, R --> outliers ?

# for col in y_train.columns:
#     plt.hist(y_train[col], bins=20, range=(0, 1), color='skyblue', edgecolor='black')
#     plt.title(f'{col}')
#     plt.show()
    
y_train_bin= (y_train != 0).astype(int)

 

# for col in y_train_bin.columns:
#     plt.hist(y_train_bin[col], color='skyblue', edgecolor='black')
#     plt.title(f'{col}')
#     plt.show()


min_pos = 10
valid_cols = y_train_bin.columns[y_train_bin.sum() >= min_pos]
y_train_bin = y_train_bin[valid_cols]
print(f"Kept {len(valid_cols)} columns with at least {min_pos} positive samples")


X_smoted= []
y_smoted = []

for i, col in tqdm(enumerate(y_train_bin.columns)):
    print(f"Applying SMOTE on label: {col}")
    sm = SMOTE(random_state=42)
    
    # Fit on X and the current label
    X_resampled, y_resampled = sm.fit_resample(x_train, y_train_bin[col])
    
    if i == 0:
        X_smoted = X_resampled
        y_smoted = pd.DataFrame(y_resampled, columns=[col])

    else:
        y_smoted[col] = y_resampled


# Final results
X_balanced = X_smoted
y_balanced = y_smoted

for col in y_balanced.columns:
    counts = y_balanced[col].value_counts()
    print(f"{col} → 0s: {counts.get(0, 0)}, 1s: {counts.get(1, 0)}")


for col in y_balanced.columns:
    plt.hist(y_balanced[col], color='skyblue', edgecolor='black')
    plt.title(f'{col}')
    plt.show()

# check for a model by label and a smote by label to classify for each one and create synthetic map for gas
