# Importing important Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset with pandas
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Encoding the data
from mlxtend.preprocessing import TransactionEncoder
trans = TransactionEncoder()
trans_array = trans.fit(transactions).transform(transactions)
df_X = pd.DataFrame(trans_array, columns = trans.columns_)

# Applying model to the encoded dataset
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df_sets = apriori(df_X, min_support = 0.05, use_colnames = True)
df_rules = association_rules(df_sets, metric = 'support', min_threshold = 0.05, support_only = True)