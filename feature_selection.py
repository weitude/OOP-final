train_path = "covid.train.csv"
test_path = "covid.test.csv"

import pandas as pd
import numpy as np

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

corr = train_data.iloc[:, :].corr().iloc[-1]
features = corr[abs(corr) > 0.86]
features_col = features.index.to_list()[:-1]
features_id = np.array([train_data.columns.to_list().index(i) for i in features_col]) - 1

print (features)
print ("\nfeatures' id:", repr(features_id))
