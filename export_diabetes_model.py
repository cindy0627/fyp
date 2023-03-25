import pandas as pd
import warnings

warnings.filterwarnings("ignore")

df_final = pd.read_csv('final_testset1_diabetes.csv')
X_ = df_final.loc[:, df_final.columns != 'DIQ010']
y = df_final.DIQ010

from sklearn.preprocessing import MinMaxScaler
import joblib

minmax = MinMaxScaler()
X = pd.DataFrame(minmax.fit_transform(X_), columns=X_.columns)
joblib.dump(minmax, 'minmax_dia.gz')

# oversample
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
X_train_sm = pd.DataFrame(X_train_sm, columns=X.columns)

from sklearn.linear_model import LogisticRegression

best_random = LogisticRegression(C=7.853887219515014, max_iter=100, solver='liblinear', random_state=11)
best_random.fit(X_train_sm, y_train_sm)
y_pred_lr2 = best_random.predict(X_test)

# save
joblib.dump(best_random, 'lr_clf_dia.pkl')
