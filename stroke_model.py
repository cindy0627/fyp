import pandas as pd
import warnings

warnings.filterwarnings("ignore")

df_final = pd.read_csv('final_testset1_stroke.csv')
X_ = df_final.loc[:, df_final.columns != 'MCQ160F']
y = df_final.MCQ160F

from sklearn.preprocessing import MinMaxScaler
import joblib

minmax = MinMaxScaler()
X = pd.DataFrame(minmax.fit_transform(X_), columns=X_.columns)
joblib.dump(minmax, 'minmax_stroke.gz')

# oversample
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
X_train_sm = pd.DataFrame(X_train_sm, columns=X.columns)

from sklearn.ensemble import RandomForestClassifier

best_random = RandomForestClassifier(n_estimators=100, max_depth=5, bootstrap=True, random_state=42)
best_random.fit(X_train_sm, y_train_sm)

# save the model
joblib.dump(best_random, 'rnd_clf_stroke.pkl')
