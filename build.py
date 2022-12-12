import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv('Zomato_clean.csv')

# dropping the index column
df.drop('Unnamed: 0', axis=1, inplace=True)

X = df.drop('rate', axis=1)
y = df['rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

et_model = ExtraTreesRegressor(n_estimators=120)
et_model.fit(X_train,y_train)

# print(et_model.predict(X_test))
# print(r2_score(y_test, et_model.predict(X_test)))

pickle.dump(et_model, open('model.pkl', 'wb'))