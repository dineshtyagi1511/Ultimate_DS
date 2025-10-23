import pandas as pd 

df = pd.read_csv("Data/train.csv")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor , LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X = df.drop(columns=["ID","medv"])
y = df["medv"]

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

numeric_features = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'black', 'lstat']
categorical_features = ['chas', 'rad']

preprocessor = ColumnTransformer(
    transformers= [
        ("num",StandardScaler(),numeric_features),
        ("cat",OneHotEncoder(handle_unknown="ignore",drop="first"),categorical_features)
    ],remainder="passthrough"
)

X_train_processed= preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# TRAIN MODEL USING SGDREGRESSOR
sgd_reg = SGDRegressor(
    max_iter=1000,
    eta0=0.01, # Learning rate
    random_state=42
)

sgd_reg.fit(X_train_processed, y_train)

# TRAIN MODEL USING OLS
linear_reg_ols = LinearRegression()
linear_reg_ols.fit(X_train_processed, y_train)

