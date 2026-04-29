from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

#Dummy dataset
data=pd.DataFrame({
    'hours_studied': [1,2,3,4,5,6,7,8],
    'sleep_hours':[8,7,6,6,5,5,4,4],
    'pass':[0,0,0,1,1,1,1,1]
})
X=data[['hours_studied','sleep_hours']]
y=data['pass']

pipeline=Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X,y)
joblib.dump(pipeline, 'model_pipeline.pkl')
print("Model saved successfully!")