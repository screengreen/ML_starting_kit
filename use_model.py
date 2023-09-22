from src.utils import load_data
from src.sub_main import data_preprocessing_block
import joblib
import pandas as pd

# loading, preprocessing and predicting for test dataset
model_name = ''

path = 'models/model-data_comet-sklearn-model.joblib'

loaded_model = joblib.load(path)
data_test = load_data('test')
X_test = data_preprocessing_block(data_test, 0, 'test', columns_to_drop=None)

#use model 
y_test_pred = loaded_model.predict(X_test)


#save predictions
# y_test_pred.to_csv('data/predictions/predictions.csv', index=False)