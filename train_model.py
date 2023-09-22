from src.sub_main import  full_prerocessing
import src.evaluation as ev
from src.utils import load_data, get_options, get_model_options, start_experiment, end_experiment, make_logs
from src.models import use_optuna, return_model, return_hparams

import numpy as np


#opening config file
options = get_options()

#set random state
if options['random_state']:
  np.random.seed(options['random_state'])

experiment = start_experiment()

data_train = load_data('train')

#data preprocessing and spliting
X_train, X_val, y_train, y_val = full_prerocessing(data_train,experiment = experiment, type = 'train', columns_to_drop=None)

#model 
model = return_model()
params = return_hparams(model=model, experiment=experiment, X_train=X_train, y_train=y_train, cv=3, n_trials=90)

model = model.set_params(**params)
model.fit(X_train, y_train)

#logging all params
make_logs(experiment, params)


#evaluating model
y_pred = model.predict(X_val)
ev.evaluate_classification_model(y_val, y_pred, experiment)

#ending experiment
end_experiment(experiment, model)


