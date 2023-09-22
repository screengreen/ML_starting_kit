import pandas as pd
import yaml
from datetime import datetime
from comet_ml import Experiment


#opening config file
options_path = 'config/config.yaml'
with open(options_path, 'r') as option_file:
    options = yaml.safe_load(option_file)


def find_minority_and_majority_classes(y):
    class_counts = y.value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    return minority_class, majority_class


def load_data(type='train'):
    #загрузка данных 
    if type == 'train':
        data= pd.read_csv(options['train_path'])
    else:
        data = pd.read_csv(options['test_path'])

    # y_train = data_train['target']
    # X_train = data_train.drop(columns='target')
    return  data

def get_options():
    options_path = 'config/config.yaml'
    with open(options_path, 'r') as option_file:
        options = yaml.safe_load(option_file)
    return options

def get_model_options():
    if options['task_type'] == 'clasfic':
        return options['classification']
    elif options['task_type'] == 'clust':
        return options['clusterisation']
    
def start_experiment():

    if options['logging']:
        # set esperiment 
        experiment = Experiment(
            api_key=options['comet_api_key'],
            project_name=options['comet_project_name'],
            workspace=options['comet_workspace']
        ) 
        sub_options = get_model_options()['model']
        #set experiment name with time
        now = datetime.now()
        experiment.set_name(f"{sub_options['model_name']}-{now.day}-{now.hour}:{now.minute}:{now.second}")

        return experiment
    else: 
        return 0

def end_experiment(experiment, model):
    sub_options = get_model_options()['model']
    model_name = sub_options['model_name']
    if options['logging']:
        from comet_ml.integration.sklearn import log_model

    if options['save_model']:
        if options['logging'] and sub_options['model_class'] == 'sklearn':
            log_model(experiment, model_name ,model)
        else:
            print('did not save the model, as logging is off or model is not sklearn')

        # with open(model_name + '.pkl', 'wb') as model_file:
        #   pickle.dump(model, model_file)
    if options['logging']:
        experiment.end()

    print('\n\n\nMODEL HAS TRAINED\n\n\n')

def make_logs(experiment, params):
    model_options = get_model_options()
    if options['logging']:
        experiment.log_parameters(params)
        experiment.log_parameter('model_name', model_options['model']['model_name']) ###########
    
    

