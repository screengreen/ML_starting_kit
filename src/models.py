from sklearn.model_selection import  cross_val_score
from sklearn.metrics import silhouette_score
from src.utils import  get_options, get_model_options
import optuna

options = get_options()

def use_optuna(model, X_train, y_train, cv = 3, n_trials = 100):
    def objective(trial):
        model_options = get_model_options()['model']
        params = {}
        
        if options['task_type'] == 'clasfic':
            if y_train is None:
                raise ValueError('Need target data for classification')

            if model_options['model_name'] in ['random_forest',  'grad_boost',  'xgboost', 'lightgbm' ]:
                sub_params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                    'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
                    'min_samples_split' : trial.suggest_float('min_samples_split', 0.1, 1),
                    'min_samples_leaf' : trial.suggest_float('min_samples_leaf', 0.1, 0.5)
                }
                params.update(sub_params)        
            if model_options['model_name'] in [ 'grad_boost',  'xgboost', 'lightgbm', 'catboost' ]:
                sub_params = {
                    "learning_rate" : trial.suggest_float('learning_rate', 0.001, 0.5, log=True)
                }
                params.update(sub_params) 

            if model_options['model_name'] == 'log_reg':
                sub_params = {
                    "C" : trial.suggest_loguniform('C', 1e-5, 1e5),
                    "penalty" : trial.suggest_categorical('penalty', ['l1', 'l2']),
                    "solver" : trial.suggest_categorical('solver', ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'])
                }
                params.update(sub_params) 
            
            if model_options['model_name'] == 'lightgbm':
                sub_params = {
                    'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                }
                params.update(sub_params) 
            
            if model_options['model_name'] == 'xgboost':
                sub_params = {
                    'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 1.0),
                }
                params.update(sub_params) 

            if model_options['model_name'] == 'catboost':
                sub_params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 1, 10),
                    'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-5, 1e1),
                    'border_count': trial.suggest_int('border_count', 1, 255),
                }
                params.update(sub_params) 
            if model_options['model_name'] == 'svm':
                sub_params = {
                    "kernel" :trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                    "C" : trial.suggest_loguniform('C', 1e-3, 1e3),
                    "gamma" : trial.suggest_loguniform('gamma', 1e-3, 1e3)
                    }
                params.update(sub_params) 
            
        elif options['task_type'] == 'cluster':

            if model_options['model_name'] == 'kmeans':
                # Оптимизация гиперпараметров для K-Means
                sub_params = {
                    "n_clusters" : trial.suggest_int('n_clusters', 2, 10)
                }
                params.update(sub_params) 
                
                
            elif model_options['model_name'] == 'hierarchical_clust':
                # Оптимизация гиперпараметров для иерархической кластеризации
                sub_params = {
                    "n_clusters" : trial.suggest_int('n_clusters', 2, 10),
                    "linkage" : trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
                }
                params.update(sub_params) 
                
                
            elif model_options['model_name'] == 'dbscan':
                # Оптимизация гиперпараметров для DBSCAN
                sub_params = {
                    "eps" : trial.suggest_float('eps', 0.1, 1.0),
                    "min_samples" : trial.suggest_int('min_samples', 2, 10)
                }
                params.update(sub_params) 
                
                
            elif model_options['model_name'] == 'mean_shift':
                # Оптимизация гиперпараметров для Mean Shift
                sub_params = {
                    "bandwidth" : trial.suggest_float('bandwidth', 0.1, 1.0)
                }
                params.update(sub_params) 
                
                

        optuna_model = model.set_params(**params)
        if options['task_type'] == 'clasfic':
            return cross_val_score(optuna_model, X_train, y_train, n_jobs=-1, cv=cv).mean()
        
        elif options['task_type'] == 'cluster':
            labels = optuna_model.fit_predict(X_train)
            return silhouette_score(X_train, labels)

    # Create study object and specify the direction is maximize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Get the best hyperparameters
    best_params = study.best_params
    best_score = study.best_value

    print(f"The best parameters are {best_params} with a score of {best_score}.")
    return best_params


def return_model():

    if options['task_type'] == 'clasfic':
        sub_options = options['classification']
        model_name = sub_options['model']['model_name']
        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            # Код для модели Random Forest
            model = RandomForestClassifier()
        
        elif model_name == "log_reg":
            from sklearn.linear_model import LogisticRegression
            # Код для модели Logistic Regression
            model = LogisticRegression()
            
        elif model_name == "grad_boost":
            from sklearn.ensemble import GradientBoostingClassifier
            # Код для модели Gradient Boosting
            model = GradientBoostingClassifier()
            
        elif model_name == "svm":
            from sklearn.svm import SVC
            # Код для модели Support Vector Machine (SVM)
            model = SVC()
            
        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            # Код для модели CatBoost
            model = CatBoostClassifier()
            
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            # Код для модели XGBoost
            model = XGBClassifier()
            
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            # Код для модели LightGBM
            model = LGBMClassifier()
            
        elif model_name == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            # Код для модели K-Nearest Neighbors (KNN)
            model = KNeighborsClassifier()
    

    elif options['task_type'] == 'cluster':
        sub_options = options['clusterisation']
        cluster_algorithm = sub_options['model']['model_name']
        if cluster_algorithm == "kmeans":
            from sklearn.cluster import KMeans
            # Код для алгоритма K-Means
            model = KMeans()  # Пример: 3 кластера
    
        elif cluster_algorithm == "hierarchical clust":
            from sklearn.cluster import AgglomerativeClustering
            # Код для алгоритма Hierarchical Clustering
            model = AgglomerativeClustering()  # Пример: 3 кластера
            
        elif cluster_algorithm == "dbscan":
            from sklearn.cluster import DBSCAN
            # Код для алгоритма DBSCAN
            cluster_model = DBSCAN()  # Пример: параметры eps и min_samples
            
        elif cluster_algorithm == "mean_shift":
            from sklearn.cluster import MeanShift
            # Код для алгоритма Mean Shift
            model = MeanShift()

    elif options['task_type'] ==  'regres':
        pass

    return model


def return_hparams(model, experiment, X_train, y_train=None, cv=5, n_trials=100):
    model_options = get_model_options()

    if options['use_optuna']:
        experiment.log_parameter('use_optuna', options['use_optuna'])
        params = use_optuna(model=model, X_train=X_train, y_train=y_train, cv = cv, n_trials=n_trials)
        return params
    else:
        params = model_options['model']['hparams'] 
        return params


# def train_model(model=model, X_train, y_train):
#     model.fit(X_train, y_train)

