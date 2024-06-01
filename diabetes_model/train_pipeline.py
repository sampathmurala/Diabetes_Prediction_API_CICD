import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import optuna
from optuna import logging

from diabetes_model.config.core import config
from diabetes_model.processing.data_manager import load_dataset, save_pipeline

# Set the logging level to WARNING to suppress INFO messages
# logging.set_verbosity(logging.WARNING)

def objective(trial):
    # Define the search space for hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', config.model_config.max_depth_low, config.model_config.max_depth_high),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', config.model_config.n_estimators_low, config.model_config.n_estimators_high),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0, step=0.1),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10, step=1)
    }


    # Create a pipeline with StandardScaler and XGBoost Classifier
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(**params, random_state=config.model_config.random_state)),
    ])

    data = load_dataset(file_name=config.app_config.training_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size, 
        random_state=config.model_config.random_state,
    )
    
    # Fit the model on training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

     # Apply threshold for classification
    # y_pred_labels = ["Yes" if p >= config.model_config.threshold else "No" for p in y_pred]
    # print(y_pred, y_pred_labels)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def run_training() -> None:
    
    """
    Train the model.
    """
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    # print('read training data', data.shape)
    # print(data.info())
    
    # divide train and test
    # print('divide into train and test sets...')
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size, 
        random_state=config.model_config.random_state,
    )
    
    # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=config.model_config.no_trials)  # You can adjust the number of trials

    # Get the best hyperparameters
    best_params = study.best_params
    # print("\nBest Hyperparameters:", best_params)

    # Build the final model using the best parameters
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(**best_params, random_state=config.model_config.random_state)),
    ])

    
    # Pipeline fitting
    final_model.fit(X_train, y_train)
    
    # print("After fit....") 
    # print(X_test.head(2), X_test.info())
    y_pred = final_model.predict(X_test)
    # y_pred_labels = ["Yes" if p >= config.model_config.threshold else "No" for p in y_pred]
    # print(y_pred, y_pred_labels)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on test set: {accuracy:.4f}")
 
    # persist trained model
    save_pipeline(pipeline_to_persist= final_model)
    # printing the score


if __name__ == "__main__":
    run_training()