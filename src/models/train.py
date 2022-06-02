import click
import pandas as pd
import joblib as jb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from typing import List
import mlflow
from mlflow.models.signature import infer_signature
import os
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(remote_server_uri)

FEATURES = ['price', 'geo_lat', 'geo_lon', 'building_type', 'level', 'levels',
            'area', 'kitchen_area', 'object_type', 'year', 'month',
            'level_to_levels', 'area_to_rooms', 'cafes_0.012', 'cafes_0.08']


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path(), nargs=2)
def train(input_paths: List[str], output_path: List[str]):
    """
    Train the model and log params, metrics and artifacts in MLflow
    :param input_paths: train (for [0]) and test (for [1]) dataframes
    :param output_path: model (for [0]) and score (for [1]) artifact's path
    :return:
    """
    with mlflow.start_run():
        mlflow.get_artifact_uri()
        train_df = pd.read_csv(input_paths[0])
        test_df = pd.read_csv(input_paths[1])
        assert isinstance(train_df, pd.DataFrame), "input[0] must be a valid dataframe"
        assert isinstance(test_df, pd.DataFrame), "input[1] must be a valid dataframe"

        x_train = train_df.drop('price', axis=1)
        y_train = train_df['price']
        x_holdout = test_df.drop('price', axis=1)
        y_holdout = test_df['price']
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_holdout, y_holdout, reference=lgb_train)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l1'},
            'max_depth': 11,
            'num_leaves': 130,
            'learning_rate': 0.25,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'n_estimators': 1000,
            'bagging_freq': 2,
            'verbose': -1
        }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=200,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        early_stopping_rounds=30)  # categorical_feature=['building_type']
        jb.dump(gbm, output_path[0])

        y_predicted = gbm.predict(x_holdout, num_iteration=gbm.best_iteration)
        score = dict(
            mae=mean_absolute_error(y_holdout, y_predicted),
            rmse=mean_squared_error(y_holdout, y_predicted)
        )

        with open(output_path[1], "w") as score_file:
            json.dump(score, score_file, indent=4)

        signature = infer_signature(x_holdout, y_predicted)

        mlflow.log_params(params)
        mlflow.log_metrics(score)
        mlflow.lightgbm.log_model(lgb_model=gbm,
                                  artifact_path="model",
                                  registered_model_name="real_estate_lgbm",
                                  signature=signature)

    client = MlflowClient()
    experiment = dict(mlflow.get_experiment_by_name("Default")) #client.get_experiment_by_name("Default")
    experiment_id = experiment['experiment_id']
    df = mlflow.search_runs([experiment_id])
    best_run_id = df.loc[0, 'run_id']
    print(best_run_id)

    # metrics = client.get_metric_history(runID, metricKey)



if __name__ == "__main__":
    train()
