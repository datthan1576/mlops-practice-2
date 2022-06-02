import click
import pandas as pd
import joblib as jb
from typing import List
import json
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
mlflow.set_experiment('lgbm1')


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path())
def evaluate(input_paths: List[str], output_path: str):
    test_df = pd.read_csv(input_paths[0])
    model = jb.load(input_paths[1])

    x_holdout = test_df.drop('price', axis=1)
    y_holdout = test_df['price']

    y_predicted = model.predict(x_holdout, num_iteration=model.best_iteration)
    score = dict(
            mae=mean_absolute_error(y_holdout, y_predicted),
            rmse=mean_squared_error(y_holdout, y_predicted)
        )
    mlflow.log_metrics(score)

    with open(output_path, "w") as score_file:
        json.dump(score, score_file, indent=4)


if __name__ == "__main__":
    evaluate()

