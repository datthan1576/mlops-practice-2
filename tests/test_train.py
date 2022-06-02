from src import train
from click.testing import CliRunner
import pytest
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Initialize runner
runner = CliRunner()


def test_cli_command():
    result = runner.invoke(train, 'data/processed/train.csv data/processed/test.csv models/model.clf reports/metrics.json')
    assert result.exit_code == 0

# def test_score():
#     result = runner.invoke(train, 'data/processed/train.csv data/processed/test.csv models/model.clf reports/metrics.json')
#     mlflow.get_artifact_uri()
#     client = MlflowClient()
#     experiment_id = "0"
#     run = client.create_run(experiment_id)
#     print("run_id:{}".format(run.info.run_id))
#     print("--")
#     # accuracy = train(model, inputs=batches[0])
#     # assert accuracy == pytest.approx(0.95, abs=0.05) # 0.95 ± 0.05
