from src import select_region
from click.testing import CliRunner

# Initialize runner
runner = CliRunner()


def test_cli_command():
    result = runner.invoke(select_region, 'data/raw/all_v2.csv data/interim/data_regional.csv 2661')
    assert result.exit_code == 0

