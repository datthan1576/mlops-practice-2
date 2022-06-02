from src import get_osm_cafes_data
from click.testing import CliRunner

# Initialize runner
runner = CliRunner()


def test_cli_command():
    result = runner.invoke(get_osm_cafes_data, ["https://maps.mail.ru/osm/tools/overpass/api/interpreter?data=[out:json];nwr['addr:street'='Лиговский проспект']['addr:housenumber'=101];node[amenity=cafe](around:25000);out geom;", "data/external/data_cafes.geojson"])
    assert result.exit_code == 0

