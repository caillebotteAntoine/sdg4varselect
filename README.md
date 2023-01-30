

poetry add lib

poetry install
poetry init


conda activate sdg

pytest --cov-report term-missing --cov=myproj tests/
pytest --cov-report term --cov=myproj tests/