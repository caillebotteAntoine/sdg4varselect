
poetry add lib

poetry install
poetry init


cd projects/sdg4varselect
conda activate sdg

pytest --cov-report term-missing --cov=sdg4varselect tests/
pytest --cov-report term --cov=sdg4varselect tests/