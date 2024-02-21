install:
	pip install --upgrade pip && pip install -r requirements/requirements.txt
format:
	black *.py
lint:
	pylint *.py

## Vikrant added
#  model training, testing, package building, api dockerizing, and docker image pushing
train: 
	python diabetes_model/train_pipeline.py  

# test step - already present below this portion
predict:
	python diabetes_model/predict.py

build: pip install build
		python -m build

copy_package:
	cp .\dist\*.whl .\diabetes_prediction_API\

all: install train predict build copy_package