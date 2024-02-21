import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd 

from diabetes_model import __version__ as _version
from diabetes_model.config.core import config
from diabetes_model.processing.data_manager import load_pipeline
from diabetes_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
diabetes_model_pipe  = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    validated_data=validated_data.reindex(columns=config.model_config.features)
    # print(validated_data)

    predictions = diabetes_model_pipe.predict(validated_data)
    results = {"predictions": ["Yes" if p >= config.model_config.threshold else "No" for p in predictions][0],"version": _version, "errors": errors}
    # print(results) 
    
    return results

if __name__ == "__main__":
    data_in={ "Pregnancies":2,"Glucose": 115.863387,'BloodPressure': 56.410731, 'SkinThickness': 24.336736,'Insulin':94.385783,
             'BMI': 26.455940, 'DiabetesPedigreeFunction': 0.272682, 'Age':20.100494}
    data_in = pd.DataFrame([data_in])
    
    result = make_prediction(input_data=data_in)
    print(result)
