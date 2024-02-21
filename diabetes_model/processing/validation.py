import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Tuple, Union

from diabetes_model.config.core import config
from diabetes_model.processing.data_manager import pre_pipeline_preparation

def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    # print(pre_processed.info(), config.model_config.features)    
    validated_data = pre_processed[config.model_config.features].copy()
    # print(validated_data.to_dict(orient="records"))
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        print(error)
        errors = error.json()

    return validated_data, errors

class DataInputSchema(BaseModel):
    Pregnancies:Optional[int]
    Glucose: Optional[float]
    BloodPressure: Optional[float]
    SkinThickness: Optional[float]
    Insulin: Optional[float]
    BMI: Optional[float]
    DiabetesPedigreeFunction: Optional[float]
    Age: Optional[float]



class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]