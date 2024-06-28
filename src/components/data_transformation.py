import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from data_ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path =  os.path.join("artifacts","pre-processor.pkl")

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            numerical_column  = ["writing_score","reading_score"]
            categorical_column = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(

                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ine_hot_encoder",OneHotEncoder()),
                    ("scaling",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Features in the dataset-{categorical_column}")
            logging.info(f"Categorical Features in the dataset-{categorical_column}")

            preprocessor = ColumnTransformer(

                [
                    ("num_pipeline",numeric_pipeline,numerical_column),
                    ("cat_pipeline",categorical_pipeline,categorical_column)

                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)    


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading of Train and Test data completed")
            logging.info("Fettching Preprocessor Obejct")
            preprocessor_object = self.get_data_transformation_object()
            target_column="math_score"
            numerical_columns = ["writing_score","reading_score"]
            input_feature_training_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_training_df = train_df[target_column]
            input_feature_testing_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_testing_df = test_df[target_column]
            logging.info("Applying Pre Processing activities on Training and testing data")
            input_feature_train_arr = preprocessor_object.fit_transform(input_feature_training_df)
            input_feature_test_arr = preprocessor_object.transform(input_feature_testing_df)
            training_arr = np.c_[input_feature_train_arr,np.array(target_feature_training_df)]
            testing_arr = np.c_[input_feature_test_arr,np.array(target_feature_testing_df)]
            logging.info("Target Feature save now")
            save_object(file_path=self.data_transformation_config.preprocessor_object_file_path,obj =preprocessor_object)
            return(training_arr,testing_arr,self.data_transformation_config.preprocessor_object_file_path)
            

        except Exception as e:
            raise CustomException(e,sys)



if __name__ == "__main__":
    obj_di = DataIngestion()
    _,train_data,test_data = obj_di.initiate_data_ingestion()
    obj_dt = DataTransformation()
    obj_dt.initiate_data_transformation(train_data,test_data)
