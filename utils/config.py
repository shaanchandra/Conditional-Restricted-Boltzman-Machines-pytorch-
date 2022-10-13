import logging
import os
import json
from unicodedata import decimal


class Config:
    def __init__(self, jsonpath):
        self.job_name = 'ClinicalGAN_job'
        self.jsonpath = jsonpath

        # read the JSON file
        with open(jsonpath, 'r') as j:
            self.input_argument = json.loads(j.read())
        
        self.sourceDetails = self.input_argument["sourceDetails"] 
        # self.loggingPaths = self.input_argument['loggingPaths']
        # self.columnsDescription = self.input_argument["columnsDescription"] 
        # self.commonModelParameter = self.input_argument["commonModelParameter"] 
        # self.generatorParam = self.input_argument["generatorParam"]
        # self.discriminatorParam = self.input_argument["discriminatorParam"]
    
    def load_config(self, params, config_type=None, immutable_params=None):
        try:
            for key in params:
                if immutable_params is not None and key in immutable_params:
                    continue
                value = params[key]
                if not isinstance(value, list):
                    value = int(value) if str(value).isdigit() else value
                
                if config_type is None:
                    setattr(self, key, value)
                else:
                    prop = getattr(self, config_type)
                    prop[key] = value
                logging.debug("Config {key} updated with value: {value}".format(key=key, value=value))
            logging.debug(self.__dict__)
        except Exception as e:
            logging.error(e)

if __name__ == "__main__":
    # temp test script
    PKS_TEMP_PATH = r"C:\Users\PP9596\Documents\Bitbucket\00_ai_coe\seqdatagen\model.json" 
    confi = Config(PKS_TEMP_PATH)
    confi.load_config((confi.__dict__))
    print("Parameters - ", confi.__dict__)
    

    