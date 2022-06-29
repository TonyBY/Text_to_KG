import json
import logging
import os
from tqdm import tqdm
import pandas as pd

LOGGER_NAME='MISQA'
SEED = 1024

logger = logging.getLogger(LOGGER_NAME)

def init_logger(verbose: bool = False, log_file: str = ''):
    logging.getLogger().setLevel(logging.DEBUG)
    logger = logging.getLogger(LOGGER_NAME)
    
    if not len(logger.handlers):
        # log.info will always be show in console
        # log.debug will also be shown when verbose flag is set
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        if log_file != '':
            log_file_dir = '/'.join(log_file.split('/')[:-1])
            make_directory(log_file_dir)
            f_handler = logging.FileHandler(log_file)
            logger.addHandler(f_handler)
            logger.info("file handler added.")
    return logger

def read_json(json_path:str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)

def save_json(json_obj: dict, output_path:str='./data/processed_tweets.json'):
    saving_dir = '/'.join(output_path.split('/')[:-1])
    make_directory(saving_dir)
    jsonFile =  open(output_path, 'w')
    jsonFile.write(json.dumps(json_obj, indent=4, sort_keys=False))
    jsonFile.close()

def make_directory(dir: str):
    print(f"dir: {dir}")
    if dir[-1] != '/':
        dir = dir + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def drop_empty_rows_by_column(df, column):
    index_to_drop = []
    for index, row in tqdm(df.iterrows()):
        if row[column] == ''\
            or str(row[column]).lower() == 'none'\
            or pd.isna(row[column])\
            or str(row[column]).lower == '[]'\
            or row[column] == []:
            index_to_drop.append(index)
    df = df.drop(index=index_to_drop, inplace=False)
    return df