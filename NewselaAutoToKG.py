import json
import nltk
nltk.download('punkt')
import os

from pyopenie import OpenIE5

import pandas as pd
from tqdm import tqdm
from utils import read_json, save_json, logger, init_logger, drop_empty_rows_by_column
from NewselaManualToKG import get_best_triplet_of_sentence, get_data_dict

import argparse

def define_args(parser):
    parser.add_argument('--tsv-file', 
                        type=str, 
                        default='../data/Newsela/newsela-auto/newsela-auto/all_data/aligned-sentence-pairs-all.tsv',
                        help='CRF aligned Newsela tsv file path.')

    parser.add_argument('--output-file', 
                        type=str, 
                        default='../data/Document-level-text-simplification/Dataset/test.csv', 
                        help='Output file path.')

    parser.add_argument('--log-file', 
                        type=str, 
                        default='../data/Document-level-text-simplification/Dataset/test.log', 
                        help='Log file path.')

    parser.add_argument('--cache-file', 
                        type=str, 
                        default='../data/Document-level-text-simplification/Dataset/test.json', 
                        help='Cache file of data dict.')

    parser.add_argument('--server-port', 
                        type=int, 
                        default=8000, 
                        help='OpenIE server port.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Tweet Retrievers.')
    define_args(parser)
    args = parser.parse_args()

    init_logger(verbose=False, log_file=args.log_file)

    extractor = OpenIE5(f'http://localhost:{args.server_port}')

    tsv_file = args.tsv_file

    headers=["sentence1_idx", "sentence1", "sentence2_idx",  'sentence2']

    df= pd.read_csv(tsv_file, sep='\t', on_bad_lines='skip', names=headers)
    df = df.reset_index()

    source_sents = list(df.sentence1)
    target_sents = list(df.sentence2)

    data_dict = get_data_dict(source_sents, target_sents, args.cache_file, extractor)

    data_df = pd.DataFrame.from_dict(data_dict)
    cleaned_df = drop_empty_rows_by_column(data_df, 'source_kg')
    cleaned_df = drop_empty_rows_by_column(cleaned_df, 'target_kg')

    cleaned_df.to_csv(args.output_file, index=False)
    