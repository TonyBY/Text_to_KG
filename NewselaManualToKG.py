import json
import nltk
nltk.download('punkt')
import os

from pyopenie import OpenIE5

import pandas as pd
from tqdm import tqdm
from utils import read_json, save_json, logger, init_logger, drop_empty_rows_by_column

import argparse

def define_args(parser):
    parser.add_argument('--tsv-file', 
                        type=str, 
                        default='../data/Newsela/newsela-auto/newsela-manual/crowdsourced/dev.tsv', 
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

def get_best_triplet_of_sentence(sent, extractor):
    try:
        extractions = extractor.extract(sent)
    except Exception as e:
        logger.error(f"Exception: {e}")
        logger.error(f"sent: {sent}")
        return [], True
    max_conf = 0
    best_idx = -1
    for i in range(len(extractions)):
        conf = float(extractions[i]['confidence'])
        if conf > max_conf:
            max_conf = conf
            best_idx = i
    try:
        trip = [extractions[best_idx]['extraction']['arg1']['text'], extractions[best_idx]['extraction']['rel']['text'], extractions[best_idx]['extraction']['arg2s'][0]['text']]
        return list(trip), False
    except Exception as e:
        logger.error(f"Exception: {e}")
        logger.error(f"sent: {sent}")
        logger.error(f"extractions: {extractions}")
        return [], True

def get_data_dict(src_lines, tgt_lines, cache_file, extractor):

    if os.path.exists(cache_file):
        data_dict = read_json(cache_file)
        logger.info(f"Loading {len(data_dict['source_kg'])} examples from cached data_dict.")
    else:
        data_dict = {'source_kg': [], 'source_doc': [], 'target_doc': [], 'target_kg': []}

    assert(len(src_lines) == len(tgt_lines))
    
    for i in tqdm(range(len(data_dict['source_kg']), len(src_lines), 1)):
    # for i in tqdm(range(len(data_dict['source_kg']), 6, 1)):
        FAILED = False
        logger.debug(f"\n============== i: {i}=========================")
        logger.debug(f"{i}: Source: {src_lines[i]}\n   Target: {tgt_lines[i]}")
        src_doc_sents = nltk.sent_tokenize(src_lines[i])
        src_doc_trips = []
        for sent in src_doc_sents:
            trip, FAILED = get_best_triplet_of_sentence(sent, extractor)
            if FAILED:
                break
            src_doc_trips.append(trip)
            logger.debug(f"   {trip}")
        
        if FAILED:
            data_dict['source_kg'].append([])
            data_dict['target_kg'].append([])
            data_dict['source_doc'].append([])
            data_dict['target_doc'].append([])
            continue
        
        try:
            tgt_doc_sents = nltk.sent_tokenize(tgt_lines[i])
        except:
            FAILED = True
            logger.info(f"tgt_lines[i]: {tgt_lines[i]}")
        tgt_doc_trips = []
        for sent in tgt_doc_sents:
            trip, FAILED = get_best_triplet_of_sentence(sent, extractor)
            if FAILED:
                break
            tgt_doc_trips.append(trip)
            logger.debug(f"   {trip}")
        if FAILED:
            data_dict['source_kg'].append([])
            data_dict['target_kg'].append([])
            data_dict['source_doc'].append([])
            data_dict['target_doc'].append([])
            continue
        data_dict['source_kg'].append(json.dumps(src_doc_trips))
        data_dict['target_kg'].append(json.dumps(tgt_doc_trips))
        data_dict['source_doc'].append(src_lines[i])
        data_dict['target_doc'].append(tgt_lines[i])

        if i % 10 == 0 or i == len(src_lines) - 1:
            save_json(data_dict, cache_file)
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Tweet Retrievers.')
    define_args(parser)
    args = parser.parse_args()

    init_logger(verbose=False, log_file=args.log_file)

    extractor = OpenIE5(f'http://localhost:{args.server_port}')

    tsv_file = args.tsv_file

    headers=["align_type", "sentence1_idx", "sentence2_idx", "sentence1", 'sentence2']

    df= pd.read_csv(tsv_file, sep='\t', on_bad_lines='skip', names=headers)
    df = df.iloc[df.index[df.align_type=='aligned']]
    df = df.reset_index()

    source_sents = list(df.sentence1)
    target_sents = list(df.sentence2)

    assert len(source_sents) == len(target_sents)

    data_dict = get_data_dict(source_sents, target_sents, args.cache_file, extractor)

    data_df = pd.DataFrame.from_dict(data_dict)
    cleaned_df = drop_empty_rows_by_column(data_df, 'source_kg')
    cleaned_df = drop_empty_rows_by_column(cleaned_df, 'target_kg')

    cleaned_df.to_csv(args.output_file, index=False)