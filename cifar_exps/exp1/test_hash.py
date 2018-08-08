import sys
sys.path.append('../../configs')
sys.path.append('../../utils')
sys.path.append('../../tfops')

# ../../utils
from datasetmanager import DATASETMANAGER_DICT
from format_op import params2id, listformat
from shutil_op import remove_file, remove_dir, copy_file, copy_dir
from csv_op import CsvWriter2, CsvWriter
from writer import create_muldir, write_pkl

# ./
from local_config import K_SET, RESULT_DIR,\
                         DATASET, GPU_ID, BATCH_SIZE, EPOCH, NSCLASS,\
                         CONV_NAME, LOSS_TYPE, EMBED_M, BUCKET_D,\
                         HASH_LOSS_TYPE, HASH_DISTANCE_TYPE,\
                         PAIRWISE_LAMBDA, HASH_LAMBDA, HASH_MARGIN_ALPHA,\
                         HASH_DECAY_TYPE, HASH_DECAY_PARAM_TYPE

from deepmetric import DeepMetric

import numpy as np
import itertools
import shutil
import glob
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", default = GPU_ID, help="Utilize which gpu", type = int)
parser.add_argument("--dataset", default = DATASET, help="dataset to be used", type = str)
parser.add_argument("--nbatch", default = BATCH_SIZE, help="size of batch", type = int)
parser.add_argument("--nsclass", default = NSCLASS, help="the number of selected class", type = int)
parser.add_argument("--epoch", default = EPOCH, help="epoch to be ran", type = int)
parser.add_argument("--conv", default = CONV_NAME, help="conv network", type = str)
parser.add_argument("--ltype", default = LOSS_TYPE, help="loss type", type = str)
parser.add_argument("--m", default = EMBED_M, help="embedding m", type = int)
parser.add_argument("--d", default = BUCKET_D, help="bucket d", type = int)
parser.add_argument("--hltype", default = HASH_LOSS_TYPE, help="hash loss type", type = str)
parser.add_argument("--hdt", default = HASH_DISTANCE_TYPE, help="hash distance type", type = str)
parser.add_argument("--plamb", default = PAIRWISE_LAMBDA, help="lambda for pairwise cost", type = float)
parser.add_argument("--hlamb", default = HASH_LAMBDA, help="hash lambda", type = float)
parser.add_argument("--hma", default = HASH_MARGIN_ALPHA, help="hash margin alpha", type = float)
parser.add_argument("--hdtype", default = HASH_DECAY_TYPE, help="decay type", type = str)
parser.add_argument("--hdptype", default = HASH_DECAY_PARAM_TYPE, help="hash decay param type", type = str)

args = parser.parse_args()

nactivate = len(ACTIVATE_K_SET)

if __name__ == '__main__':
    args.m=args.d
    args.ltype = args.hltype

    FILE_ID = params2id(args.dataset, args.conv, args.ltype, args.m)

    if args.hltype=='npair': args.nsclass = args.nbatch//2

    HASH_METRIC_PARAM = args.hlamb if args.hltype=='npair' else args.hma
    HASH_FILE_ID = params2id(FILE_ID, args.hltype, args.hdt, args.d, args.k)
    HASH_FILE_ID_TEST = params2id(HASH_FILE_ID, 'test')
    HASH_FILE_ID_TRAIN = params2id(HASH_FILE_ID, 'train')
    QUERY_FILE_ID = params2id(FILE_ID, '*', '*', args.hltype, args.hdt, args.d, args.k, '*', '*', '*', '*', 'test')

    print("file id : {}\nquery id : {}".format(HASH_FILE_ID, QUERY_FILE_ID))

    PKL_DIR = RESULT_DIR+'exp1/pkl/'
    CSV_DIR = RESULT_DIR+'exp1/csv/'
    SAVE_DIR = RESULT_DIR+'exp1/save/%s/'%HASH_FILE_ID

    copy_dst_csv_test = CSV_DIR+HASH_FILE_ID_TEST+'.csv'
    copy_dst_pkl_test = PKL_DIR+HASH_FILE_ID_TEST+'.pkl'
    
    copy_dst_csv_train = CSV_DIR+HASH_FILE_ID_TRAIN+'.csv'
    copy_dst_pkl_train = PKL_DIR+HASH_FILE_ID_TRAIN+'.pkl'

    if os.path.exists(SAVE_DIR): remove_dir(SAVE_DIR)
    if os.path.exists(copy_dst_csv_train): remove_file(copy_dst_csv_train)
    if os.path.exists(copy_dst_csv_test): remove_file(copy_dst_csv_test)

    if os.path.exists(copy_dst_pkl_train): remove_file(copy_dst_pkl_train)
    if os.path.exists(copy_dst_pkl_test): remove_file(copy_dst_pkl_test)

    pkl_files = glob.glob(PKL_DIR+QUERY_FILE_ID+'.pkl')
    print(pkl_files)
    if len(pkl_files)==0:
        print("No such pkl files")
        sys.exit() 

    best_file_id = os.path.basename(pkl_files[0])[:-9] # -_test.pkl'
    best_performance = np.sum(read_pkl(pkl_files[0])['te_te_precision_at_k'])
    for pkl_idx in range(len(pkl_files)):
        file_id = os.path.basename(pkl_files[pkl_idx])[:-9] # -_test.pkl'
        performance = np.sum(read_pkl(pkl_files[pkl_idx])['te_te_precision_at_k'])
        print("performance : {} from {}".format(performance, file_id))
        if performance > best_performance:
            best_performance, best_file_id = performance, file_id
    print("best performance : {} from {}".format(best_performance, best_file_id))

    copy_file(CSV_DIR+best_file_id+'_test.csv', copy_dst_csv_test)
    copy_file(CSV_DIR+best_file_id+'_train.csv', copy_dst_csv_train)
    copy_file(PKL_DIR+best_file_id+'_test.pkl', copy_dst_pkl_test)
    copy_file(PKL_DIR+best_file_id+'_train.pkl', copy_dst_pkl_train)
    copy_dir(RESULT_DIR+'exp1/save/'+best_file_id, SAVE_DIR)

