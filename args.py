"""
Command-line argument parsing.
"""

import argparse
#from functools import partial

import time
import tensorflow as tf
import json
import os

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    file_time = int(time.time())
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--arch',           help='name architecture',   default="fcn",   type=str)
    parser.add_argument('--seed',           help='random seed', default=0, type=int)
    parser.add_argument('--name',           help='name add-on',      type=str,  default='Model_config-'+str(file_time))
    parser.add_argument('--dataset',        help='data set to evaluate on',      type=str,  default='Omniglot')
    parser.add_argument('--data_path',      help='path to data folder',      type=str,  default='/home/')
    parser.add_argument('--config',         help='json config file', type=str,  default=None)
    parser.add_argument('--checkpoint',     help='checkpoint directory', default='model_checkpoint')
    parser.add_argument('--test',           help='Testing or Not', action='store_true')
    parser.add_argument('--testintrain',    help='Testing during train or Not', action='store_true')
    parser.add_argument('--min_classes',    help='minimum number of classes for n-way', default=2, type=int)
    parser.add_argument('--max_classes',    help='maximum (excluded) number of classes for n-way', default=2, type=int)
    parser.add_argument('--ttrain_shots',   help='number of examples per class in meta train', default=5, type=int)
    parser.add_argument('--ttest_shots',    help='number of examples per class in meta test',  default=15, type=int)
    parser.add_argument('--etrain_shots',   help='number of examples per class in meta train', default=5, type=int)
    parser.add_argument('--etest_shots',    help='number of examples per class in meta test',  default=15, type=int)
    parser.add_argument('--train_inner_K',  help='number of inner gradient steps during meta training', default=5, type=int)
    parser.add_argument('--test_inner_K',   help='number of inner gradient steps during meta testing', default=5, type=int)
    parser.add_argument('--learning_rate',  help='Adam step size for inner training', default=0.4, type=float)
    parser.add_argument('--meta_step',      help='meta-training step size', default=0.01, type=float)
    parser.add_argument('--meta_batch',     help='meta-training batch size', default=1, type=int)
    parser.add_argument('--meta_iters',     help='meta-training iterations', default=70001, type=int)
    parser.add_argument('--eval_iters',     help='meta-training iterations', default=2000, type=int)
    parser.add_argument('--step',           help='Checkpoint step to load', default=59999,     type=float)
    # python main_emb.py --meta_step 0.005 --meta_batch 8 --learning_rate 0.3 --test --checkpoint Model_config-1568818723

    args = vars(parser.parse_args())
    #os.system("mkdir -p " + args['checkpoint'])
    if args['config'] is None:
        args['config'] = f"{args['checkpoint']}/{args['name']}/{args['name']}.json"
        print(args['config'])
        # os.system("mkdir -p " + f"{args['checkpoint']}")
        os.system("mkdir -p " + f"{args['checkpoint']}/{args['name']}")
        with open(args['config'], 'w') as write_file:
            print("Json Dumping...")
            json.dump(args, write_file)
    else:
        with open(args['config'], 'r') as open_file:
            args = json.load(open_file)
    return parser

def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'min_classes':          parsed_args.min_classes,
        'max_classes':          parsed_args.max_classes,
        'train_shots':          parsed_args.ttrain_shots,
        'test_shots':           parsed_args.ttest_shots,
        'meta_batch':           parsed_args.meta_batch,
        'meta_iters':           parsed_args.meta_iters,
        'test_iters':           parsed_args.eval_iters,
        'train_step'  :         parsed_args.step,
        'name':	                parsed_args.name,

    }

def test_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'eval_step'  :          parsed_args.step,
        'min_classes':          parsed_args.min_classes,
        'max_classes':          parsed_args.max_classes,
        'train_shots':          parsed_args.etrain_shots,
        'test_shots':           parsed_args.etest_shots,
        'meta_batch':           parsed_args.meta_batch,
        'meta_iters':           parsed_args.eval_iters,
        'name':                 parsed_args.name,

    }