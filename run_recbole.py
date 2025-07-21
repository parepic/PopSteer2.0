# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse
import torch
import pandas as pd
from recbole.quick_start import run, run_recbole, load_data_and_model
from recbole.utils import (
    get_trainer,
    plot_ndcg_vs_fairness,
    remove_sparse_users_items
)
import csv

from tune import tune

from recbole.data import create_item_popularity_csv


if __name__ == "__main__":
    # remove_sparse_users_items(20, "yelp2018")
    # exit()
    # parameter_dict = {
    # 'train_neg_samplze_args': None,
    # }
    # run_recbole(model='SASRec', dataset='ml-100k', config_dict=parameter_dict)
    # exit()
    # create_item_popularity_csv("ml-1m", 0.2)
    # plot_ndcg_vs_fairness(dataset="ml-1mm")
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument("--train", action="store_true", help="Whether to train model")
    parser.add_argument("--test", action="store_true", help="Whether to test model")
    parser.add_argument("--fair", action="store_true", help="Whether to use FAIR")

    parser.add_argument("--tune", action="store_true", help="Whether to train model")

    parser.add_argument('--config_json', type=str, default=None,
                    help="JSON string with config overrides")

    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )

    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--base_path", type=str, default='no path', help="base model path"
    )
    parser.add_argument(
        "--path", type=str, default='no path', help="model path"
    )

    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    config_dict = dict()
    if args.config_json:
        import json, ast
        # Allow either strict JSON or python-literal (for lists)
        try:
            config_dict = json.loads(args.config_json)
        except json.JSONDecodeError:
            config_dict = ast.literal_eval(args.config_json)
    
    if args.tune == True:
        tune(args)
        exit()
    if args.train == True:
        if args.config_json is None:
            config_dict = {
                "base_path": "./saved/lastfm.pth",
                "sae_scale_size": [16, 16],
                "sae_k": [8, 8],
                "learning_rate": 1e-3,
                "alpha": [1.0, 1.0],
                "steer": [0, 0],
                "analyze": False
            }
        run(
            args.model,
            args.dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            nproc=args.nproc,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            group_offset=args.group_offset,
        )


    elif args.test == True:
        if args.config_json is None:
            config_dict = {
                "alpha": [0.5, 0.5],
                "steer": [1, 1],
                "analyze": True,
                "tail_ratio": 0.2,
                "metrics": ["Recall","MRR","NDCG","Hit","Precision", "Gini", "Deep_LT_Coverage", "GiniIndex", "TailPercentage", "AveragePopularity", "ShannonEntropy"]        
            }
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, dict=config_dict
        )
        if args.fair:
            model.fair = True
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.eval_collector.data_collect(train_data)

        test_result = trainer.evaluate(
            valid_data, model_file=args.path, load_best_model = False, show_progress=config["show_progress"]
        )
        
        keys = [
            'recall@10',
            'mrr@10',
            'ndcg@10',
            'hit@10',
            'deep_lt_coverage@10',
            'giniindex@10',
            'averagepopularity@10',
            'tailpercentage@10',
            'shannonentropy@10'
        ]

        max_key_len = max(len(k) for k in keys)

        # print header
        print(f"{'Metric':<{max_key_len}} | Value")
        print(f"{'-'*max_key_len}-|-------")

        # print each metric with its dynamic value
        for key in keys:
            value = test_result[key]             # get value from your OrderedDict
            print(f"{key:<{max_key_len}} | {value:>7.4f}")


