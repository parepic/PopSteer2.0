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
    get_trainer
)

from recbole.data import create_item_popularity_csv

def tune(args):
    if args.config_json is None:
        config_dict = {
            "alpha": [0, 0],
            "steer": [0, 1],
            "analyze": True,
            "tail_ratio": 0.2,
            "metrics": ["Recall","MRR","NDCG","Hit","Precision","SAE_Loss_i", "SAE_Loss_u", "SAE_Loss_total", "Gini", "Deep_LT_Coverage", "GiniIndex", "TailPercentage", "AveragePopularity", "ShannonEntropy"]        
            }
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, dict=config_dict
    )
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.data_collect(train_data)

    change1 = [0.0, -0.5, -1, -1.5, -2.0, -2.5, -3.0]

    metric_keys = [
        'mrr@10',
        'ndcg@10',
        'hit@10',
        'deep_lt_coverage@10',
        'giniindex@10',
        'averagepopularity@10',
        'shannonentropy@10'
    ]

    SHORT_NAMES = {
        'mrr@10': 'MRR@10',
        'ndcg@10': 'NDCG@10',
        'hit@10': 'HIT@10',
        'deep_lt_coverage@10': 'DLTC@10',
        'giniindex@10': 'GINI@10',
        'averagepopularity@10': 'AVGPOP@10',
        'shannonentropy@10': 'SHANNON@10'
    }

    # Collect raw metric values
    rows_raw = []
    for c1 in change1:
        trainer.model.sae_module_u.alpha = c1
        test_result = trainer.evaluate(
            valid_data,
            model_file=args.path,
            load_best_model=False,
            show_progress=config["show_progress"]
        )
        trainer.model.restore_item_e = None
        rows_raw.append({
            'alpha': c1,
            **{k: test_result[k] for k in metric_keys}
        })

    baseline = rows_raw[0]

    value_decimals = 4
    pct_decimals = 2
    show_zero_pct_on_baseline = False  # set True if you want (+0.00%)

    # First build formatted string cells (without widths yet)
    header_labels = ['alpha'] + [SHORT_NAMES[k] for k in metric_keys]

    formatted_rows = []
    for i, r in enumerate(rows_raw):
        is_baseline = (i == 0)
        formatted_row = {}
        formatted_row['alpha'] = f"{r['alpha']:.2f}"

        for k in metric_keys:
            val = r[k]
            base = baseline[k]
            if is_baseline:
                if show_zero_pct_on_baseline and base != 0:
                    formatted_row[SHORT_NAMES[k]] = f"{val:.{value_decimals}f} (+0.00%)"
                else:
                    formatted_row[SHORT_NAMES[k]] = f"{val:.{value_decimals}f}"
            else:
                if base == 0:
                    formatted_row[SHORT_NAMES[k]] = f"{val:.{value_decimals}f} (n/a)"
                else:
                    pct = (val - base) / base * 100.0
                    sign = '+' if pct >= 0 else ''
                    formatted_row[SHORT_NAMES[k]] = f"{val:.{value_decimals}f} ({sign}{pct:.{pct_decimals}f}%)"
        formatted_rows.append(formatted_row)

    # Compute column widths: max of header and all cell contents
    col_width = {}
    for h in header_labels:
        max_cell = max(len(row[h]) for row in formatted_rows)
        col_width[h] = max(len(h), max_cell)

    # Build and print header
    header_line = " | ".join(f"{h:<{col_width[h]}}" for h in header_labels)
    sep_line = "-+-".join("-" * col_width[h] for h in header_labels)
    print(header_line)
    print(sep_line)

    # Print each row
    for fr in formatted_rows:
        line = " | ".join(f"{fr[h]:<{col_width[h]}}" for h in header_labels)
        print(line)

    # Return raw plus formatted if needed downstream
    return rows_raw, formatted_rows



def remove_sparse_users_items(n, dataset):
    # --- Step 1: Load the Data ---
    # The files use tab as the delimiter and have headers that include type annotations.
    items = pd.read_csv(rf"./dataset/{dataset}/{dataset}.item", sep="\t", header=0)
    interactions = pd.read_csv(rf"./dataset/{dataset}/{dataset}.inter", sep="\t", header=0)
    # --- Step 2: Iterative Filtering ---
    # We use a threshold of at least 5 interactions for both users and items.
    iteration = 0
    while True:
        iteration += 1
        current_shape = interactions.shape[0]
        
        # Remove users with fewer than 5 interactions:
        user_counts = interactions["user_id:token"].value_counts()
        valid_users = user_counts[user_counts >= n].index
        interactions = interactions[interactions["user_id:token"].isin(valid_users)]
                
        # Remove items with fewer than 5 interactions:
        item_counts = interactions["item_id:token"].value_counts()
        valid_items = item_counts[item_counts >= n].index
        interactions = interactions[interactions["item_id:token"].isin(valid_items)]
        
        new_shape = interactions.shape[0]
        print(f"Iteration {iteration}: {current_shape} -> {new_shape} interactions remain")
        
        if new_shape == current_shape:
            break
    # --- Step 3: Synchronize Items With Interactions ---
    # Keep only those items that still appear in the filtered interactions.
    items = items[items["item_id:token"].isin(interactions["item_id:token"])]

    # --- Step 4: Save the Filtered Files ---
    # Files are saved with the header intact (including the type annotations).
    items.to_csv(rf"./dataset/{dataset}/{dataset}.item.filtered", sep="\t", index=False, header=True)
    interactions.to_csv(rf"./dataset/{dataset}/{dataset}.inter.filtered", sep="\t", index=False, header=True)

    print(f"Filtering complete. Files saved as '{dataset}.item.filtered', '{dataset}.inter.filtered'.")



if __name__ == "__main__":
    # remove_sparse_users_items(5, "lastfm")
    # exit()
    # parameter_dict = {
    # 'train_neg_sample_args': None,
    # }
    # run_recbole(model='SASRec', dataset='ml-100k', config_dict=parameter_dict)
    # exit()
    # create_item_popularity_csv("ml-1m", 0.2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument("--train", action="store_true", help="Whether to train model")
    parser.add_argument("--test", action="store_true", help="Whether to test model")
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

    if hasattr(args, "train") and args.train == True:
        if args.config_json is None:
            config_dict = {
                "base_path": "./saved/zorduda4.pth",
                "sae_scale_size": [32, 32],
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


    elif hasattr(args, "test") and args.test == True:
        if args.config_json is None:
            config_dict = {
                "alpha": [0.5, 1.0],
                "steer": [1, 1],
                "analyze": True,
                "N": [4096, 8192],
                "tail_ratio": 0.2,
                "metrics": ["Recall","MRR","NDCG","Hit","Precision", "Gini", "Deep_LT_Coverage", "GiniIndex", "TailPercentage", "AveragePopularity", "ShannonEntropy"]        
            }

        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, dict=config_dict
        )

        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.eval_collector.data_collect(train_data)

        test_result = trainer.evaluate(
            test_data, model_file=args.path, load_best_model = False, show_progress=config["show_progress"]
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


