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
    # remove_sparse_users_items(5, "ml-1m")
    # exit()
    # parameter_dict = {
    # 'train_neg_sample_args': None,
    # }
    # run_recbole(model='SASRec', dataset='ml-100k', config_dict=parameter_dict)
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument("--train", action="store_true", help="Whether to train model")
    parser.add_argument("--test", action="store_true", help="Whether to test model")

    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument('--sae_k', '-k', type=int, default=32,
                        help="Sparsity parameter K: keep only the top‑k activations per input in the SAE (Eq. 1).")
    parser.add_argument('--lr', '-lr', type=float, default=1e-4,
                        help="learning rate")

    parser.add_argument('--scale', '--scale_size', type=int, default=8,
                        dest='scale',
                        help="Scale factor s controlling the SAE hidden size relative to the input (s × d).")
    parser.add_argument('--N', '-n', type=int, default=0,
                        help="Number of neurons to steer")
    parser.add_argument('--alpha', '-a', type=int, default=0,
                        help="Alpha")

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
    if hasattr(args, "train") and args.train == True:
        if hasattr(args, "base_path") and args.base_path != "no path":
            config_dict["base_path"] = args.base_path
        if hasattr(args, "scale_size"):
            config_dict["sae_scale_size"] = args.base_path
        if hasattr(args, "sae_k"):
            config_dict["sae_k"] = args.sae_k
        if hasattr(args, "lr"):
            config_dict["learning_rate"] = args.lr

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

    
    elif hasattr(args, "analyze") and args.analyze == True:
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, device=device
        )  
        
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.analyze_neurons(train_data, eval_data=False, sae=False)

        save_mean_SD(config["dataset"], popular=True)
        save_mean_SD(config["dataset"], popular=False)
        save_cohens_d(config["dataset"])


    elif hasattr(args, "test") and args.test == True:
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path
        )  
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

        test_result = trainer.evaluate(
            valid_data, model_file=args.path, load_best_model = False, show_progress=config["show_progress"]
        )
        
        keys = [
            'recall@10',
            'mrr@10',
            'ndcg@10',
            'hit@10',
            'deep_lt_coverage@10',
            'gini@10'
        ]

        max_key_len = max(len(k) for k in keys)

        # print header
        print(f"{'Metric':<{max_key_len}} | Value")
        print(f"{'-'*max_key_len}-|-------")

        # print each metric with its dynamic value
        for key in keys:
            value = test_result[key]             # get value from your OrderedDict
            print(f"{key:<{max_key_len}} | {value:>7.4f}")