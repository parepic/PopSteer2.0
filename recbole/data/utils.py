# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/9, 2020/9/17, 2020/8/31, 2021/2/20, 2021/3/1, 2022/7/6
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li, Haoran Cheng, Jiawei Guan, Gaowei Zhang
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com, chenghaoran29@foxmail.com, guanjw@ruc.edu.cn, zgw15630559577@163.com

"""
recbole.data.utils
########################
"""

import copy
import importlib
import os
import pickle
import warnings
from typing import Literal
import pandas as pd
from recbole.data.dataloader import *
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, ensure_dir, get_local_time, set_color
from recbole.utils.argument_list import dataset_arguments


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module("recbole.data.dataset")
    if hasattr(dataset_module, config["model"] + "Dataset"):
        dataset_class = getattr(dataset_module, config["model"] + "Dataset")
    else:
        model_type = config["MODEL_TYPE"]
        type2class = {
            ModelType.GENERAL: "Dataset",
            ModelType.SEQUENTIAL: "SequentialDataset",
            ModelType.CONTEXT: "Dataset",
            ModelType.KNOWLEDGE: "KnowledgeBasedDataset",
            ModelType.TRADITIONAL: "Dataset",
            ModelType.DECISIONTREE: "Dataset",
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(
        config["checkpoint_dir"], f'{config["dataset"]}-{dataset_class.__name__}.pth'
    )
    file = config["dataset_save_path"] or default_file
    if os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ["seed", "repeatable"]:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color("Load filtered dataset from", "pink") + f": [{file}]")
            return dataset

    dataset = dataset_class(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset


def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config["checkpoint_dir"])
    save_path = config["checkpoint_dir"]
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color("Saving split dataloaders into", "pink") + f": [{file_path}]")
    Serialization_dataloaders = []
    for dataloader in dataloaders:
        generator_state = dataloader.generator.get_state()
        dataloader.generator = None
        dataloader.sampler.generator = None
        Serialization_dataloaders += [(dataloader, generator_state)]

    with open(file_path, "wb") as f:
        pickle.dump(Serialization_dataloaders, f)


def load_split_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """

    default_file = os.path.join(
        config["checkpoint_dir"],
        f'{config["dataset"]}-for-{config["model"]}-dataloader.pth',
    )
    dataloaders_save_path = config["dataloaders_save_path"] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, "rb") as f:
        dataloaders = []
        for data_loader, generator_state in pickle.load(f):
            generator = torch.Generator()
            generator.set_state(generator_state)
            data_loader.generator = generator
            data_loader.sampler.generator = generator
            dataloaders.append(data_loader)

        train_data, valid_data, test_data = dataloaders
    for arg in dataset_arguments + ["seed", "repeatable", "eval_args"]:
        if config[arg] != train_data.config[arg]:
            return None
    train_data.update_config(config)
    valid_data.update_config(config)
    test_data.update_config(config)
    logger = getLogger()
    logger.info(
        set_color("Load split dataloaders from", "pink")
        + f": [{dataloaders_save_path}]"
    )
    return train_data, valid_data, test_data


def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['[valid|test]_eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
        dataset._change_feat_format()
    else:
        model_type = config["MODEL_TYPE"]
        built_datasets = dataset.build()

        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(
            config, dataset, built_datasets
        )

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, shuffle=config["shuffle"]
            )
        else:
            kg_sampler = KGSampler(
                dataset,
                config["train_neg_sample_args"]["distribution"],
                config["train_neg_sample_args"]["alpha"],
            )
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, kg_sampler, shuffle=True
            )

        valid_data = get_dataloader(config, "valid")(
            config, valid_dataset, valid_sampler, shuffle=False
        )
        test_data = get_dataloader(config, "test")(
            config, test_dataset, test_sampler, shuffle=False
        )
        if config["save_dataloaders"]:
            save_split_dataloaders(
                config, dataloaders=(train_data, valid_data, test_data)
            )

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_data, valid_data, test_data


def get_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.
    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError(
            "`phase` can only be 'train', 'valid', 'test' or 'evaluation'."
        )
    if phase == "evaluation":
        phase = "test"
        warnings.warn(
            "'evaluation' has been deprecated, please use 'valid' or 'test' instead.",
            DeprecationWarning,
        )

    register_table = {
        "MultiDAE": _get_AE_dataloader,
        "MultiVAE": _get_AE_dataloader,
        "MacridVAE": _get_AE_dataloader,
        "CDAE": _get_AE_dataloader,
        "ENMF": _get_AE_dataloader,
        "RaCT": _get_AE_dataloader,
        "RecVAE": _get_AE_dataloader,
        "DiffRec": _get_AE_dataloader,
        "LDiffRec": _get_AE_dataloader,
    }

    if config["model"] in register_table:
        return register_table[config["model"]](config, phase)

    model_type = config["MODEL_TYPE"]
    if phase == "train":
        if model_type != ModelType.KNOWLEDGE:
            return TrainDataLoader
        else:
            return KnowledgeBasedDataLoader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _get_AE_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Customized function for VAE models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError(
            "`phase` can only be 'train', 'valid', 'test' or 'evaluation'."
        )
    if phase == "evaluation":
        phase = "test"
        warnings.warn(
            "'evaluation' has been deprecated, please use 'valid' or 'test' instead.",
            DeprecationWarning,
        )

    if phase == "train":
        return UserDataLoader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _create_sampler(
    dataset,
    built_datasets,
    distribution: str,
    repeatable: bool,
    alpha: float = 1.0,
    base_sampler=None,
):
    phases = ["train", "valid", "test"]
    sampler = None
    if distribution != "none":
        if base_sampler is not None:
            base_sampler.set_distribution(distribution)
            return base_sampler
        if not repeatable:
            sampler = Sampler(
                phases,
                built_datasets,
                distribution,
                alpha,
            )
        else:
            sampler = RepeatableSampler(
                phases,
                dataset,
                distribution,
                alpha,
            )
    return sampler


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    train_neg_sample_args = config["train_neg_sample_args"]
    valid_neg_sample_args = config["valid_neg_sample_args"]
    test_neg_sample_args = config["test_neg_sample_args"]
    repeatable = config["repeatable"]
    base_sampler = _create_sampler(
        dataset,
        built_datasets,
        train_neg_sample_args["distribution"],
        repeatable,
        train_neg_sample_args["alpha"],
    )
    train_sampler = base_sampler.set_phase("train") if base_sampler else None

    valid_sampler = _create_sampler(
        dataset,
        built_datasets,
        valid_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    valid_sampler = valid_sampler.set_phase("valid") if valid_sampler else None

    test_sampler = _create_sampler(
        dataset,
        built_datasets,
        test_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    test_sampler = test_sampler.set_phase("test") if test_sampler else None
    return train_sampler, valid_sampler, test_sampler




def create_item_popularity_csv(dataset: str, p_top: float, p_bottom: float):
    """
    Create a CSV assigning popularity labels to items with three classes:
        +1 : item lies within the top cumulative fraction p_top (most popular)
        -1 : item lies within the bottom cumulative fraction p_bottom (least popular)
         0 : all other items
         
    Args:
        dataset   : name of the dataset directory under ./dataset/{dataset}
        p_top     : threshold in (0,1] for cumulative fraction from the head (descending sort)
        p_bottom  : threshold in (0,1] for cumulative fraction from the tail (ascending sort)
    """
    assert 0 < p_top <= 1, "p_top must be in (0, 1]"
    assert 0 < p_bottom <= 1, "p_bottom must be in (0, 1]"
    
    # -------------------------------
    # Step 1: Load training interactions and compute frequencies.
    # -------------------------------
    dataset_path = os.path.join(".", "dataset", dataset)
    train_npz_path = os.path.join(dataset_path, "biased_eval_train.npz")
    data = np.load(train_npz_path)
    item_ids = data["item_id"]          # array of item IDs
    total_interactions = len(item_ids)

    unique_items, counts = np.unique(item_ids, return_counts=True)
    pop_scores = counts / total_interactions

    df = pd.DataFrame({
        "item_id:token": unique_items,
        "interaction_count": counts,
        "pop_score": pop_scores
    })

    # -------------------------------
    # Step 2: Identify top popular items.
    # -------------------------------
    df_top = df.sort_values(by="interaction_count", ascending=False).reset_index(drop=True)
    total_sum = df_top["interaction_count"].sum()
    df_top["cum_interaction"] = df_top["interaction_count"].cumsum()
    df_top["cum_frac"] = df_top["cum_interaction"] / total_sum
    df_top["label_top"] = (df_top["cum_frac"] <= p_top).astype(int)
    top_labels = df_top.set_index("item_id:token")["label_top"].to_dict()

    # -------------------------------
    # Step 3: Identify bottom (least popular) items.
    # -------------------------------
    df_bottom = df.sort_values(by="interaction_count", ascending=True).reset_index(drop=True)
    df_bottom["cum_interaction"] = df_bottom["interaction_count"].cumsum()
    df_bottom["cum_frac"] = df_bottom["cum_interaction"] / total_sum
    df_bottom["label_bottom"] = (df_bottom["cum_frac"] <= p_bottom).astype(int)
    bottom_labels = df_bottom.set_index("item_id:token")["label_bottom"].to_dict()

    # -------------------------------
    # Step 4: Assign final labels.
    # -------------------------------
    def assign_label(item_id):
        if top_labels.get(item_id, 0) == 1:
            return 1
        if bottom_labels.get(item_id, 0) == 1:
            return -1
        return 0

    df["popularity_label"] = df["item_id:token"].apply(assign_label)

    # (Optional) For transparency, you can record whether an item was in either set
    df = df.sort_values(by="interaction_count", ascending=False).reset_index(drop=True)

    output_csv = os.path.join(dataset_path, "item_popularity_labels.csv")
    df.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' created successfully.")



# def create_item_popularity_csv(dataset: str, p: float):
#     """
#     Create a CSV assigning popularity labels based on item rank (not interaction mass).

#     Args:
#         dataset : dataset directory name under ./dataset/{dataset}
#         p       : fraction (0 < p < 1). Example: 0.1 -> top 10% items get +1, bottom 10% get -1.
#                   Usually choose p <= 0.5 to avoid overlap.

#     Output:
#         Writes ./dataset/{dataset}/item_popularity_labels.csv with columns:
#             item_id:token, interaction_count, pop_score, popularity_label
#     """
#     if not (0 < p < 1):
#         raise ValueError("p must be in (0,1)")
#     if p > 0.5:
#         print("Warning: p > 0.5 causes overlap of top and bottom sets. Proceeding but resolving conflicts toward +1.")

#     dataset_path = os.path.join(".", "dataset", dataset)
#     train_npz_path = os.path.join(dataset_path, "biased_eval_train.npz")
#     data = np.load(train_npz_path)
#     item_ids = data["item_id"]
#     total_interactions = len(item_ids)

#     unique_items, counts = np.unique(item_ids, return_counts=True)
#     pop_scores = counts / total_interactions

#     df = pd.DataFrame({
#         "item_id:token": unique_items,
#         "interaction_count": counts,
#         "pop_score": pop_scores
#     })

#     n_items = len(df)
#     k = max(1, int(round(p * n_items)))   # number of items in each extreme set

#     # Sort descending for top set
#     df_sorted_desc = df.sort_values("interaction_count", ascending=False).reset_index(drop=True)
#     # Sort ascending for bottom set
#     df_sorted_asc = df.sort_values("interaction_count", ascending=True).reset_index(drop=True)

#     # Determine cut indices
#     # Using rank positions: first k items in desc list -> top, first k items in asc list -> bottom
#     top_item_ids = set(df_sorted_desc.head(k)["item_id:token"])
#     bottom_item_ids = set(df_sorted_asc.head(k)["item_id:token"])

#     # If overlap occurs (possible if p > 0.5 or ties + small n) resolve: priority to top (+1)
#     overlap = top_item_ids & bottom_item_ids
#     if overlap:
#         bottom_item_ids -= overlap

#     def assign_label(item_id):
#         if item_id in top_item_ids:
#             return 1
#         if item_id in bottom_item_ids:
#             return -1
#         return 0

#     df["popularity_label"] = df["item_id:token"].apply(assign_label)

#     # Optional: sort by interaction_count descending for readability
#     df_out = df.sort_values("interaction_count", ascending=False).reset_index(drop=True)

#     out_path = os.path.join(dataset_path, "item_popularity_labels.csv")
#     df_out.to_csv(out_path, index=False)
#     print(f"Written {out_path}")
#     print(f"Top set size: {len(top_item_ids)}  Bottom set size: {len(bottom_item_ids)}  Total items: {n_items}")


import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from typing import Optional



from pathlib import Path
import numpy as np
import pandas as pd

def create_user_popularity_csv(
    dataset: str,
    user_frac: float = 0.8,
) -> None:
    """
    Create <dataset>/user_popularity_labels.csv by computing per-user fractions
    of interactions with popular/unpopular items (based on item popularity_label)
    and labeling based on a fraction threshold.

    Source files expected in "./dataset/<dataset>/":
        - biased_eval_train.npz    (arrays "item_id", "user_id" of equal length)
        - item_popularity_labels.csv  (columns: item_id:token, popularity_label, ...)

    Steps
    -----
    1. Load all interactions (user_id, item_id) from the NPZ.
    2. Load item labels and identify head/tail item sets:
       - Head items: items with popularity_label == 1.
       - Tail items: items with popularity_label == -1.
    3. For each user, compute:
       - fraction_pop: proportion of interacted items in head set.
       - fraction_unpop: proportion of interacted items in tail set.
    4. Label users:
          * popular: fraction_pop >= user_frac    → label  1
          * unpopular: fraction_unpop >= user_frac → label -1
          * everyone else → label 0
    5. Users with no interactions are labeled 0 with fraction_pop=0.
    6. Write CSV sorted by descending fraction_pop (as "pop_score").

    Parameters
    ----------
    dataset : str
        Dataset subdirectory name under "./dataset".
    user_frac : float, default 0.8
        Fraction threshold for labeling a user as popular/unpopular.

    Output
    ------
    Writes "./dataset/<dataset>/user_popularity_labels.csv".
    """
    # Paths
    dataset_path = Path(".", "dataset", dataset)
    train_npz_path = dataset_path / "biased_eval_train.npz"
    item_pop_csv = dataset_path / "item_popularity_labels.csv"
    out_csv = dataset_path / "user_popularity_labels.csv"

    # Load interactions
    data = np.load(train_npz_path)
    item_ids = data["item_id"]
    user_ids = data["user_id"]

    # Build interaction DataFrame
    df_inter = pd.DataFrame({"user_id": user_ids, "item_id:token": item_ids})

    # Load item labels
    df_items = pd.read_csv(item_pop_csv, usecols=["item_id:token", "popularity_label"])

    # Drop NaNs for computation
    df_items = df_items.dropna(subset=["popularity_label"])

    if df_items.empty:
        # Degenerate case: no labeled items
        head_items = set()
        tail_items = set()
    else:
        # Head items: popularity_label == 1
        head_df = df_items[df_items["popularity_label"] == 1]
        head_items = set(head_df["item_id:token"])

        # Tail items: popularity_label == -1
        tail_df = df_items[df_items["popularity_label"] == -1]
        tail_items = set(tail_df["item_id:token"])

    # Group interactions by user to get set of unique interacted items
    user_groups = df_inter.groupby("user_id")["item_id:token"].apply(set).reset_index()

    # Compute fractions
    def calc_fractions(items_set):
        if not items_set:
            return 0.0, 0.0
        num_total = len(items_set)
        num_pop = len(items_set & head_items)
        num_unpop = len(items_set & tail_items)
        return num_pop / num_total, num_unpop / num_total

    fractions = user_groups["item_id:token"].apply(calc_fractions)
    user_groups["fraction_pop"] = fractions.apply(lambda x: x[0])
    user_groups["fraction_unpop"] = fractions.apply(lambda x: x[1])

    # Labels
    def get_label(row):
        if row["fraction_pop"] >= user_frac:
            return 1
        elif row["fraction_unpop"] >= user_frac:
            return -1
        else:
            return 0

    user_groups["popularity_label"] = user_groups.apply(get_label, axis=1)

    # Output DF
    out_df = user_groups[["user_id", "fraction_pop", "popularity_label"]].rename(
        columns={"user_id": "user_id:token", "fraction_pop": "pop_score"}
    ).sort_values("pop_score", ascending=False)

    out_df.to_csv(out_csv, index=False)

    
# def create_item_popularity_csv(dataset: str, p: float):
#     # -------------------------------
#     # Step 1: Load the training NPZ file and compute item frequencies.
#     # -------------------------------
#     dataset_path = os.path.join(".", "dataset", dataset)
#     train_npz_path = os.path.join(dataset_path, "biased_eval_train.npz")
#     data = np.load(train_npz_path)
#     labels = data["labels"]  # assuming this array contains item IDs (item_id:token)
#     total_interactions = len(labels)

#     # Compute frequency counts for each unique item.
#     unique_items, counts = np.unique(labels, return_counts=True)
#     pop_scores = counts / total_interactions

#     df_counts = pd.DataFrame({
#         "item_id:token": unique_items,
#         "interaction_count": counts,
#         "pop_score": pop_scores
#     })

#     # -------------------------------
#     # Step 2: Load the items_remapped CSV file.
#     # -------------------------------
#     items_csv_path = os.path.join(dataset_path, "items_remapped.csv")
#     df_titles = pd.read_csv(items_csv_path)

#     # -------------------------------
#     # Step 3: Merge with interaction counts.
#     # -------------------------------
#     df_merged = pd.merge(df_titles, df_counts, on="item_id:token", how="left")
#     df_merged["interaction_count"] = df_merged["interaction_count"].fillna(0).astype(int)
#     df_merged["pop_score"] = df_merged["pop_score"].fillna(0)

#     # -------------------------------
#     # Step 4: Compute popularity_label
#     # -------------------------------
#     df_top = df_merged.sort_values(by="interaction_count", ascending=False).reset_index(drop=True)
#     total_sum = df_top["interaction_count"].sum()
#     df_top["cum_interaction"] = df_top["interaction_count"].cumsum()
#     df_top["cum_frac"] = df_top["cum_interaction"] / total_sum
#     df_top["popularity_label_top"] = (df_top["cum_frac"] <= p).astype(int)

#     df_bottom = df_merged.sort_values(by="interaction_count", ascending=True).reset_index(drop=True)
#     df_bottom["cum_interaction"] = df_bottom["interaction_count"].cumsum()
#     df_bottom["cum_frac"] = df_bottom["cum_interaction"] / total_sum
#     df_bottom["popularity_label_bottom"] = (df_bottom["cum_frac"] <= p).astype(int)

#     top_labels = df_top.set_index("item_id:token")["popularity_label_top"].to_dict()
#     bottom_labels = df_bottom.set_index("item_id:token")["popularity_label_bottom"].to_dict()

#     def assign_pop_label(item_id):
#         if top_labels.get(item_id, 0) == 1:
#             return 1
#         elif bottom_labels.get(item_id, 0) == 1:
#             return -1
#         else:
#             return 0

#     df_merged["popularity_label"] = df_merged["item_id:token"].apply(assign_pop_label)

#     # -------------------------------
#     # Step 5: Save the final DataFrame to a CSV file.
#     # -------------------------------
#     df_final = df_merged.sort_values(by="interaction_count", ascending=False).reset_index(drop=True)
#     output_csv = os.path.join(dataset_path, "item_popularity_labels_with_titles.csv")
#     df_final.to_csv(output_csv, index=False)
#     print(f"CSV file '{output_csv}' created successfully.")
