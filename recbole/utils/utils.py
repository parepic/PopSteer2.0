# -*- coding: utf-8 -*-
# @Time   : 2020/7/17
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2021/3/8, 2022/7/12, 2023/2/11
# @Author : Jiawei Guan, Lei Wang, Gaowei Zhang
# @Email  : guanjw@ruc.edu.cn, zxcptss@gmail.com, zgw2022101006@ruc.edu.cn

"""
recbole.utils.utils
################################
"""

import datetime
import importlib
import os
import random
import pandas as pd
import h5py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from texttable import Texttable


from recbole.utils.enum_type import ModelType


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        "general_recommender",
        "context_aware_recommender",
        "sequential_recommender",
        "knowledge_aware_recommender",
        "exlib_recommender",
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = ".".join(["recbole.model", submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError(
            "`model_name` [{}] is not the name of an existing model.".format(model_name)
        )
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    try:
        return getattr(
            importlib.import_module("recbole.trainer"), model_name + "Trainer"
        )
    except AttributeError:
        if model_type == ModelType.KNOWLEDGE:
            return getattr(importlib.import_module("recbole.trainer"), "KGTrainer")
        elif model_type == ModelType.TRADITIONAL:
            return getattr(
                importlib.import_module("recbole.trainer"), "TraditionalTrainer"
            )
        else:
            return getattr(importlib.import_module("recbole.trainer"), "Trainer")


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r"""validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value >= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value <= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, valid_metric=None):
    r"""return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result["Recall@10"]


def dict2str(result_dict):
    r"""convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return "    ".join(
        [str(metric) + " : " + str(value) for metric, value in result_dict.items()]
    )


def init_seed(seed, reproducibility):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_tensorboard(logger):
    r"""Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for
    visualization within the TensorBoard UI.
    For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

    Args:
        logger: its output filename is used to name the SummaryWriter's log_dir.
                If the filename is not available, we will name the log_dir according to the current time.

    Returns:
        SummaryWriter: it will write out events and summaries to the event file.
    """
    base_path = "log_tensorboard"

    dir_name = None
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            dir_name = os.path.basename(getattr(handler, "baseFilename")).split(".")[0]
            break
    if dir_name is None:
        dir_name = "{}-{}".format("model", get_local_time())

    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer


def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return "{:.2f} G/{:.2f} G".format(reserved, total)


def get_flops(model, dataset, device, logger, transform, verbose=False):
    r"""Given a model and dataset to the model, compute the per-operator flops
    of the given model.
    Args:
        model: the model to compute flop counts.
        dataset: dataset that are passed to `model` to count flops.
        device: cuda.device. It is the device that the model run on.
        verbose: whether to print information of modules.

    Returns:
        total_ops: the number of flops for each operation.
    """
    if model.type == ModelType.DECISIONTREE:
        return 1
    if model.__class__.__name__ == "Pop":
        return 1

    import copy

    model = copy.deepcopy(model)

    def count_normalization(m, x, y):
        x = x[0]
        flops = torch.DoubleTensor([2 * x.numel()])
        m.total_ops += flops

    def count_embedding(m, x, y):
        x = x[0]
        nelements = x.numel()
        hiddensize = y.shape[-1]
        m.total_ops += nelements * hiddensize

    class TracingAdapter(torch.nn.Module):
        def __init__(self, rec_model):
            super().__init__()
            self.model = rec_model

        def forward(self, interaction):
            return self.model.predict(interaction)

    custom_ops = {
        torch.nn.Embedding: count_embedding,
        torch.nn.LayerNorm: count_normalization,
    }
    wrapper = TracingAdapter(model)
    inter = dataset[torch.tensor([1])].to(device)
    inter = transform(dataset, inter)
    inputs = (inter,)
    from thop.profile import register_hooks
    from thop.vision.basic_hooks import count_parameters

    handler_collection = {}
    fn_handles = []
    params_handles = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                logger.info("Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                logger.info("Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and verbose:
                logger.warning(
                    "[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params."
                    % m_type
                )

        if fn is not None:
            handle_fn = m.register_forward_hook(fn)
            handle_paras = m.register_forward_hook(count_parameters)
            handler_collection[m] = (
                handle_fn,
                handle_paras,
            )
            fn_handles.append(handle_fn)
            params_handles.append(handle_paras)
        types_collection.add(m_type)

    prev_training_status = wrapper.training

    wrapper.eval()
    wrapper.apply(add_hooks)

    with torch.no_grad():
        wrapper(*inputs)

    def dfs_count(module: nn.Module, prefix="\t"):
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            next_dict = {}
            if m in handler_collection and not isinstance(
                m, (nn.Sequential, nn.ModuleList)
            ):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params

        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(wrapper)

    # reset wrapper to original status
    wrapper.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")
    for i in range(len(fn_handles)):
        fn_handles[i].remove()
        params_handles[i].remove()

    return total_ops


def list_to_latex(convert_list, bigger_flag=True, subset_columns=[]):
    result = {}
    for d in convert_list:
        for key, value in d.items():
            if key in result:
                result[key].append(value)
            else:
                result[key] = [value]

    df = pd.DataFrame.from_dict(result, orient="index").T

    if len(subset_columns) == 0:
        tex = df.to_latex(index=False)
        return df, tex

    def bold_func(x, bigger_flag):
        if bigger_flag:
            return np.where(x == np.max(x.to_numpy()), "font-weight:bold", None)
        else:
            return np.where(x == np.min(x.to_numpy()), "font-weight:bold", None)

    style = df.style
    style.apply(bold_func, bigger_flag=bigger_flag, subset=subset_columns)
    style.format(precision=4)

    num_column = len(df.columns)
    column_format = "c" * num_column
    tex = style.hide(axis="index").to_latex(
        caption="Result Table",
        label="Result Table",
        convert_css=True,
        hrules=True,
        column_format=column_format,
    )

    return df, tex


def get_environment(config):
    gpu_usage = (
        get_gpu_usage(config["device"])
        if torch.cuda.is_available() and config["use_gpu"]
        else "0.0 / 0.0"
    )

    import psutil

    memory_used = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    memory_total = psutil.virtual_memory()[0] / 1024**3
    memory_usage = "{:.2f} G/{:.2f} G".format(memory_used, memory_total)
    cpu_usage = "{:.2f} %".format(psutil.cpu_percent(interval=1))
    """environment_data = [
        {"Environment": "CPU", "Usage": cpu_usage,},
        {"Environment": "GPU", "Usage": gpu_usage, },
        {"Environment": "Memory", "Usage": memory_usage, },
    ]"""

    table = Texttable()
    table.set_cols_align(["l", "c"])
    table.set_cols_valign(["m", "m"])
    table.add_rows(
        [
            ["Environment", "Usage"],
            ["CPU", cpu_usage],
            ["GPU", gpu_usage],
            ["Memory", memory_usage],
        ]
    )

    return table


import torch
import pandas as pd
from pathlib import Path
from typing import Union


# def compute_neuron_stats_by_row(
#         activations: torch.Tensor,
#         dataset: str
#     ) -> None:
    
#     labels_csv_path = rf"./dataset/{dataset}/item_popularity_labels.csv"
#     popular_out = rf"./dataset/{dataset}/neuron_stats_popular.csv"
#     unpopular_out = rf"./dataset/{dataset}/neuron_stats_unpopular.csv"
#     cohens_d_out = rf"./dataset/{dataset}/cohens_d.csv"
#     if activations.ndim != 2:
#         raise ValueError("`activations` must have shape (B, N)")
#     B, N = activations.shape

#     # ── 1. Load popularity labels ───────────────────────────────────────────────
#     label_ser = (
#         pd.read_csv(labels_csv_path, usecols=["item_id:token", "popularity_label"])
#         .rename(columns={"item_id:token": "item_id"})
#         .set_index("item_id")["popularity_label"]
#     )

#     # ── 2. Build per-row label tensor (1, −1, 0) ───────────────────────────────
#     labels = torch.zeros(B, dtype=torch.int8)
#     known_idx = label_ser.index.intersection(range(B))
#     labels[known_idx] = torch.tensor(label_ser.loc[known_idx].values, dtype=torch.int8)

#     pop_mask  = labels ==  1
#     unpop_mask = labels == -1

#     # Helper: stats for a boolean mask
#     def _stats(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
#         n = int(mask.sum().item())
#         if n:
#             subset = activations[mask]        # (n, N)
#             mean  = subset.mean(0)            # (N,)
#             sd    = subset.std(0, unbiased=False)
#         else:
#             mean = torch.zeros(N)
#             sd   = torch.zeros(N)
#         return mean, sd, n

#     # ── 3. Compute group stats ──────────────────────────────────────────────────
#     mean_pop,  sd_pop,  n_pop  = _stats(pop_mask)
#     mean_unp,  sd_unp,  n_unp  = _stats(unpop_mask)

#     # ── 4. Save per-group CSVs ──────────────────────────────────────────────────
#     def _to_csv(fname: str | Path, mean: torch.Tensor, sd: torch.Tensor):
#         pd.DataFrame({
#             "neuron": range(N),
#             "mean":   mean.tolist(),
#             "sd":     sd.tolist(),
#         }).to_csv(fname, index=False)

#     _to_csv(popular_out,   mean_pop, sd_pop)
#     _to_csv(unpopular_out, mean_unp, sd_unp)

#     # ── 5. Cohen’s d per neuron ────────────────────────────────────────────────
#     # pooled SD: sqrt( ((n1-1)*s1² + (n2-1)*s2²) / (n1+n2−2) )
#     # handle zero-row or zero-variance cases gracefully
#     denom = max(n_pop + n_unp - 2, 1)                      # scalar, ≥1
#     pooled_var = ((n_pop - 1) * sd_pop.pow(2) +
#                   (n_unp - 1) * sd_unp.pow(2)) / denom
#     pooled_sd = torch.sqrt(pooled_var)

#     valid = (pooled_sd != 0) & (n_pop > 0) & (n_unp > 0)
#     cohens_d = torch.full((N,), float('nan'))
#     cohens_d[valid] = (mean_pop[valid] - mean_unp[valid]) / pooled_sd[valid]

#     pd.DataFrame({
#         "neuron":   range(N),
#         "cohens_d": cohens_d.tolist(),
#     }).to_csv(cohens_d_out, index=False)


def compute_weighted_neuron_stats_by_row_item(
        activations: torch.Tensor,
        dataset: str,
        side: str
    ) -> None:
    labels_csv_path = rf"./dataset/{dataset}/{side}_popularity_labels.csv"
    popular_out = rf"./dataset/{dataset}/{side}/neuron_stats_popular.csv"
    unpopular_out = rf"./dataset/{dataset}/{side}/neuron_stats_unpopular.csv"
    cohens_d_out = rf"./dataset/{dataset}/{side}/cohens_d.csv"
    if activations.ndim != 2:
        raise ValueError("`activations` must have shape (B, N)")
    B, N = activations.shape

    df = pd.read_csv(labels_csv_path, usecols=[rf"{side}_id:token", "popularity_label", "pop_score"])
    df['item_id'] = df[rf"{side}_id:token"].astype(int)  # Assuming general 'item_id' for index
    label_ser = df.set_index('item_id')["popularity_label"]
    pop_score_ser = df.set_index('item_id')["pop_score"]

    labels = torch.zeros(B, dtype=torch.int8)
    known_idx = label_ser.index.intersection(range(B))
    labels[known_idx] = torch.tensor(label_ser.loc[known_idx].values, dtype=torch.int8)

    pop_scores = torch.zeros(B, dtype=torch.float)
    pop_scores[known_idx] = torch.tensor(pop_score_ser.loc[known_idx].values, dtype=torch.float)
    # Normalize pop_scores to [0, 1]
    min_pop = pop_scores.min()
    max_pop = pop_scores.max()
    if max_pop > min_pop:
        pop_scores = (pop_scores - min_pop) / (max_pop - min_pop)

    pop_mask = labels == 1
    unpop_mask = labels == -1  # Assuming -1 for unpopular, adjust if necessary to ==0

    # Helper: weighted stats for a boolean mask
    def _stats(mask: torch.Tensor, is_pop: bool) -> tuple[torch.Tensor, torch.Tensor, float]:
        mask_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        n_items = len(mask_idx)
        if n_items == 0:
            return torch.zeros(N), torch.zeros(N), 0.0
        subset = activations[mask_idx]  # (n, N)
        group_pop_scores = pop_scores[mask_idx]
        weights = group_pop_scores if is_pop else (1.0 - group_pop_scores)
        effective_n = weights.sum().item()
        if effective_n <= 0:
            return torch.zeros(N), torch.zeros(N), 0.0
        # Weighted mean
        mean = torch.sum(weights.unsqueeze(1) * subset, dim=0) / effective_n
        # Weighted variance (population style, matching original std unbiased=False)
        var = torch.sum(weights.unsqueeze(1) * (subset - mean.unsqueeze(0))**2, dim=0) / effective_n
        sd = torch.sqrt(var)
        return mean, sd, effective_n

    # Compute group stats
    mean_pop, sd_pop, effective_n_pop = _stats(pop_mask, is_pop=True)
    mean_unp, sd_unp, effective_n_unp = _stats(unpop_mask, is_pop=False)

    # Save per-group CSVs
    def _to_csv(fname: str, mean: torch.Tensor, sd: torch.Tensor):
        pd.DataFrame({
            "neuron": range(N),
            "mean":   mean.tolist(),
            "sd":     sd.tolist(),
        }).to_csv(fname, index=False)

    _to_csv(popular_out, mean_pop, sd_pop)
    _to_csv(unpopular_out, mean_unp, sd_unp)

    # Cohen’s d per neuron
    denom = max(effective_n_pop + effective_n_unp - 2, 1)
    pooled_var = ((effective_n_pop - 1) * sd_pop.pow(2) +
                  (effective_n_unp - 1) * sd_unp.pow(2)) / denom
    pooled_sd = torch.sqrt(pooled_var)

    valid = (pooled_sd != 0) & (effective_n_pop > 0) & (effective_n_unp > 0)
    cohens_d = torch.full((N,), float('nan'))
    cohens_d[valid] = (mean_pop[valid] - mean_unp[valid]) / pooled_sd[valid]

    pd.DataFrame({
        "neuron":   range(N),
        "cohens_d": cohens_d.tolist(),
    }).to_csv(cohens_d_out, index=False)



def compute_neuron_stats_by_row(
        activations: torch.Tensor,
        dataset: str,
        side: str
    ) -> None:
    labels_csv_path = rf"./dataset/{dataset}/{side}_popularity_labels.csv"
    popular_out = rf"./dataset/{dataset}/{side}/neuron_stats_popular.csv"
    unpopular_out = rf"./dataset/{dataset}/{side}/neuron_stats_unpopular.csv"
    cohens_d_out = rf"./dataset/{dataset}/{side}/cohens_d.csv"
    if activations.ndim != 2:
        raise ValueError("`activations` must have shape (B, N)")
    B, N = activations.shape


    label_ser = (
        pd.read_csv(labels_csv_path, usecols=[rf"{side}_id:token", "popularity_label"])
        .set_index(rf"{side}_id:token")["popularity_label"]
    )

    labels = torch.zeros(B, dtype=torch.int8)
    known_idx = label_ser.index.intersection(range(B))
    labels[known_idx] = torch.tensor(label_ser.loc[known_idx].values, dtype=torch.int8)

    pop_mask  = labels ==  1
    unpop_mask = labels == 0

    # Helper: stats for a boolean mask
    def _stats(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        n = int(mask.sum().item())
        if n:
            subset = activations[mask]        # (n, N)
            mean  = subset.mean(0)            # (N,)
            sd    = subset.std(0, unbiased=False)
        else:
            mean = torch.zeros(N)
            sd   = torch.zeros(N)
        return mean, sd, n

    # ── 3. Compute group stats ──────────────────────────────────────────────────
    mean_pop,  sd_pop,  n_pop  = _stats(pop_mask)
    mean_unp,  sd_unp,  n_unp  = _stats(unpop_mask)

    # ── 4. Save per-group CSVs ──────────────────────────────────────────────────
    def _to_csv(fname: str | Path, mean: torch.Tensor, sd: torch.Tensor):
        pd.DataFrame({
            "neuron": range(N),
            "mean":   mean.tolist(),
            "sd":     sd.tolist(),
        }).to_csv(fname, index=False)

    _to_csv(popular_out,   mean_pop, sd_pop)
    _to_csv(unpopular_out, mean_unp, sd_unp)

    # ── 5. Cohen’s d per neuron ────────────────────────────────────────────────
    # pooled SD: sqrt( ((n1-1)*s1² + (n2-1)*s2²) / (n1+n2−2) )
    # handle zero-row or zero-variance cases gracefully
    denom = max(n_pop + n_unp - 2, 1)                      # scalar, ≥1
    pooled_var = ((n_pop - 1) * sd_pop.pow(2) +
                  (n_unp - 1) * sd_unp.pow(2)) / denom
    pooled_sd = torch.sqrt(pooled_var)

    valid = (pooled_sd != 0) & (n_pop > 0) & (n_unp > 0)
    cohens_d = torch.full((N,), float('nan'))
    cohens_d[valid] = (mean_pop[valid] - mean_unp[valid]) / pooled_sd[valid]

    pd.DataFrame({
        "neuron":   range(N),
        "cohens_d": cohens_d.tolist(),
    }).to_csv(cohens_d_out, index=False)






def get_extreme_correlations(file_name: str, dataset=None):
    """
    Retrieves all positive and all negative correlation indexes and their values.

    Parameters:
    file_name (str): CSV file name containing correlation values.
    unpopular_only (bool): If True, returns an empty positive list and the full negative list.

    Returns:
    tuple:
      - pos_list: list of (index, value) for all positives (empty if unpopular_only=True)
      - neg_list: list of (index, value) for all negatives
    """
    

    # 1) load
    df = pd.read_csv(rf"./dataset/{dataset}/{file_name}")
    # indices = pd.read_csv(r"./dataset/ml-1m/nonzero_activations_sasrecsae_k48-32.csv")["index"].tolist()
    # # 2) if they passed a subset of row positions, slice with .iloc
    # if indices is not None:
    #     df = df.iloc[indices]

    # 3) split out positives / negatives
    pos_series = df.loc[df["cohens_d"] > 0, "cohens_d"]
    neg_series = df.loc[df["cohens_d"] < 0, "cohens_d"]

    # 4) zip index-labels (which by default are 0,1,2… or the original row numbers)
    pos_list = list(pos_series.items())  # each item is (index_label, value)
    neg_list = list(neg_series.items())


    return pos_list, neg_list


import matplotlib.pyplot as plt


def plot_tensor_sorted_by_popularity(tensor: torch.Tensor, dataset: str):
    """
    Sorts the given tensor (index 1 onwards) based on the pop_score from CSV,
    and plots the sorted tensor values.

    Parameters:
        tensor (torch.Tensor): 1D tensor of size N+1, where index 0 is unused (item ID 0 doesn't exist).
        csv_path (str): Path to the CSV file with 'item_id:token' and 'pop_score' columns.
    """
    # Load CSV
    df = pd.read_csv(rf"./dataset/{dataset}/item_popularity_labels.csv")

    # Use item_id:token as integer item ID
    df['item_id'] = df['item_id:token'].astype(int)

    # Sanity check
    assert tensor.shape[0] == df['item_id'].max() + 1, "Tensor size must match max item ID + 1"

    # Build pop_score tensor aligned to item ID
    pop_scores = torch.zeros_like(tensor)
    pop_scores[df['item_id'].values] = torch.tensor(df['pop_score'].values, dtype=torch.long)

    # Skip index 0 (no item with ID 0)
    tensor_valid = tensor[1:]
    pop_scores_valid = pop_scores[1:]

    # Sort tensor by popularity score
    sorted_indices = torch.argsort(pop_scores_valid)
    sorted_tensor = tensor_valid[sorted_indices]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sorted_tensor)), sorted_tensor.numpy())
    plt.xlabel('Items sorted by pop_score')
    plt.ylabel('Tensor values')
    plt.title('Tensor values sorted by item popularity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import csv

import os
import csv
import matplotlib.pyplot as plt

def plot_ndcg_vs_fairness(show=True, dataset=None, add_lightgcn=True, model="LightGCN"):
    """
    Create three scatter plots: NDCG vs each fairness metric (dltc@10, avgpop@10, gini@10),
    overlaying points from user-side, item-side, full, FAIR CSV results, and (optionally) a LightGCN baseline.

    Parameters
    ----------
    show : bool
        If True, call plt.show() at the end.
    dataset : str
        Name of the dataset directory under ./dataset/ that contains the CSV files.
    add_lightgcn : bool
        Whether to plot the LightGCN reference point.

    Returns
    -------
    figs : dict[str, matplotlib.figure.Figure]
        Mapping from metric key to the created Figure.
    """
    if dataset is None:
        raise ValueError("Please provide dataset name (e.g. dataset='lastfm').")

    # user_file = rf"dataset/{dataset}/results/PopSteer_{dataset}_user.csv"
    # item_file = rf"dataset/{dataset}/results/PopSteer_{dataset}_item.csv"
    # full_file = rf"dataset/{dataset}/results/PopSteer_{dataset}_full.csv"
    # fair_file = rf"dataset/{dataset}/results/FAIR_{dataset}.csv"
    user_file = rf"dataset/{dataset}/results/{model}_user_{dataset}-new.csv"
    item_file = rf"dataset/{dataset}/results/{model}_item_{dataset}-new.csv"
    full_file = rf"dataset/{dataset}/results/{model}_full_{dataset}-new.csv"
    fair_file = rf"dataset/{dataset}/results/{model}_fair_{dataset}-new.csv"

    def load_csv(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "ndcg": float(row["ndcg"]),
                        "dltc@10": float(row["dltc@10"]),
                        "avgpop@10": float(row["avgpop@10"]),
                        "gini@10": float(row["gini@10"]),
                        "cov@10": float(row["cov@10"]),

                    })
                except (KeyError, ValueError):
                    # Skip malformed / incomplete rows
                    continue
        return rows

    user_rows = load_csv(user_file)
    item_rows = None
    full_rows = None
    fair_rows = load_csv(fair_file)

    # LightGCN reference metrics (single point). Add other dataset baselines if needed.
    # lightgcn_point_lastfm = {
    #     "ndcg": 0.8289,
    #     "dltc@10": 0.7291,
    #     "avgpop@10": 91.7490,
    #     "gini@10": 0.7053,
    #     "cov@10": 0.8051,
    # }

    lightgcn_point_lastfm = {
        "ndcg": 0.6103,
        "dltc@10": 0.7946,
        "avgpop@10": 87.6849,
        "gini@10": 0.6555,
        "cov@10": 0.8461,
    }


    lightgcn_point_ml_1m = {
        "ndcg": 0.2190,
        "dltc@10": 0.3016,
        "avgpop@10": 1101.8943,
        "gini@10": 0.8696,
        "cov@10": 0.5236,
    }

    lightgcn_point = None
    # if dataset.lower() == "lastfm":
    #     lightgcn_point = lightgcn_point_lastfm
    # if dataset.lower() == "ml-1mm":
    #     lightgcn_point = lightgcn_point_ml_1m

    fairness_metrics = [
        ("dltc@10", "Deep LT Coverage @10"),
        ("avgpop@10", "Average Popularity @10"),
        ("gini@10", "Gini Index @10"),
        ("cov@10", "Coverage @10"),
    ]

    figs = {}

    for metric_key, metric_full_name in fairness_metrics:
        fig, ax = plt.subplots()

        if user_rows:
            ax.scatter(
                [r["ndcg"] for r in user_rows],
                [r[metric_key] for r in user_rows],
                marker="o",
                label="User-side",
                alpha=0.8,
                edgecolors="none"
            )
        if item_rows:
            ax.scatter(
                [r["ndcg"] for r in item_rows],
                [r[metric_key] for r in item_rows],
                marker="s",
                label="Item-side",
                alpha=0.8,
                edgecolors="none"
            )
        if full_rows:
            ax.scatter(
                [r["ndcg"] for r in full_rows],
                [r[metric_key] for r in full_rows],
                marker="^",
                label="Both-sides",
                alpha=0.8,
                edgecolors="none"
            )
        if fair_rows:
            ax.scatter(
                [r["ndcg"] for r in fair_rows],
                [r[metric_key] for r in fair_rows],
                marker="D",
                label="FAIR",
                alpha=0.85,
                edgecolors="none"
            )

        if add_lightgcn and lightgcn_point is not None:
            ax.scatter(
                [lightgcn_point["ndcg"]],
                [lightgcn_point[metric_key]],
                marker="*",
                s=180,
                label="LightGCN",
                alpha=0.95,
                edgecolors="black",
                linewidths=0.6
            )

        ax.set_xlabel("NDCG@10")
        ax.set_ylabel(metric_full_name)
        ax.set_title(f"{dataset} NDCG vs {metric_full_name}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend()
        figs[metric_key] = fig

    if show:
        plt.show()

    return figs

import shutil

def remove_sparse_users_items(n: int, dataset: str, base_dir: str = "./dataset") -> None:
    ds_dir = Path(base_dir) / dataset
    inter_path = ds_dir / f"{dataset}.inter"
    # item_path  = ds_dir / f"{dataset}.item"
    inter_bak  = ds_dir / f"{dataset}.inter.original"
    # item_bak   = ds_dir / f"{dataset}.item.original"

    # --- Step 0: Backups (only once) ---
    if not inter_bak.exists():
        shutil.copy2(inter_path, inter_bak)
    # if not item_bak.exists():
        # shutil.copy2(item_path, item_bak)

    # --- Step 1: Load ---
    interactions = pd.read_csv(inter_path, sep="\t", header=0)
    # items        = pd.read_csv(item_path,  sep="\t", header=0)

    # --- Step 2: Iterative filtering ---
    iteration = 0
    while True:
        iteration += 1
        before = interactions.shape[0]

        valid_users = interactions["session_id:token"].value_counts()
        valid_users = valid_users[valid_users >= n].index
        interactions = interactions[interactions["session_id:token"].isin(valid_users)]

        valid_items = interactions["item_id:token"].value_counts()
        valid_items = valid_items[valid_items >= n].index
        interactions = interactions[interactions["item_id:token"].isin(valid_items)]

        after = interactions.shape[0]
        print(f"Iteration {iteration}: {before} -> {after} interactions remain")
        if after == before:
            break

    # --- Step 3: Sync items ---
    # items = items[items["item_id:token"].isin(interactions["item_id:token"])]

    # --- Step 4: Overwrite originals (atomic-ish) ---
    tmp_inter = inter_path.with_suffix(".inter.tmp")
    # tmp_item  = item_path.with_suffix(".item.tmp")

    interactions.to_csv(tmp_inter, sep="\t", index=False)
    # items.to_csv(tmp_item, sep="\t", index=False)

    tmp_inter.replace(inter_path)
    # tmp_item.replace(item_path)

    print(f"Done. Wrote {interactions.shape[0]} interactions and {len(interactions['item_id:token'].unique())} items.")



def create_pop_unpop_mappings(dataset: str, embeddings: torch.Tensor) -> None:
    """
    Creates mapping CSV files for popular and unpopular item pairs based on embeddings and popularity labels.

    Args:
        embeddings (torch.Tensor): Tensor of item embeddings with shape (N, 64), where N is the number of items
                                   and the nth row corresponds to item ID n (0 to N-1).
        item_pop_csv (str): Path to the input CSV file containing 'item_id:token' and 'popularity_label' columns.
        pop_mapping_csv (str): Path to save the popular mapping CSV (columns: item_id, paired_id).
        unpop_mapping_csv (str): Path to save the unpopular mapping CSV (columns: item_id, paired_id).
    """
    
    dataset_path = Path(".", "dataset", dataset)
    item_pop_csv = dataset_path / "item_popularity_labels.csv"
    unpop_mapping_csv = dataset_path / "unpop_mapping.csv"
    pop_mapping_csv = dataset_path / "pop_mapping.csv"

    df_items = pd.read_csv(item_pop_csv, usecols=["item_id:token", "popularity_label"])
    df_items = df_items.dropna(subset=["popularity_label"])
    df_items = df_items.rename(columns={"item_id:token": "item_id"})
    df_items["item_id"] = df_items["item_id"].astype(int)

    # Get N from embeddings
    N = embeddings.shape[0]

    # Extract popular and unpopular item IDs
    popular_ids = df_items[df_items["popularity_label"] == 1]["item_id"].values
    unpopular_ids = df_items[df_items["popularity_label"] == -1]["item_id"].values

    # Create a dict for quick label lookup (default to 0 if missing)
    label_dict = df_items.set_index("item_id")["popularity_label"].to_dict()

    # Unpopular mapping
    if len(unpopular_ids) > 0:
        unpop_embeddings = embeddings[unpopular_ids]  # (num_unpop, 64)
        sim_unpop = embeddings @ unpop_embeddings.T  # (N, num_unpop)
    else:
        sim_unpop = torch.empty((N, 0))  # Handle edge case with no unpopular items

    pairs_unpop = []
    for i in range(N):
        label = label_dict.get(i, 0)
        if label == -1:
            pairs_unpop.append(i)
        else:
            if len(unpopular_ids) == 0:
                pairs_unpop.append(i)  # Fallback to self if no unpopular items
            else:
                closest_idx = sim_unpop[i].argmax().item()
                pairs_unpop.append(unpopular_ids[closest_idx])

    df_unpop = pd.DataFrame({"item_id": range(N), "paired_id": pairs_unpop})
    df_unpop.to_csv(unpop_mapping_csv, index=False)

    # Popular mapping
    if len(popular_ids) > 0:
        pop_embeddings = embeddings[popular_ids]  # (num_pop, 64)
        sim_pop = embeddings @ pop_embeddings.T  # (N, num_pop)
    else:
        sim_pop = torch.empty((N, 0))  # Handle edge case with no popular items

    pairs_pop = []
    for i in range(N):
        label = label_dict.get(i, 0)
        if label == 1:
            pairs_pop.append(i)
        else:
            if len(popular_ids) == 0:
                pairs_pop.append(i)  # Fallback to self if no popular items
            else:
                closest_idx = sim_pop[i].argmax().item()
                pairs_pop.append(popular_ids[closest_idx])

    df_pop = pd.DataFrame({"item_id": range(N), "paired_id": pairs_pop})
    df_pop.to_csv(pop_mapping_csv, index=False)


def create_pop_unpop_mappings(dataset: str, embeddings: torch.Tensor) -> None:
    """
    Creates mapping CSV files for popular and unpopular item pairs based on embeddings and popularity labels.

    Args:
        embeddings (torch.Tensor): Tensor of item embeddings with shape (N, 64), where N is the number of items
                                   and the nth row corresponds to item ID n (0 to N-1).
        item_pop_csv (str): Path to the input CSV file containing 'item_id:token' and 'popularity_label' columns.
        pop_mapping_csv (str): Path to save the popular mapping CSV (columns: item_id, paired_id).
        unpop_mapping_csv (str): Path to save the unpopular mapping CSV (columns: item_id, paired_id).
    """
    
    dataset_path = Path(".", "dataset", dataset)
    item_pop_csv = dataset_path / "item_popularity_labels.csv"
    unpop_mapping_csv = dataset_path / "unpop_mapping.csv"
    pop_mapping_csv = dataset_path / "pop_mapping.csv"

    df_items = pd.read_csv(item_pop_csv, usecols=["item_id:token", "popularity_label"])
    df_items = df_items.dropna(subset=["popularity_label"])
    df_items = df_items.rename(columns={"item_id:token": "item_id"})
    df_items["item_id"] = df_items["item_id"].astype(int)

    # Get N from embeddings
    N = embeddings.shape[0]

    # Extract popular and unpopular item IDs
    popular_ids = df_items[df_items["popularity_label"] == 1]["item_id"].values
    unpopular_ids = df_items[df_items["popularity_label"] == -1]["item_id"].values

    # Create a dict for quick label lookup (default to 0 if missing)
    label_dict = df_items.set_index("item_id")["popularity_label"].to_dict()

    # Unpopular mapping
    if len(unpopular_ids) > 0:
        unpop_embeddings = embeddings[unpopular_ids]  # (num_unpop, 64)
        sim_unpop = embeddings @ unpop_embeddings.T  # (N, num_unpop)
    else:
        sim_unpop = torch.empty((N, 0))  # Handle edge case with no unpopular items

    pairs_unpop = []
    for i in range(N):
        if i == 0:
            pairs_unpop.append(0)
            continue
        label = label_dict.get(i, 0)
        if label == -1:
            pairs_unpop.append(i)
        else:
            if len(unpopular_ids) == 0:
                pairs_unpop.append(i)  # Fallback to self if no unpopular items
            else:
                closest_idx = sim_unpop[i].argmax().item()
                pairs_unpop.append(unpopular_ids[closest_idx])

    df_unpop = pd.DataFrame({"item_id": range(N), "paired_id": pairs_unpop})
    df_unpop.to_csv(unpop_mapping_csv, index=False)

    # Popular mapping
    if len(popular_ids) > 0:
        pop_embeddings = embeddings[popular_ids]  # (num_pop, 64)
        sim_pop = embeddings @ pop_embeddings.T  # (N, num_pop)
    else:
        sim_pop = torch.empty((N, 0))  # Handle edge case with no popular items

    pairs_pop = []
    for i in range(N):
        if i == 0:
            pairs_pop.append(0)
            continue
        label = label_dict.get(i, 0)
        if label == 1:
            pairs_pop.append(i)
        else:
            if len(popular_ids) == 0:
                pairs_pop.append(i)  # Fallback to self if no popular items
            else:
                closest_idx = sim_pop[i].argmax().item()
                pairs_pop.append(popular_ids[closest_idx])

    df_pop = pd.DataFrame({"item_id": range(N), "paired_id": pairs_pop})
    df_pop.to_csv(pop_mapping_csv, index=False)



def replace_with_mappings(sequences: torch.Tensor, popular: bool, dataset: str) -> torch.Tensor:
    """
    Replaces item IDs in the input sequences tensor with their mapped paired IDs based on the popularity flag.

    Args:
        sequences (torch.Tensor): Input tensor of shape (B, M) containing item IDs (integers from 0 to N-1).
        popular (bool): If True, use popular mapping; if False, use unpopular mapping.
        pop_mapping_csv (str): Path to the popular mapping CSV (columns: item_id, paired_id).
        unpop_mapping_csv (str): Path to the unpopular mapping CSV (columns: item_id, paired_id).

    Returns:
        torch.Tensor: Output tensor of shape (B, M) with replaced item IDs.
    """
    dataset_path = Path(".", "dataset", dataset)
    unpop_mapping_csv = dataset_path / "unpop_mapping.csv"
    pop_mapping_csv = dataset_path / "pop_mapping.csv"

    mapping_csv = pop_mapping_csv if popular else unpop_mapping_csv
    df_map = pd.read_csv(mapping_csv)
    max_id = df_map['item_id'].max()
    map_list = [0] * (max_id + 1)
    for _, row in df_map.iterrows():
        map_list[int(row['item_id'])] = int(row['paired_id'])
    map_tensor = torch.tensor(map_list, dtype=torch.long, device=sequences.device)
    result = map_tensor[sequences]
    return result


def save_batch_activations(bulk_data, neuron_count, dataset, popular):
    if popular == True:
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final_pop.h5"
    if popular == False:
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final_unpop.h5"
        
    bulk_data = bulk_data.permute(1, 0).detach().cpu().numpy()  # [neuron_count, batch_size]
    real_batch_size = bulk_data.shape[1]  # Might be < batch_size in final step

    if not os.path.exists(file_path):
        with h5py.File(file_path, "w") as f:
            max_shape = (neuron_count, None)
            f.create_dataset(
                "dataset",
                data=bulk_data,
                maxshape=max_shape,
                chunks=(neuron_count, real_batch_size),
                dtype="float32",
            )
    else:
        with h5py.File(file_path, "a") as f:
            dset = f["dataset"]
            current_cols = dset.shape[1]
            new_cols = current_cols + real_batch_size
            dset.resize((neuron_count, new_cols))
            dset[:, current_cols:new_cols] = bulk_data
            

def save_mean_SD(dataset, popular=None):
    # Load your .h5 file
    if popular == True:  
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final_pop.h5"
    elif popular == False:  
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final_unpop.h5"
    dataset_name = 'dataset'  # Replace with actual dataset name inside the h5 file

    # Load the real indices from the filtered CSV
    # index_csv = r"./dataset/ml-1m/nonzero_activations_sasrecsae_k48-32.csv"
    # real_indices = pd.read_csv(index_csv, index_col=0).index.tolist()

    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][()]  # Reads full dataset into memory

    # Compute mean and standard deviation for each row
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    # Combine into a DataFrame with the correct index
    df = pd.DataFrame({
        'mean': means,
        'sd': stds,
    })

    if popular == True:  
        output_csv_path = rf"./dataset/{dataset}/user/neuron_stats_popular.csv"
    if popular == False:  
        output_csv_path = rf"./dataset/{dataset}/user/neuron_stats_unpopular.csv"
    df.to_csv(output_csv_path)
    print(f"Row-wise mean and std saved to {output_csv_path}")
    
    

def save_cohens_d(dataset):
    df1 = pd.read_csv(rf"./dataset/{dataset}/user/neuron_stats_popular.csv", index_col=0)
    df2 = pd.read_csv(rf"./dataset/{dataset}/user/neuron_stats_unpopular.csv", index_col=0)

    # Compute pooled standard deviation
    s_pooled = np.sqrt((df1['sd']**2 + df2['sd']**2) / 2)

    # Compute Cohen's d
    cohen_d = (df1['mean'] - df2['mean']) / s_pooled

    # Create result DataFrame with same index
    df_result = pd.DataFrame({'cohens_d': cohen_d})

    # Save to CSV with index column
    df_result.to_csv(rf"./dataset/{dataset}/user/cohens_d.csv")

    print("Cohen's d values saved to cohens_d.csv")



def make_items_popular(item_seq_len, dataset, n):
    item_labels = pd.read_csv(rf"./dataset/{dataset}/item_popularity_labels.csv")
    
    # Filter rows where popularity_label == -1
    filtered_items = item_labels[item_labels['popularity_label'] == 1]
    available_ids = filtered_items['item_id:token'].tolist()
    
    # Count how many items are in each row of the batch
    nonzero_counts = (item_seq_len != 0).sum(dim=1).tolist()
    selected_item_ids = []

    for count in nonzero_counts:
        sampled = pd.Series(available_ids).sample(n=count, replace=True).tolist()
        
        # Pad with 0s if needed to reach length 50
        if len(sampled) < n:
            sampled += [0] * (n - len(sampled))
        else:
            sampled = sampled[:n]  # In case count > 50 for any reason

        selected_item_ids.append(sampled)

    # Convert to tensor of shape (batch_size, 50)
    selected_tensor = torch.tensor(selected_item_ids)
    return selected_tensor


def make_items_unpopular(item_seq_len, dataset, n):

    item_labels = pd.read_csv(rf"./dataset/{dataset}/item_popularity_labels.csv")
    
    # Filter rows where popularity_label == -1
    filtered_items = item_labels[item_labels['popularity_label'] == -1]
    available_ids = filtered_items['item_id:token'].tolist()
    
    # Count how many items are in each row of the batch
    nonzero_counts = (item_seq_len != 0).sum(dim=1).tolist()
    selected_item_ids = []

    for count in nonzero_counts:
        sampled = pd.Series(available_ids).sample(n=count, replace=True).tolist()
        
        # Pad with 0s if needed to reach length 50
        if len(sampled) < n:
            sampled += [0] * (n - len(sampled))
        else:
            sampled = sampled[:n]  # In case count > 50 for any reason

        selected_item_ids.append(sampled)

    # Convert to tensor of shape (batch_size, 50)
    selected_tensor = torch.tensor(selected_item_ids)

    return selected_tensor



