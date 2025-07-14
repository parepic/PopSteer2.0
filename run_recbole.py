# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse

from recbole.quick_start import run, run_recbole

if __name__ == "__main__":
    # parameter_dict = {
    # 'train_neg_sample_args': None,
    # }
    # run_recbole(model='SASRec', dataset='ml-100k', config_dict=parameter_dict)
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument('--sae_k', '-k', type=int, default=32,
                        help="Sparsity parameter K: keep only the top‑k activations per input in the SAE (Eq. 1).")

    parser.add_argument('--scale', '--scale_size', type=int, default=8,
                        dest='scale',
                        help="Scale factor s controlling the SAE hidden size relative to the input (s × d).")

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
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    config_dict = dict()
    if hasattr(args, "base_path") and args.base_path != "no path":
        config_dict["base_path"] = args.base_path
    if hasattr(args, "scale_size"):
        config_dict["sae_scale_size"] = args.base_path
    if hasattr(args, "sae_k"):
        config_dict["sae_k"] = args.sae_k

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
