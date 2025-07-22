from recbole.quick_start import load_data_and_model
from recbole.utils import (
    get_trainer,
)
import csv
import torch


def tune(args):
    if args.fair:
        tune_FAIR(args)
        exit()

    if args.config_json is None:
        config_dict = {
            "alpha": [0, 0],
            "steer": [0, 1],
            "analyze": True,
            "tail_ratio": 0.2,
            "metrics": ["Recall","MRR","NDCG","Hit","Precision","SAE_Loss_i", "SAE_Loss_u", "SAE_Loss_total", "Gini", "Deep_LT_Coverage", "GiniIndex", "TailPercentage", "AveragePopularity", "ItemCoverage"]        
            }
    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, dict=config_dict
    )
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.data_collect(train_data)
    # change2 = [0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
    # change1 = [0.0]
    change2 = [0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    change1 = [0.0]

    # change2 = [0.0, 0.5, 1, 1.5, 2.0, 2.5, 3.0]

    # change1 = [0, 50, 256, 512, 1024, 4096, 8192]

    metric_keys = [
        'mrr@10',
        'ndcg@10',
        'hit@10',
        'deep_lt_coverage@10',
        'giniindex@10',
        'averagepopularity@10',
        'itemcoverage@10'
    ]

    SHORT_NAMES = {
        'mrr@10': 'MRR@10',
        'ndcg@10': 'NDCG@10',
        'hit@10': 'HIT@10',
        'deep_lt_coverage@10': 'DLTC@10',
        'giniindex@10': 'GINI@10',
        'averagepopularity@10': 'AVGPOP@10',
        'itemcoverage@10': 'COV@10'
    }

    rows_raw = []
    for a_i in change1:
        for a_u in change2:
            trainer.model.recommendation_count = torch.zeros(trainer.model.n_items, dtype=torch.long, device=trainer.device)
            trainer.model.sae_module_i.alpha = a_i
            trainer.model.sae_module_u.alpha = a_u
            test_result = trainer.evaluate(
                valid_data,
                model_file=args.path,
                load_best_model=False,
                show_progress=config["show_progress"]
            )
            trainer.model.restore_item_e = None
            rows_raw.append({
                'alpha_u': a_u,
                'alpha_i': a_i,
                **{k: test_result[k] for k in metric_keys}
            })

    # Baseline: first (alpha_u, alpha_i) pair (assumes change lists start with 0.0)
    baseline = rows_raw[0]

    value_decimals = 4
    pct_decimals = 2
    show_zero_pct_on_baseline = False  # set True if you want (+0.00%)

    # Headers (rename alpha columns)
    header_labels = ['alpha_u', 'alpha_i'] + [SHORT_NAMES[k] for k in metric_keys]

    # Build formatted rows
    formatted_rows = []
    for i, r in enumerate(rows_raw):
        is_baseline = (i == 0)
        formatted_row = {
            'alpha_u': f"{r['alpha_u']:.2f}",
            'alpha_i': f"{r['alpha_i']:.2f}",
        }
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

    # Compute column widths
    col_width = {}
    for h in header_labels:
        max_cell = max(len(row[h]) for row in formatted_rows)
        col_width[h] = max(len(h), max_cell)

    # Print table
    header_line = " | ".join(f"{h:<{col_width[h]}}" for h in header_labels)
    sep_line = "-+-".join("-" * col_width[h] for h in header_labels)
    print(header_line)
    print(sep_line)
    for fr in formatted_rows:
        line = " | ".join(f"{fr[h]:<{col_width[h]}}" for h in header_labels)
        print(line)

    # --- Write selected results to CSV (with separate alphas) --
    csv_path = rf'./dataset/{config["dataset"]}/results/PopSteer_{config["dataset"]}_user.csv'
    fieldnames = ["alpha_u", "alpha_i", "ndcg", "dltc@10", "avgpop@10", "gini@10", "cov@10"]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_raw:
            writer.writerow({
                "alpha_u": r["alpha_u"],
                "alpha_i": r["alpha_i"],
                "ndcg": r["ndcg@10"],
                "dltc@10": r["deep_lt_coverage@10"],
                "avgpop@10": r["averagepopularity@10"],
                "gini@10": r["giniindex@10"],
                "cov@10": r["itemcoverage@10"],

            })

    return rows_raw, formatted_rows


def tune_FAIR(args):
    if args.config_json is None:
        config_dict = {
            "alpha": [0.5, 0.5],
            "metrics": ["Recall","MRR","NDCG","Hit","Precision", "Gini", "Deep_LT_Coverage", "GiniIndex", "TailPercentage", "AveragePopularity", "ItemCoverage" ]        
            }
    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, dict=config_dict
    )

    model.fair = True
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.data_collect(train_data)
    change1 = [0.3, 0.5, 0.7, 0.9, 0.99]
    change2 = [0.01, 0.05, 0.1]

    # change1 = [0.25, 0.5, 0.75, 1]
    # change2 = [0.25, 0.5, 0.75, 1]

    # change1 = [0, 50, 256, 512, 1024, 4096, 8192]

    metric_keys = [
        'mrr@10',
        'ndcg@10',
        'hit@10',
        'deep_lt_coverage@10',
        'giniindex@10',
        'averagepopularity@10',
        'itemcoverage@10'
    ]

    SHORT_NAMES = {
        'mrr@10': 'MRR@10',
        'ndcg@10': 'NDCG@10',
        'hit@10': 'HIT@10',
        'deep_lt_coverage@10': 'DLTC@10',
        'giniindex@10': 'GINI@10',
        'averagepopularity@10': 'AVGPOP@10',
        'itemcoverage@10': 'COV@10'
    }

    rows_raw = []
    for a_u in change1:
        for a_i in change2:
            trainer.model.recommendation_count = torch.zeros(trainer.model.n_items, dtype=torch.long, device=trainer.device)
            trainer.model.a1 = a_u
            trainer.model.a2 = a_i
            test_result = trainer.evaluate(
                valid_data,
                model_file=args.path,
                load_best_model=False,
                show_progress=config["show_progress"]
            )
            trainer.model.restore_item_e = None
            rows_raw.append({
                'alpha_u': a_u,
                'alpha_i': a_i,
                **{k: test_result[k] for k in metric_keys}
            })

    # Baseline: first (alpha_u, alpha_i) pair (assumes change lists start with 0.0)
    baseline = rows_raw[0]

    value_decimals = 4
    pct_decimals = 2
    show_zero_pct_on_baseline = False  # set True if you want (+0.00%)

    # Headers (rename alpha columns)
    header_labels = ['alpha_u', 'alpha_i'] + [SHORT_NAMES[k] for k in metric_keys]

    # Build formatted rows
    formatted_rows = []
    for i, r in enumerate(rows_raw):
        is_baseline = (i == 0)
        formatted_row = {
            'alpha_u': f"{r['alpha_u']:.2f}",
            'alpha_i': f"{r['alpha_i']:.2f}",
        }
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

    # Compute column widths
    col_width = {}
    for h in header_labels:
        max_cell = max(len(row[h]) for row in formatted_rows)
        col_width[h] = max(len(h), max_cell)

    # Print table
    header_line = " | ".join(f"{h:<{col_width[h]}}" for h in header_labels)
    sep_line = "-+-".join("-" * col_width[h] for h in header_labels)
    print(header_line)
    print(sep_line)
    for fr in formatted_rows:
        line = " | ".join(f"{fr[h]:<{col_width[h]}}" for h in header_labels)
        print(line)

    # --- Write selected results to CSV (with separate alphas) ---
    csv_path = rf'./dataset/{config["dataset"]}/results/FAIR_{config["dataset"]}.csv'
    fieldnames = ["alpha_u", "alpha_i", "ndcg", "dltc@10", "avgpop@10", "gini@10", "cov@10"]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_raw:
            writer.writerow({
                "alpha_u": r["alpha_u"],
                "alpha_i": r["alpha_i"],
                "ndcg": r["ndcg@10"],
                "dltc@10": r["deep_lt_coverage@10"],
                "avgpop@10": r["averagepopularity@10"],
                "gini@10": r["giniindex@10"],
                "cov@10": r["itemcoverage@10"],

            })

    return rows_raw, formatted_rows
