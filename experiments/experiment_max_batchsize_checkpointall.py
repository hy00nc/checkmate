import argparse
import logging
import os
import pathlib
import shutil
import uuid
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas
import tensorflow as tf
from tqdm import tqdm

from experiments.common.definitions import remat_data_dir
from experiments.common.load_keras_model import MODEL_NAMES, get_keras_model
from experiments.common.graph_plotting import render_dfgraph
from experiments.common.profile.cost_model import CostModel
from experiments.common.profile.platforms import PLATFORM_CHOICES, platform_memory
from remat.core.schedule import ScheduledResult
from remat.core.enum_strategy import SolveStrategy
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from remat.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from remat.core.solvers.strategy_chen import solve_chen_sqrtn, solve_chen_greedy
from remat.tensorflow2.extraction import dfgraph_from_keras


def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', default="flops", choices=PLATFORM_CHOICES)
    parser.add_argument('--model-name', default="VGG19", choices=list(sorted(MODEL_NAMES)))
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])
    parser.add_argument("--batch-size-min", type=int, default=4)
    parser.add_argument("--batch-size-max", type=int, default=512)
    parser.add_argument("--batch-size-increment", type=int, default=8)

    _args = parser.parse_args()
    _args.input_shape = _args.input_shape if _args.input_shape else None
    return _args


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # due to bug on havoc, limit parallelism on high-core machines
    if os.cpu_count() > 48:
        os.environ["OMP_NUM_THREADS"] = "1"
    args = extract_params()

    key = "_".join(map(str, [args.platform, args.model_name, args.input_shape]))
    log_base = remat_data_dir() / "max_batch_size" / key
    shutil.rmtree(log_base, ignore_errors=True)
    pathlib.Path(log_base).mkdir(parents=True, exist_ok=True)
    result_dict: Dict[int, ScheduledResult] = defaultdict()
    model_name = args.model_name

    # load costs, and plot optionally, if platform is not flops
    logging.info(f"Loading costs")
    if args.platform == "flops":
        cost_model = None
    else:
        cost_model = CostModel(model_name, args.platform, log_base, quantization=5)
        cost_model.fit()
        cost_model.plot_costs()

    model = get_keras_model(model_name, input_shape=args.input_shape)
    tf.keras.utils.plot_model(model, to_file=log_base / f"plot_{model_name}.png",
                              show_shapes=True, show_layer_names=True)

    platform_ram = platform_memory("p32xlarge")
    #platform_ram = 12 * 1000 * 1000 * 1000
    bs_param_ram_cost: Dict[int, int] = {}
    bs_fwd2xcost: Dict[int, int] = {}
    #bs_list = [(i * 4) for i in range(1, 129)] # 4 ~ 512 
    bs_list = [i for i in range(4, 512)] # more fine-grained experiment

    max_batch_size = 0
    for index, bs in enumerate(bs_list):

        # load model at batch size
        g = dfgraph_from_keras(model, batch_size=bs, cost_model=cost_model, loss_cpu_cost=0, loss_ram_cost=(4 * bs))
        bs_fwd2xcost[index] = sum(g.cost_cpu_fwd.values()) + sum(g.cost_cpu.values())
        bs_param_ram_cost[index] = g.cost_ram_fixed
        render_dfgraph(g, log_base, name=model_name)

        # checkpoint all
        #result_dict[bs][SolveStrategy.CHECKPOINT_ALL] = [solve_checkpoint_all(g)]
        result_dict[index] = solve_checkpoint_all(g)

        result = result_dict[index]
        if (result.schedule_aux_data is not None \
            and result.schedule_aux_data.peak_ram <= platform_ram - bs_param_ram_cost[index] \
            and result.schedule_aux_data.cpu <= bs_fwd2xcost[index]):
            max_batch_size = bs
            logging.info(f"Checkpoint-All succeeded at batch size {bs}")
        else:
            break

    logging.info(f"Max batch size = {int(max_batch_size)}")
