import os, random, argparse
import numpy as np
import pandas as pd
from ary import *
from utils import *


def main(args):
    if not os.path.exists(args.tmp_path):
        os.makedirs(args.tmp_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if len(args.data) == 1:
        assert args.data in "samukorea"
        env = np.array(PD_ENV[f"{args.data}{args.size}"]).astype(np.float32)
        save_name = args.data
    else:
        env = pd.read_csv(args.data, header=None).values.astype(np.float32)
        save_name = args.data.split(".")[1][1:]

    print("Initializing...")
    solver = Solver(env, "auto", args.step)
    solver.train()
    solution, _ = solver.get_routes(0)

    pd.DataFrame(solution).to_csv(
        os.path.join(args.save_path, save_name+".csv"),
        index=False, header=None
    )

    save_images_rgb(env, solution, args.tmp_path)
    save_gif(
        png_path=args.tmp_path,
        save_path=os.path.join(args.save_path, save_name+".gif")
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="s")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--tmp_path", type=str, default="./tmp")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="./save")

    args = parser.parse_args()
    main(args)
