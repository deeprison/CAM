import os
import random
import argparse
import pandas as pd
from ary import *
from utils import *


def save_results(solution, save_path, save_name, env, tmp_path):
    pd.DataFrame(solution).to_csv(
        os.path.join(save_path, save_name + ".csv"), index=False, header=None
    )
    save_images_rgb(env, solution, tmp_path)
    save_gif(
        png_path=tmp_path,
        save_path=os.path.join(save_path, save_name + ".gif")
    )


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
        save_name = args.data + str(args.size)
    else:
        env = pd.read_csv(args.data, header=None).values.astype(np.float32)
        save_name = args.data.split(".")[1][1:]

    print("Initializing...")
    solver = Solver(env, "auto", args.step)
    solver.train()
    solution, _ = solver.get_routes(0, True)

    if not args.save_all:
        f_name = save_name + f"_step0{args.step}"
        save_results(solution[args.step-1], args.save_path, f_name, env, args.tmp_path)
        print("Done.")
    else:
        if args.step >= 1:
            f_name = save_name + f"_step01"
            save_results(solution[0], args.save_path, f_name, env, args.tmp_path)
            print("Step 1 Done.")
            if args.step >= 2:
                f_name = save_name + f"_step02"
                save_results(solution[1], args.save_path, f_name, env, args.tmp_path)
                print("Step 2 Done.")
                if args.step >= 3:
                    f_name = save_name + f"_step03"
                    save_results(solution[2], args.save_path, f_name, env, args.tmp_path)
                    print("Step 3 Done.")
                    if args.step >= 4:
                        f_name = save_name + f"_step04"
                        save_results(solution[3], args.save_path, f_name, env, args.tmp_path)
                        print("Step 4 Done.")
                        if args.step >= 5:
                            f_name = save_name + f"_step05"
                            save_results(solution[4], args.save_path, f_name, env, args.tmp_path)
                            print("Step 5 Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="s")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--tmp_path", type=str, default="./tmp")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--save_all", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default="./save")

    args = parser.parse_args()
    main(args)
