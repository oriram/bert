import argparse
import glob
import os
import re

import wandb


def parse_log_line(line):
    split = re.split(" ", line)
    loss = float(split[2].replace(",", ""))  # Remove ',' right after the loss value
    step_index = split.index("step") + 2
    step = int(split[step_index])
    return loss, step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_pattern", "-l", required=True, type=str)
    parser.add_argument("--wandb_project", "-p", type=str, default="splinter")

    args = parser.parse_args()

    log_files = glob.glob(args.log_pattern)
    for file_path in log_files:
        print("*" * 20)
        print(f"Processing {file_path}")
        print("*" * 20)

        run_name = os.path.basename(file_path).replace(".log", "").replace(".txt", "")
        run = wandb.init(project=args.wandb_project, name=run_name)

        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("INFO") and "loss" in line and "step" in line:
                loss, step = parse_log_line(line)
                run.log(data={"loss": loss}, step=step)

        run.finish()