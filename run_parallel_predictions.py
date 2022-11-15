import os

name = "synthetic_reactions_1"

atom_desc_file = f"atom_desc_{name}_wln.pkl"

reaction_desc_file = f"reaction_desc_{name}_wln.pkl"

predictions_dir = "synthetic_reactions_1"

model_dir = "final_model_1"

log_dir = "output"


def run_predictions(csv_file, predictions_dir, model_dir, log_dir, atom_desc_file, reaction_desc_file):

    command_line_list = [
        "python",
        "reactivity.py",
        "--data_path",
        f"{predictions_dir}/{csv_file}",
        "--model_dir",
        model_dir,
        "--output_file",
        f"{csv_file.split('.csv')[0]}_predicted.csv",
        "--atom_desc_path",
        f"descriptors/{atom_desc_file}",
        "--reaction_desc_path",
        f"descriptors/{reaction_desc_file}",
        "--select_bond_descriptors",
        "none",
        "--depth", "2",
        "--ini_lr", "0.0011",
        "--lr_ratio", "0.9",
        "--w_atom", "2.5",
        "--w_reaction", "4.5",
        "--hidden_size_multiplier", "0",
        "--depth_mol_ffn", "1",
        "--random_state", "0",
        "--ensemble_size", "10",
        "-p"
    ]

    launch_jobs(command_line_list, csv_file, log_dir)


def launch_jobs(experiment, csv_file, log_dir):
    with open("generic_slurm.sh", "w") as f:
        f.write("#!/bin/bash \n")
        f.write("#SBATCH -N 1 \n")
        f.write("#SBATCH -n 8 \n")
        f.write("#SBATCH --time=40:59:00 \n")
        f.write("#SBATCH --gres=gpu:1 \n")
        f.write("#SBATCH --constraint=centos7 \n")
        f.write("#SBATCH --partition=sched_mit_ccoley \n")
        f.write("#SBATCH --mem 32000 \n")
        f.write(
            f"#SBATCH --output={log_dir}/{csv_file.split('.csv')[0]}.out \n"
        )
        f.write("source /home/tstuyver/.bashrc \n")
        f.write("conda activate tf_gpu \n \n")

        command = " ".join(experiment)
        print(command)

        f.write(command)
        f.close()

        os.system("sbatch generic_slurm.sh")


if __name__ == "__main__":
    os.makedirs(log_dir, exist_ok=True)
    csv_files = [file for file in os.listdir(predictions_dir) if file.endswith('.csv')]

    for csv_file in csv_files:
        run_predictions(csv_file, predictions_dir, model_dir, log_dir, atom_desc_file, reaction_desc_file)
