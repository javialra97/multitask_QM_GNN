import os

name = "amide_double_split"

cross_val_dir = f"selective_sampling/cross_val_selective/cross_val_{name}"

train_valid_set_file = f"train_valid_set_{name}.csv"

test_set_file = f"test_set_{name}.csv"

atom_desc_file = "atom_desc_final_wln.pkl"

reaction_desc_file = "reaction_desc_final_wln.pkl"

log_dir_head = f"selective_sampling/log_test_{name}"


def run_experiments(partition_scheme, atom_desc_file, reaction_desc_file, sample=None):
    os.makedirs(f"{log_dir_head}/{partition_scheme}", exist_ok=True)
    log_dir = f"{log_dir_head}/{partition_scheme}"
    os.makedirs(f"{cross_val_dir}/{partition_scheme}", exist_ok=True)

    fixed_command_list = [
        "python",
        "cross_val.py",
        "--train_valid_set_path",
        f"../datasets_selective_sampling_final/{train_valid_set_file}",
        "--test_set_path",
        f"../datasets_selective_sampling_final/{test_set_file}",
        "--atom_desc_path",
        f"../descriptors_final/{atom_desc_file}",
        "--reaction_desc_path",
        f"../descriptors_final/{reaction_desc_file}",
        "--k_fold",
        "5",
        "--select_bond_descriptors",
        "none",
        "--depth", "2",
        "--ini_lr", "0.00165",
        "--lr_ratio", "0.93",
        "--w_atom", "0.5",
        "--w_reaction", "0.3",
        "--hidden_size_multiplier", "0",
        "--depth_mol_ffn", "1",
        "--random_state", "2",
    ]

    experiments = [
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/GNN",
            "--select_atom_descriptors",
            "none",
            "--select_reaction_descriptors",
            "none",
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/only_atomic_desc",
            "--select_atom_descriptors",
            "nmr",
            "partial_charge",
            "fukui_elec",
            "fukui_neu",
            "--select_reaction_descriptors",
            "none",
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/only_reaction_desc",
            "--select_atom_descriptors",
            "none",
        ],
	[
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/trad_desc",
            "--select_atom_descriptors",
            "nmr",
            "partial_charge",
            "fukui_elec",
            "fukui_neu",
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/all_desc",
            "--select_atom_descriptors",
            "nmr",
            "partial_charge",
            "spin_dens_triplet",
            "fukui_elec",
            "fukui_neu",
        ],

    ]

    command_lines = []
    for experiment in experiments:
        if sample:
            command_lines.append(fixed_command_list + experiment + ["--sample", sample])
        else:
            command_lines.append(fixed_command_list + experiment)

    launch_jobs(command_lines, log_dir)


def launch_jobs(experiments, log_dir):
    for experiment in experiments:
        with open("generic_slurm.sh", "w") as f:
            f.write("#!/bin/bash \n")
            f.write("#SBATCH -N 1 \n")
            f.write("#SBATCH -n 6 \n")
            f.write("#SBATCH --time=11:59:00 \n")
            f.write("#SBATCH --gres=gpu:1 \n")
            f.write("#SBATCH --constraint=centos7 \n")
            f.write("#SBATCH --partition=sched_mit_ccoley \n")
            f.write("#SBATCH --mem 32000 \n")
            f.write(
                f"#SBATCH --output={log_dir}/{experiment[31].split('/')[-1]}.out \n"
            )
            f.write("source /home/tstuyver/.bashrc \n")
            f.write("conda activate tf_gpu \n \n")

            command = " ".join(experiment)
            print(command)

            f.write(command)
            f.close()

            os.system("sbatch generic_slurm.sh")


if __name__ == "__main__":
    os.makedirs(cross_val_dir, exist_ok=True)
    os.makedirs(log_dir_head, exist_ok=True)

    run_experiments("all_points", atom_desc_file, reaction_desc_file)


