import os

name = "all"

cross_val_dir = "cross_val"

dataset = f"{name}_data.csv"

atom_desc_file = f"atom_desc_{name}_wln.pkl"

reaction_desc_file = f"reaction_desc_{name}_wln.pkl"

log_dir_head = f"log_test_{name}"


def run_experiments(partition_scheme, atom_desc_file, reaction_desc_file, sample=None):
    os.makedirs(f"{log_dir_head}/{partition_scheme}", exist_ok=True)
    log_dir = f"{log_dir_head}/{partition_scheme}"
    os.makedirs(f"{cross_val_dir}/{partition_scheme}", exist_ok=True)

    fixed_command_list = [
        "python",
        "cross_val.py",
        "--data_path",
        f"datasets/{dataset}",
        "--atom_desc_path",
        f"descriptors/{atom_desc_file}",
        "--reaction_desc_path",
        f"descriptors/{reaction_desc_file}",
        "--k_fold",
        "5",
        "--select_bond_descriptors",
        "none",
        "depth", "3",
        "ini_lr", "0.0009",
        "lr_ratio", "0.98",
        "w_atom", "3.5",
        "w_reaction", "1.5",
        "hidden_size_multiplier", "0",
        "depth_mol_ffn", "2"
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
            "spin_dens",
            "spin_dens_triplet",
            "fukui_elec",
            "fukui_neu",
            "--select_reaction_descriptors",
            "none",
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/only_atomic_desc_no_spin",
            "--select_atom_descriptors",
            "nmr",
            "partial_charge",
            "spin_dens_triplet",
            "fukui_elec",
            "fukui_neu",
            "--select_reaction_descriptors",
            "none",
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/only_atomic_desc_morfeus",
            "--select_atom_descriptors",
            "nmr",
            "partial_charge",
            "spin_dens_triplet",
            "fukui_elec",
            "fukui_neu",
            "sasa",
            "pint",
            "--select_reaction_descriptors",
            "none"
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/only_reaction_desc",
            "--select_atom_descriptors",
            "none",
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/trad",
            "--select_atom_descriptors",
            "nmr",
            "partial_charge",
            "fukui_elec",
            "fukui_neu",
            "--select_reaction_descriptors",
            "none"
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/RP_full",
            "--select_atom_descriptors",
            "nmr",
            "partial_charge",
            "spin_dens",
            "spin_dens_triplet",
            "fukui_elec",
            "fukui_neu",
            "--select_reaction_descriptors",
            "E_r",
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/all_full_morfeus_rp",
            "--select_atom_descriptors",
            "nmr",
            "partial_charge",
            "spin_dens",
            "spin_dens_triplet",
            "fukui_elec",
            "fukui_neu",
            "sasa",
            "pint",
            "--select_reaction_descriptors",
            "E_r",
            "G",
            "G_alt1",
            "G_alt2"
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/react_only",
            "--select_atom_descriptors",
            "nmr",
            "partial_charge",
            "spin_dens_triplet",
            "fukui_elec",
            "fukui_neu",
            "sasa",
            "pint",
            "--select_reaction_descriptors",
            "G",
            "G_alt1",
            "G_alt2",
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/react_only_hard",
            "--select_atom_descriptors",
            "nmr",
            "partial_charge",
            "--select_reaction_descriptors",
            "G",
            "G_alt1",
            "G_alt2",
        ],
        [
            "--model_dir",
            f"{cross_val_dir}/{partition_scheme}/react_only_soft",
            "--select_atom_descriptors",
            "fukui_elec",
            "fukui_neu",
            "--select_reaction_descriptors",
            "G",
            "G_alt1",
            "G_alt2",
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
            f.write("#SBATCH --cpus-per-task=6 \n")
            f.write("#SBATCH --time=11:59:00 \n")
            f.write("#SBATCH --gres=gpu:1 \n")
            f.write("#SBATCH --constraint=centos7 \n")
            f.write("#SBATCH --partition=sched_mit_ccoley \n")
            f.write("#SBATCH --nodelist node1237 \n")
            f.write("#SBATCH --mem 32000 \n")
            f.write(
                f"#SBATCH --output={log_dir}/{experiment[27].split('/')[-1]}.out \n"
            )

            f.write("source /home/tstuyver/.bashrc \n")
            f.write("conda activate tf_gpu \n \n")

            command = " ".join(experiment)
            print(command)

            f.write(command)
            f.close()

            os.system("sbatch generic_slurm.sh")


def main():

    os.makedirs(cross_val_dir, exist_ok=True)
    os.makedirs(log_dir_head, exist_ok=True)

    run_experiments("100_points", atom_desc_file, reaction_desc_file, sample=str(100))
    run_experiments("400_points", atom_desc_file, reaction_desc_file, sample=str(400))
    run_experiments("800_points", atom_desc_file, reaction_desc_file, sample=str(800))
    run_experiments("800_points", atom_desc_file, reaction_desc_file, sample=str(1600))
    run_experiments("all_points", atom_desc_file, reaction_desc_file)


main()
