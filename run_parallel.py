import os

cross_val_dir = "cross_val"

dataset = "cycloaddition_attempt2.csv"

atom_desc_file = "atom_desc_cycloadd_wln.pkl"

reaction_desc_file = "reaction_desc_cycloadd_wln.pkl"


def run_experiments_with_sample(partition_scheme, atom_desc_file, reaction_desc_file, sample):
    os.makedirs("log_test/{}".format(partition_scheme), exist_ok=True)
    log_dir = "log_test/{}".format(partition_scheme)
    os.makedirs(cross_val_dir + "/" + partition_scheme, exist_ok=True)

    experiments = [
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/GNN'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', 'none',
            '--select_reaction_descriptors', 'none', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample, 
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/only_atomic_desc'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'none', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/only_reaction_desc'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "none",
            '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file), '--reaction_desc_path',
            'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_full'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--atom_desc_path',
            'descriptors/{}'.format(atom_desc_file), '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file),
            '--k_fold', "5", "--sample", sample, "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/trad'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "fukui_elec", "fukui_neu", '--atom_desc_path',
            'descriptors/{}'.format(atom_desc_file), '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file),
            '--k_fold', "5", "--sample", sample, "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_hard'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_soft'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "spin_dens",
            "spin_dens_triplet", "fukui_elec", "fukui_neu", '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_GRP'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'G', 'E_r', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_RP'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'E_r', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_G_alt1'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'G_alt1', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_G_alt2'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'G_alt2', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/none_RP'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "none",
            '--select_reaction_descriptors', 'E_r', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_hard_morfeus'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", 'sasa', 'pint', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_morfeus'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', 'sasa', 'pint', 
            '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--sample", sample,
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_full_morfeus'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", 'sasa', 'pint', '--atom_desc_path',
            'descriptors/{}'.format(atom_desc_file), '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file),
            '--k_fold', "5", "--sample", sample, "--select_bond_descriptors", "none"]
        ]

    launch_jobs(experiments, log_dir)


def run_experiments_no_sample(partition_scheme, atom_desc_file, reaction_desc_file):
    os.makedirs("log_test/{}".format(partition_scheme), exist_ok=True)
    log_dir = "log_test/{}".format(partition_scheme)
    os.makedirs(cross_val_dir + "/" + partition_scheme, exist_ok=True)

    experiments = [
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/GNN'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', 'none',
            '--select_reaction_descriptors', 'none', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/only_atomic_desc'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'none', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/only_reaction_desc'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "none",
            '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file), '--reaction_desc_path',
            'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_full'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--atom_desc_path',
            'descriptors/{}'.format(atom_desc_file), '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file),
            '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/trad'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "fukui_elec", "fukui_neu", '--atom_desc_path',
            'descriptors/{}'.format(atom_desc_file), '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file),
            '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_hard'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_soft'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "spin_dens",
            "spin_dens_triplet", "fukui_elec", "fukui_neu", '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_GRP'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'G', 'E_r', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_RP'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'E_r', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_G_alt1'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'G_alt1', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_G_alt2'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'G_alt2', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/none_RP'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "none",
            '--select_reaction_descriptors', 'E_r', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5", "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_hard_morfeus'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", 'sasa', 'pint', '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5",
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_morfeus'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', 'sasa', 'pint', 
            '--atom_desc_path', 'descriptors/{}'.format(atom_desc_file),
            '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file), '--k_fold', "5",
            "--select_bond_descriptors", "none"],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_full_morfeus'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", 'sasa', 'pint', '--atom_desc_path',
            'descriptors/{}'.format(atom_desc_file), '--reaction_desc_path', 'descriptors/{}'.format(reaction_desc_file),
            '--k_fold', "5", "--select_bond_descriptors", "none"]
        ]

    launch_jobs(experiments, log_dir)


def launch_jobs(experiments, log_dir):
    for experiment in experiments:
        with open("generic_slurm.sh", "w") as f:
            f.write("#!/bin/bash \n")
            f.write("#SBATCH -N 1 \n")
            f.write("#SBATCH -n 16 \n")
            f.write("#SBATCH --time=11:59:00 \n")
            f.write("#SBATCH --gres=gpu:1 \n")
            f.write("#SBATCH --constraint=centos7 \n")
            f.write("#SBATCH --partition=sched_mit_ccoley \n")
            f.write("#SBATCH --mem 32000 \n")
            f.write("#SBATCH --output=" + log_dir + "/" + experiment[7].split("/")[-1] + ".out \n")
            f.write("source /home/tstuyver/.bashrc \n")
            f.write("conda activate tf_gpu \n \n")

            command = ' '.join(experiment)
            print(command)

            f.write(command)
            f.close()

            os.system("sbatch generic_slurm.sh")


def main():

    os.makedirs(cross_val_dir, exist_ok=True)
    os.makedirs("log_test", exist_ok=True)

    run_experiments_with_sample('100_points', atom_desc_file, reaction_desc_file, sample=str(100))
    run_experiments_with_sample('400_points', atom_desc_file, reaction_desc_file, sample=str(400))
    run_experiments_no_sample('all_points', atom_desc_file, reaction_desc_file)


main()
