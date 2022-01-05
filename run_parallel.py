import os

cross_val_dir = "cross_val"

dataset = "final_dataset_mol_desc_mos.csv"

sample = "100"


def run_experiments(partition_scheme):
    desc_file = "{}_desc_filtered.pkl".format(partition_scheme)
    os.makedirs("log_test/{}".format(partition_scheme), exist_ok=True)
    log_dir = "log_test/{}".format(partition_scheme)
    os.makedirs(cross_val_dir + "/" + partition_scheme, exist_ok=True)

    experiments = [
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/GNN'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', 'none',
            '--select_reaction_descriptors', 'none', '--atom_desc_path', 'descriptors/hirshfeld_desc_filtered.pkl',
            '--reaction_desc_path', 'datasets/reaction_desc_mos.pickle', '--k_fold', "5", "--sample", sample],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/only_atomic_desc'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'none', '--atom_desc_path', 'descriptors/hirshfeld_desc_filtered.pkl',
            '--reaction_desc_path', 'datasets/reaction_desc_mos.pickle', '--k_fold', "5", "--sample", sample],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/only_reaction_desc'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "none",
            '--atom_desc_path', 'descriptors/hirshfeld_desc_filtered.pkl', '--reaction_desc_path', 'datasets/reaction_desc_mos.pickle',
            '--k_fold', "5", "--sample", sample],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_full'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--atom_desc_path',
            'descriptors/hirshfeld_desc_filtered.pkl', '--reaction_desc_path', 'datasets/reaction_desc_mos.pickle',
            '--k_fold', "5", "--sample", sample],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_hard'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", '--atom_desc_path', 'descriptors/hirshfeld_desc_filtered.pkl',
            '--reaction_desc_path', 'datasets/reaction_desc_mos.pickle', '--k_fold', "5", "--sample", sample],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_soft'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "spin_dens",
            "spin_dens_triplet", "fukui_elec", "fukui_neu", '--atom_desc_path', 'descriptors/hirshfeld_desc_filtered.pkl',
            '--reaction_desc_path', 'datasets/reaction_desc_mos.pickle', '--k_fold', "5", "--sample", sample],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_GRP'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'G', 'DE_RP', '--atom_desc_path', 'descriptors/hirshfeld_desc_filtered.pkl',
            '--reaction_desc_path', 'datasets/reaction_desc_mos.pickle', '--k_fold', "5", "--sample", sample],
        ['python', 'cross_val.py', '-m', 'QM_GNN', '--data_path', 'datasets/{}'.format(dataset), '--model_dir',
            '{}/{}/all_G*'.format(cross_val_dir, partition_scheme), '--select_atom_descriptors', "nmr",
            "partial_charge", "spin_dens", "spin_dens_triplet", "fukui_elec", "fukui_neu", '--select_reaction_descriptors',
            'G*', '--atom_desc_path', 'descriptors/hirshfeld_desc_filtered.pkl',
            '--reaction_desc_path', 'datasets/reaction_desc_mos.pickle', '--k_fold', "5", "--sample", sample]]

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
            if sample is None:
                f.write("#SBATCH --output=" + log_dir + "/" + experiment[7].split("/")[-1] + ".out \n")
            else:
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

    for partition_scheme in ['100_points']:
        run_experiments(partition_scheme)


main()


