import os

cross_val_dir = "cross_val"

train_valid_set_file = "train_valid_set_dipole_split.csv"
test_set_file = "test_set_dipole_split.csv"

atom_desc_file = "atom_desc_cycloadd_wln.pkl"
reaction_desc_file = "reaction_desc_cycloadd_wln.pkl"


def run_experiments(partition_scheme, atom_desc_file, reaction_desc_file, sample=None):
    os.makedirs("log_test/{}".format(partition_scheme), exist_ok=True)
    log_dir = "log_test/{}".format(partition_scheme)
    os.makedirs(cross_val_dir + "/" + partition_scheme, exist_ok=True)

    fixed_command_list = ['python', 'cross_val.py', '-m', 'QM_GNN', '--train_valid_set_path', f'datasets_selective_sampling/{train_valid_set_file}', 
            '--test_set_path', f'datasets_selective_sampling/{test_set_file}', '--atom_desc_path', f'descriptors/{atom_desc_file}',
            '--reaction_desc_path', f'descriptors/{reaction_desc_file}', '--k_fold', '5', '--select_bond_descriptors', 'none']

    experiments = [
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/GNN', '--select_atom_descriptors', 'none', '--select_reaction_descriptors', 'none'],
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/only_atomic_desc', '--select_atom_descriptors', 'nmr',
            'partial_charge', 'spin_dens', 'spin_dens_triplet', 'fukui_elec', 'fukui_neu', '--select_reaction_descriptors', 'none'],
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/only_reaction_desc', '--select_atom_descriptors', 'none'],
        ['--model_dir',  f'{cross_val_dir}/{partition_scheme}/all_full', '--select_atom_descriptors', 'nmr',
            'partial_charge', 'spin_dens', 'spin_dens_triplet', 'fukui_elec', 'fukui_neu'],
        ['--model_dir',  f'{cross_val_dir}/{partition_scheme}/trad', '--select_atom_descriptors', 'nmr',
            'partial_charge', 'fukui_elec', 'fukui_neu'],
        ['--model_dir',  f'{cross_val_dir}/{partition_scheme}/all_hard', '--select_atom_descriptors', 'nmr', 'partial_charge'],
        ['--model_dir'f'{cross_val_dir}/{partition_scheme}/all_soft', '--select_atom_descriptors', 'spin_dens',
            'spin_dens_triplet', 'fukui_elec', 'fukui_neu'],
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/GRP_full', '--select_atom_descriptors', 'nmr',
            'partial_charge', 'spin_dens', 'spin_dens_triplet', 'fukui_elec', 'fukui_neu', '--select_reaction_descriptors', 'G', 'E_r',],
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/RP_full', '--select_atom_descriptors', 'nmr',
            'partial_charge', 'spin_dens', 'spin_dens_triplet', 'fukui_elec', 'fukui_neu', '--select_reaction_descriptors', 'E_r'],
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/G_alt1_full', '--select_atom_descriptors', 'nmr',
            'partial_charge', 'spin_dens', 'spin_dens_triplet', 'fukui_elec', 'fukui_neu', '--select_reaction_descriptors', 'G_alt1'],
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/G_alt2_full', '--select_atom_descriptors', 'nmr',
            'partial_charge', 'spin_dens', 'spin_dens_triplet', 'fukui_elec', 'fukui_neu', '--select_reaction_descriptors', 'G_alt2'],
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/none_RP','--select_atom_descriptors', 'none',
            '--select_reaction_descriptors', 'E_r'],
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/all_hard_morfeus', '--select_atom_descriptors', 'nmr', 'partial_charge', 'sasa', 'pint'],
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/all_morfeus', '--select_atom_descriptors', 'sasa', 'pint'],
        ['--model_dir', f'{cross_val_dir}/{partition_scheme}/all_full_morfeus', '--select_atom_descriptors', 'nmr',
            'partial_charge', 'spin_dens', 'spin_dens_triplet', 'fukui_elec', 'fukui_neu', 'sasa', 'pint']
        ]

    command_lines = []
    for experiment in experiments:
        if sample:
            command_lines.append(fixed_command_list + experiment + ['--sample', sample])
        else:
            command_lines.append(fixed_command_list + experiment)

    launch_jobs(command_lines, log_dir)


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
            f.write("#SBATCH --output=" + log_dir + "/" + experiment[17].split("/")[-1] + ".out \n")
            f.write("source /home/tstuyver/.bashrc \n")
            f.write("conda activate tf_gpu \n \n")

            command = ' '.join(experiment)
            print(command)

            f.write(command)
            f.close()

            os.system("sbatch generic_slurm.sh")


if __name__ == '__main__':    
    os.makedirs(cross_val_dir, exist_ok=True)
    os.makedirs("log_test", exist_ok=True)

    run_experiments('100_points', atom_desc_file, reaction_desc_file, sample=str(100))
    run_experiments('400_points', atom_desc_file, reaction_desc_file, sample=str(400))
    run_experiments('all_points', atom_desc_file, reaction_desc_file)
