# import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from random import shuffle
from ..graph_utils.mol_graph import (
    smiles2graph_pr,
    pack1D,
    pack2D,
    pack2D_withidx,
    get_mask,
)
from ..graph_utils.ioutils_direct import binary_features_batch


class Graph_DataLoader(Sequence):
    def __init__(
        self,
        dataset,
        batch_size,
        selected_atom_descriptors,
        selected_bond_descriptors,
        selected_reaction_descriptors,
        shuffle=True,
        predict=False,
    ):
        self.smiles = dataset.reactant_smiles
        self.product = dataset.product_smiles
        self.rxn_id = dataset.rxn_id
        self.activation_energy = dataset.activation_energy_scaled
        self.reaction_energy = dataset.reaction_energy_scaled
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.predict = predict
        self.selected_atom_descriptors = selected_atom_descriptors
        self.selected_reaction_descriptors = selected_reaction_descriptors
        self.selected_bond_descriptors = selected_bond_descriptors

        if self.predict:
            self.shuffle = False

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.smiles) / self.batch_size))

    def __getitem__(self, index):
        smiles_tmp = self.smiles[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        product_tmp = self.product[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        rxn_id_tmp = self.rxn_id[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        if not self.predict:
            activation_energy_tmp = self.activation_energy[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            reaction_energy_tmp = self.reaction_energy[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y = self.__data_generation(
                smiles_tmp,
                product_tmp,
                rxn_id_tmp,
                activation_energy_tmp,
                reaction_energy_tmp,
            )
            return x, y
        else:
            x = self.__data_generation(smiles_tmp, product_tmp, rxn_id_tmp)
            return x

    def on_epoch_end(self):
        if self.shuffle == True:
            zipped = list(
                zip(
                    self.smiles,
                    self.product,
                    self.rxn_id,
                    self.activation_energy,
                    self.reaction_energy,
                )
            )
            shuffle(zipped)
            (
                self.smiles,
                self.product,
                self.rxn_id,
                self.activation_energy,
                self.reaction_energy,
            ) = zip(*zipped)

    def __data_generation(
        self,
        smiles_tmp,
        product_tmp,
        rxn_id_tmp,
        activation_energy_tmp=None,
        reaction_energy_tmp=None,
    ):
        prs_extend = []
        rxn_id_extend = []

        if not self.predict:
            activation_energy_extend = []
            reaction_energy_extend = []
            for r, p, rxn_id, activation_energy, reaction_energy in zip(
                smiles_tmp,
                product_tmp,
                rxn_id_tmp,
                activation_energy_tmp,
                reaction_energy_tmp,
            ):
                rxn_id_extend.extend([rxn_id])
                prs_extend.extend(
                    [
                        smiles2graph_pr(
                            r,
                            p,
                            self.selected_atom_descriptors,
                            self.selected_bond_descriptors,
                            self.selected_reaction_descriptors,
                        )
                    ]
                )
                activation_energy_extend.extend([activation_energy])
                reaction_energy_extend.extend([reaction_energy])
        else:
            for r, p, rxn_id in zip(smiles_tmp, product_tmp, rxn_id_tmp):
                rxn_id_extend.extend([rxn_id])
                prs_extend.extend(
                    [
                        smiles2graph_pr(
                            r,
                            p,
                            self.selected_atom_descriptors,
                            self.selected_bond_descriptors,
                            self.selected_reaction_descriptors,
                        )
                    ]
                )

        prs_extends, smiles_extend = zip(*prs_extend)

        (
            fatom_list,
            fatom_qm_list,
            fbond_list,
            gatom_list,
            gbond_list,
            nb_list,
            core_mask,
            freaction_qm,
            fatom_list_p,
            fbond_list_p,
            gatom_list_p,
            gbond_list_p,
            nb_list_p,
        ) = zip(*prs_extends)

        res_graph_inputs = (
            pack2D(fatom_list),
            pack2D(fbond_list),
            pack2D_withidx(gatom_list),
            pack2D_withidx(gbond_list),
            pack1D(nb_list),
            get_mask(fatom_list),
            binary_features_batch(smiles_extend),
            pack1D(core_mask),
            pack2D(fatom_qm_list),
            pack1D(freaction_qm),
            pack2D(fatom_list_p),
            pack2D(fbond_list_p),
            pack2D_withidx(gatom_list_p),
            pack2D_withidx(gbond_list_p),
            pack1D(nb_list_p),
            get_mask(fatom_list_p),
        )
        if self.predict:
            return res_graph_inputs
        else:
            return res_graph_inputs, {
                "activation_energy": np.array(activation_energy_extend).astype("float"),
                "reaction_energy": np.array(reaction_energy_extend).astype("float"),
            }
