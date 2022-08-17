import tensorflow as tf
import numpy as np
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


def construct_input_pipeline(
    dataset,
    batch_size,
    selected_atom_descriptors,
    selected_bond_descriptors,
    selected_reaction_descriptors,
    shuffle=True,
    predict=False,
):
    """Generates a tf.data.Dataset object based on a graph_dataloader,
    which can be used as input for training/evaluation of the GNN model.

    Note: This is a hack to get around the issues related to having a
    Graph_dataloader in combination with multiprocessing in tensorflow.

    Args:
        dataset (utils.Dataset): a dataset object
        batch_size (int): the batch size
        selected_atom_descriptors (List[str]): the selected atom descriptors
        selected_bond_descriptors (List[str]): the selected bond descriptors
        selected_reaction_descriptors (List[str]): the selected reaction descriptors
        shuffle (bool, optional): whether the data should be shuffled. Defaults to True.
        predict (bool, optional): whether this is a prediction. Defaults to False.

    Returns:
        tf_dataset (tf.data.Dataset) : the input pipeline
        dataloader[0] (Tuple[tf.Tensor]): example input to set up loaded model
    """

    dataloader = Graph_DataLoader(
        dataset,
        batch_size,
        selected_atom_descriptors,
        selected_bond_descriptors,
        selected_reaction_descriptors,
        shuffle=shuffle,
        predict=predict,
    )

    def gen_data():
        for i in range(len(dataloader)):
            yield dataloader.getitem(i)

    # Since the last dimension of the input to a dense layer needs to be defined,
    # models with 'reaction descriptors' need to be treated separately since a dense layer
    # directly follows its concatenation in the GNN model.
    if predict:
        if "none" in selected_reaction_descriptors:
            tf_dataset = tf.data.Dataset.from_generator(
                gen_data,
                output_signature=(
                    tf.TensorSpec(shape=(None, None, 36), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, None, 11), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 36), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                ),
            )
        else:
            tf_dataset = tf.data.Dataset.from_generator(
                gen_data,
                output_signature=(
                    tf.TensorSpec(shape=(None, None, 36), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, None, 11), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                    tf.TensorSpec(
                        shape=(None, len(selected_reaction_descriptors)),
                        dtype=tf.float32,
                    ),
                    tf.TensorSpec(shape=(None, None, 36), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    ),
                )
    else:
        if "none" in selected_reaction_descriptors:
            tf_dataset = tf.data.Dataset.from_generator(
                gen_data,
                output_signature=(
                    (
                        tf.TensorSpec(shape=(None, None, 36), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, None, 11), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 36), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    ),
                    {
                        "activation_energy": tf.TensorSpec(
                            shape=(None, 1), dtype=tf.float32
                        ),
                        "reaction_energy": tf.TensorSpec(
                            shape=(None, 1), dtype=tf.float32
                        ),
                    },
                ),
            )
        else:
            tf_dataset = tf.data.Dataset.from_generator(
                gen_data,
                output_signature=(
                    (
                        tf.TensorSpec(shape=(None, None, 36), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, None, 11), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                        tf.TensorSpec(
                            shape=(None, len(selected_reaction_descriptors)),
                            dtype=tf.float32,
                        ),
                        tf.TensorSpec(shape=(None, None, 36), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None, 10, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    ),
                    {
                        "activation_energy": tf.TensorSpec(
                            shape=(None, 1), dtype=tf.float32
                        ),
                        "reaction_energy": tf.TensorSpec(
                            shape=(None, 1), dtype=tf.float32
                        ),
                    },
                ),
            )

    if predict:
        return tf_dataset.prefetch(100), dataloader[0]
    else:
        if shuffle:
            tf_dataset.shuffle(len(dataset))
        return tf_dataset.cache(), dataloader[0][0]


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

    def getitem(self, index):
        return self.__getitem__(index)

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
