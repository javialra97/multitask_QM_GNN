import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from .layers import WLN_Layer, Global_Attention
import numpy as np

np.set_printoptions(threshold=np.inf)


class WLNRegressor(tf.keras.Model):

    def __init__(self, hidden_size, depth, selected_atom_descriptors, selected_reaction_descriptors, w_atom, w_reaction, max_nb=10):
        super(WLNRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.reactants_WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.selected_atom_descriptors = selected_atom_descriptors
        self.selected_reaction_descriptors = selected_reaction_descriptors
        self.w_atom = tf.Variable(self.w_atom, trainable=False)
        self.w_reaction = tf.Variable(self.w_reaction, trainable=False)

        if "none" in self.selected_atom_descriptors:
            self.reaction_score0 = layers.Dense(hidden_size, activation=K.relu,
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                use_bias=False)
            self.attention = Global_Attention(hidden_size)
        else:
            self.reaction_score0 = layers.Dense(hidden_size + len(self.selected_atom_descriptors) * 20, activation=K.relu,
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                use_bias=False)
            self.attention = Global_Attention(hidden_size + len(self.selected_atom_descriptors) * 20)

        if "none" in self.selected_reaction_descriptors:
            self.mol_layer1 = layers.Dense(hidden_size, activation=K.relu,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                            use_bias=False)
            self.mol_layer2 = layers.Dense(hidden_size, activation=K.relu,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                            use_bias=False)
        else:
            self.mol_layer1 = layers.Dense(hidden_size + len(self.selected_reaction_descriptors), activation=K.relu,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                            use_bias=False)
            self.mol_layer2 = layers.Dense(hidden_size + len(self.selected_reaction_descriptors), activation=K.relu,
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                           use_bias=False)

        self.reaction_score = layers.Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=0.1))

        self.node_reshape = layers.Reshape((-1, 1))
        self.core_reshape = layers.Reshape((-1, 1))

    def call(self, inputs):
        res_inputs = inputs[:8]

        res_atom_mask = res_inputs[-3]

        res_core_mask = res_inputs[-1]

        fatom_qm = inputs[-2]
        freaction_qm = inputs[-1]

        res_atom_hidden = self.reactants_WLN(res_inputs)
        if "none" not in self.selected_atom_descriptors:
            res_atom_hidden = K.concatenate([res_atom_hidden, self.w_atom * fatom_qm], axis=-1)
        res_atom_mask = self.node_reshape(res_atom_mask)
        res_core_mask = self.core_reshape(res_core_mask)
        res_att_context, _ = self.attention(res_atom_hidden, res_inputs[-2])
        res_atom_hidden = res_atom_hidden + res_att_context
        res_atom_hidden = self.reaction_score0(res_atom_hidden)
        res_mol_hidden = K.sum(res_atom_hidden * res_atom_mask * res_core_mask, axis=-2)
        if "none" not in self.selected_reaction_descriptors:
            res_mol_hidden = K.concatenate([res_mol_hidden, self.w_reaction * freaction_qm], axis=-1)
        res_mol_hidden = self.mol_layer1(res_mol_hidden)
        res_mol_hidden = self.mol_layer2(res_mol_hidden)
        reaction_score = self.reaction_score(res_mol_hidden)

        return reaction_score
