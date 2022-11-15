import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from .layers import WLN_Layer, Global_Attention
import numpy as np

np.set_printoptions(threshold=np.inf)


class WLNRegressor(tf.keras.Model):
    def __init__(
        self,
        hidden_size,
        depth,
        selected_atom_descriptors,
        selected_reaction_descriptors,
        w_atom,
        w_reaction,
        depth_mol_ffn=2,
        hidden_size_multiplier=20,
        max_nb=10,
    ):
        super(WLNRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.selected_atom_descriptors = selected_atom_descriptors
        self.selected_reaction_descriptors = selected_reaction_descriptors
        self.w_atom = tf.Variable(w_atom, trainable=False)
        self.w_reaction = tf.Variable(w_reaction, trainable=False)
        self.depth_mol_ffn = depth_mol_ffn
        self.hidden_size_multiplier = hidden_size_multiplier

        self.mol_ffn_list_act = []
        self.mol_ffn_list_react = []

        if "none" in self.selected_atom_descriptors:
            self.reaction_score0 = layers.Dense(
                hidden_size,
                activation=K.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                use_bias=False,
            )
            self.attention = Global_Attention(hidden_size)
        else:
            self.reaction_score0 = layers.Dense(
                hidden_size + len(self.selected_atom_descriptors) * 20,
                activation=K.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                use_bias=False,
            )
            self.attention = Global_Attention(
                hidden_size + len(self.selected_atom_descriptors) * 20
            )

        if "none" in self.selected_reaction_descriptors:
            for depth in range(self.depth_mol_ffn):
                self.mol_ffn_list_act.append(
                    layers.Dense(
                        hidden_size,
                        activation=K.relu,
                        kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                        use_bias=False,
                    )
                )
                self.mol_ffn_list_react.append(
                    layers.Dense(
                        hidden_size,
                        activation=K.relu,
                        kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                        use_bias=False,
                    )
                )
        else:
            for depth in range(self.depth_mol_ffn):
                self.mol_ffn_list_act.append(
                    layers.Dense(
                        hidden_size
                        + len(self.selected_atom_descriptors)
                        * self.hidden_size_multiplier,
                        activation=K.relu,
                        kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                        use_bias=False,
                    )
                )
                self.mol_ffn_list_react.append(
                    layers.Dense(
                        hidden_size
                        + len(self.selected_atom_descriptors)
                        * self.hidden_size_multiplier,
                        activation=K.relu,
                        kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                        use_bias=False,
                    )
                )

        self.reaction_score_act = layers.Dense(
            1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.1),
            name="activation_energy",
        )
        self.reaction_score_react = layers.Dense(
            1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.1),
            name="reaction_energy",
        )

        self.node_reshape = layers.Reshape((-1, 1))
        self.core_reshape = layers.Reshape((-1, 1))

    def call(self, inputs):
        res_inputs = inputs[:8]
        res_atom_mask = res_inputs[-3]
        res_core_mask = res_inputs[-1]

        fatom_qm = inputs[8]
        freaction_qm = inputs[9]

        res_inputs_p = inputs[10:]

        res_atom_hidden_r = self.WLN(res_inputs[:6])
        res_atom_hidden_p = self.WLN(res_inputs_p)
        res_atom_hidden = res_atom_hidden_r - res_atom_hidden_p

        if "none" not in self.selected_atom_descriptors:
            res_atom_hidden = K.concatenate(
                [(tf.math.subtract(1.0, self.w_atom))*res_atom_hidden, self.w_atom * fatom_qm], axis=-1
            )
        res_atom_mask = self.node_reshape(res_atom_mask)
        res_core_mask = self.core_reshape(res_core_mask)
        res_att_context, _ = self.attention(res_atom_hidden, res_inputs[-2])
        res_atom_hidden = res_atom_hidden + res_att_context
        res_atom_hidden = self.reaction_score0(res_atom_hidden)
        res_mol_hidden = K.sum(res_atom_hidden * res_atom_mask * res_core_mask, axis=-2)
        if "none" not in self.selected_reaction_descriptors:
            res_mol_hidden = K.concatenate(
                [(tf.math.subtract(1.0, self.w_reaction))*res_mol_hidden, self.w_reaction * freaction_qm], axis=-1
            )

        res_mol_hidden_act = self.mol_ffn_list_act[0](res_mol_hidden)
        for i in range(1, self.depth_mol_ffn):
            res_mol_hidden_act = self.mol_ffn_list_act[i](res_mol_hidden_act)
        activation_energy = self.reaction_score_act(res_mol_hidden_act)

        res_mol_hidden_react = self.mol_ffn_list_react[0](res_mol_hidden)
        for i in range(1, self.depth_mol_ffn):
            res_mol_hidden_react = self.mol_ffn_list_react[i](res_mol_hidden_react)
        reaction_energy = self.reaction_score_react(res_mol_hidden_react)

        return {
            "activation_energy": activation_energy,
            "reaction_energy": reaction_energy,
        }
