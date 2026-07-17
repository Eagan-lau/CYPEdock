"""TensorFlow PocketMiner encoder compatible with the historical checkpoint.

This is the feature-producing portion of the legacy ``models.py``.  It emits
the 100 scalar channels immediately before the original pocket-classification
head.  Tracked attribute names are intentionally retained for object-checkpoint
compatibility.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LayerNormalization

from .gvp import GVP, GVPDropout, GVPLayerNorm, merge, vs_concat


class MQAModel(Model):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_dim,
        num_layers=3,
        k_neighbors=30,
        dropout=0.1,
        regression=False,
        multiclass=False,
        ablate_aa_type=False,
        ablate_sidechain_vectors=True,
        ablate_rbf=False,
        use_lm=False,
        squeeze_lm=False,
    ):
        super().__init__()
        self.multiclass = multiclass
        self.ablate_aa_type = ablate_aa_type
        self.use_lm = use_lm
        self.squeeze_lm = squeeze_lm

        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features

        self.features = StructuralFeatures(
            node_features,
            edge_features,
            top_k=k_neighbors,
            ablate_sidechain_vectors=ablate_sidechain_vectors,
            ablate_rbf=ablate_rbf,
        )

        if not use_lm:
            self.W_s = Embedding(20, self.hs)
        if use_lm and squeeze_lm:
            self.W_s = Sequential(
                [Dense(100, activation="relu"), Dropout(rate=dropout)]
            )

        self.W_v = GVP(vi=self.nv, vo=self.hv, so=self.hs, nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.ev, so=self.hs, nls=None, nlv=None)
        self.encoder = Encoder(
            hidden_dim, edge_features, num_layers=num_layers, dropout=dropout
        )
        self.W_V_out = GVP(vi=self.hv, vo=0, so=self.hs, nls=None, nlv=None)

        # Retained even though feature extraction returns before this head.  The
        # object must mirror the model associated with the TensorFlow checkpoint.
        if regression:
            self.dense = Sequential(
                [
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    LayerNormalization(),
                    Dense(1, activation=None),
                ]
            )
        elif multiclass:
            self.multiclass = True
            self.dense = Sequential(
                [
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    LayerNormalization(),
                    Dense(3, activation="softmax"),
                ]
            )
        else:
            self.dense = Sequential(
                [
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    Dense(2 * self.hs, activation="relu"),
                    Dropout(rate=dropout),
                    LayerNormalization(),
                    Dense(1, activation="sigmoid"),
                ]
            )

    def call(self, X, S, mask, train=False, res_level=False, ablate_aa_type=False):
        del res_level, ablate_aa_type  # Kept in the public signature for compatibility.
        V, E, E_idx = self.features(X, mask)
        if self.ablate_aa_type:
            h_V = self.W_v(V)
        else:
            if self.use_lm and not self.squeeze_lm:
                h_S = S
            elif self.use_lm and self.squeeze_lm:
                h_S = self.W_s(S)
            else:
                h_S = self.W_s(S)
            V = vs_concat(V, h_S, self.nv, 0)
            h_V = self.W_v(V)

        h_E = self.W_e(E)
        h_V = self.encoder(h_V, h_E, E_idx, mask, train=train)
        return self.W_V_out(h_V)


class Encoder(Model):
    def __init__(self, node_features, edge_features, num_layers=3, dropout=0.1):
        super().__init__()
        self.nv, _ = node_features
        self.ev, _ = edge_features
        self.vglayers = [
            MPNNLayer(self.nv + self.ev, node_features, dropout=dropout)
            for _ in range(num_layers)
        ]

    def call(self, h_V, h_E, E_idx, mask, train=False):
        mask_attend = tf.squeeze(
            gather_nodes(tf.expand_dims(mask, -1), E_idx), -1
        )
        mask_attend = tf.expand_dims(mask, -1) * mask_attend
        for layer in self.vglayers:
            h_M = cat_neighbors_nodes(h_V, h_E, E_idx, self.nv, self.ev)
            h_V = layer(
                h_V,
                h_M,
                mask_V=mask,
                mask_attend=mask_attend,
                train=train,
            )
        return h_V


class MPNNLayer(Model):
    def __init__(self, vec_in, num_hidden, dropout=0.1):
        super().__init__()
        self.num_hidden = num_hidden
        self.vec_in = vec_in
        self.vo, self.so = num_hidden
        self.norm = [GVPLayerNorm(self.vo) for _ in range(2)]
        self.dropout = GVPDropout(dropout, self.vo)
        self.W_EV = Sequential(
            [
                GVP(vi=vec_in + self.vo, vo=self.vo, so=self.so),
                GVP(vi=self.vo, vo=self.vo, so=self.so),
                GVP(vi=self.vo, vo=self.vo, so=self.so, nls=None, nlv=None),
            ]
        )
        self.W_dh = Sequential(
            [
                GVP(vi=self.vo, vo=2 * self.vo, so=4 * self.so),
                GVP(vi=2 * self.vo, vo=self.vo, so=self.so, nls=None, nlv=None),
            ]
        )

    def call(self, h_V, h_M, mask_V=None, mask_attend=None, train=False):
        h_V_expand = tf.tile(
            tf.expand_dims(h_V, -2), [1, 1, tf.shape(h_M)[-2], 1]
        )
        h_EV = vs_concat(h_V_expand, h_M, self.vo, self.vec_in)
        h_message = self.W_EV(h_EV)
        if mask_attend is not None:
            h_message = (
                tf.cast(tf.expand_dims(mask_attend, -1), tf.float32) * h_message
            )
        dh = tf.math.reduce_mean(h_message, -2)
        h_V = self.norm[0](h_V + self.dropout(dh, training=train))

        dh = self.W_dh(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh, training=train))
        if mask_V is not None:
            h_V = tf.cast(tf.expand_dims(mask_V, -1), tf.float32) * h_V
        return h_V


class PositionalEncodings(Model):
    def __init__(self, num_embeddings, period_range=(2, 1000)):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def call(self, E_idx):
        n_nodes = tf.shape(E_idx)[1]
        ii = tf.reshape(tf.cast(tf.range(n_nodes), tf.float32), (1, -1, 1))
        d = tf.expand_dims(tf.cast(E_idx, tf.float32) - ii, -1)
        frequency = tf.math.exp(
            tf.cast(tf.range(0, self.num_embeddings, 2), tf.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = d * tf.reshape(frequency, (1, 1, 1, -1))
        return tf.concat((tf.math.cos(angles), tf.math.sin(angles)), -1)


class StructuralFeatures(Model):
    def __init__(
        self,
        node_features,
        edge_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        ablate_sidechain_vectors=True,
        ablate_rbf=False,
    ):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.embeddings = PositionalEncodings(num_positional_embeddings)

        vo, so = node_features
        ve, se = edge_features
        vi_v = 3 if ablate_sidechain_vectors else 4
        self.node_embedding = GVP(vi=vi_v, vo=vo, so=so, nlv=None, nls=None)
        vi_e = 1 if ablate_sidechain_vectors else 2
        self.edge_embedding = GVP(vi=vi_e, vo=ve, so=se, nlv=None, nls=None)
        self.norm_nodes = LayerNormalization()
        self.norm_edges = LayerNormalization()
        self.ablate_sidechain_vectors = ablate_sidechain_vectors
        self.ablate_rbf = ablate_rbf

    def _dist(self, X, mask, eps=1e-6):
        mask = tf.cast(mask, tf.float32)
        mask_2d = tf.expand_dims(mask, 1) * tf.expand_dims(mask, 2)
        dX = tf.expand_dims(X, 1) - tf.expand_dims(X, 2)
        distances = mask_2d * tf.math.sqrt(tf.math.reduce_sum(dX**2, 3) + eps)
        distance_max = tf.math.reduce_max(distances, -1, keepdims=True)
        adjusted = distances + (1.0 - mask_2d) * distance_max
        k = tf.minimum(tf.cast(self.top_k, tf.int32), tf.shape(X)[1])
        neighbor_distances, E_idx = tf.math.top_k(-adjusted, k=k)
        neighbor_distances = -neighbor_distances
        mask_neighbors = gather_edges(tf.expand_dims(mask_2d, -1), E_idx)
        return neighbor_distances, E_idx, mask_neighbors

    @staticmethod
    def _directions(X, E_idx):
        neighbors = gather_nodes(X, E_idx)
        return normalize(neighbors - tf.expand_dims(X, -2), axis=-1)

    @staticmethod
    def _terminal_sidechain_direction(X, E_idx):
        ca, terminal = X[:, :, 1, :], X[:, :, 4, :]
        neighbors = gather_nodes(terminal, E_idx)
        return normalize(neighbors - tf.expand_dims(ca, -2))

    def _rbf(self, distances):
        distance_min, distance_max = 0.0, 20.0
        centers = tf.linspace(distance_min, distance_max, self.num_rbf)
        centers = tf.reshape(centers, [1, 1, 1, -1])
        sigma = (distance_max - distance_min) / self.num_rbf
        return tf.math.exp(-((tf.expand_dims(distances, -1) - centers) / sigma) ** 2)

    @staticmethod
    def _orientations(X):
        forward = normalize(X[:, 1:] - X[:, :-1])
        backward = normalize(X[:, :-1] - X[:, 1:])
        forward = tf.pad(forward, [[0, 0], [0, 1], [0, 0]])
        backward = tf.pad(backward, [[0, 0], [1, 0], [0, 0]])
        return tf.concat(
            [tf.expand_dims(forward, -1), tf.expand_dims(backward, -1)], -1
        )

    @staticmethod
    def _sidechains(X):
        n, origin, c = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
        c, n = normalize(c - origin), normalize(n - origin)
        bisector = normalize(c + n)
        perpendicular = normalize(tf.linalg.cross(c, n))
        return (
            -bisector * tf.math.sqrt(1.0 / 3.0)
            - perpendicular * tf.math.sqrt(2.0 / 3.0)
        )

    @staticmethod
    def _sidechain_terminal_vector(X):
        ca, terminal = X[:, :, 1, :], X[:, :, 4, :]
        return normalize(terminal - ca)

    @staticmethod
    def _dihedrals(X, eps=1e-7):
        length = tf.shape(X)[1]
        backbone = tf.reshape(X[:, :, :3, :], [tf.shape(X)[0], 3 * length, 3])
        differences = backbone[:, 1:, :] - backbone[:, :-1, :]
        unit = normalize(differences, axis=-1)
        u_2, u_1, u_0 = unit[:, :-2, :], unit[:, 1:-1, :], unit[:, 2:, :]
        n_2 = normalize(tf.linalg.cross(u_2, u_1), axis=-1)
        n_1 = normalize(tf.linalg.cross(u_1, u_0), axis=-1)
        cosine = tf.math.reduce_sum(n_2 * n_1, -1)
        cosine = tf.clip_by_value(cosine, -1.0 + eps, 1.0 - eps)
        angles = (
            tf.math.sign(tf.math.reduce_sum(u_2 * n_1, -1))
            * tf.math.acos(cosine)
        )
        angles = tf.pad(angles, [[0, 0], [1, 2]])
        angles = tf.reshape(angles, [tf.shape(angles)[0], length, 3])
        return tf.concat([tf.math.cos(angles), tf.math.sin(angles)], 2)

    def call(self, X, mask):
        X_ca = X[:, :, 1, :]
        distances, E_idx, _ = self._dist(X_ca, mask)
        edge_directions = self._directions(X_ca, E_idx)
        radial_basis = self._rbf(distances)
        positional = self.embeddings(E_idx)

        if not self.ablate_sidechain_vectors:
            edge_sidechain = self._terminal_sidechain_direction(X, E_idx)

        dihedrals = self._dihedrals(X)
        orientations = self._orientations(X_ca)
        sidechains = self._sidechains(X)
        if self.ablate_sidechain_vectors:
            node_vectors = tf.concat(
                [tf.expand_dims(sidechains, -1), orientations], -1
            )
        else:
            terminal = self._sidechain_terminal_vector(X)
            node_vectors = tf.concat(
                [
                    tf.expand_dims(sidechains, -1),
                    tf.expand_dims(terminal, -1),
                    orientations,
                ],
                -1,
            )
        V = merge(node_vectors, dihedrals)

        if self.ablate_sidechain_vectors:
            E = (
                tf.concat([edge_directions, positional], -1)
                if self.ablate_rbf
                else tf.concat([edge_directions, radial_basis, positional], -1)
            )
        else:
            E = (
                tf.concat([edge_directions, edge_sidechain, positional], -1)
                if self.ablate_rbf
                else tf.concat(
                    [edge_directions, edge_sidechain, radial_basis, positional], -1
                )
            )

        node_vectors, node_scalars = self.node_embedding(V, return_split=True)
        V = merge(node_vectors, self.norm_nodes(node_scalars))
        edge_vectors, edge_scalars = self.edge_embedding(E, return_split=True)
        E = merge(edge_vectors, self.norm_edges(edge_scalars))
        return V, E, E_idx


def normalize(tensor, axis=-1):
    return tf.math.divide_no_nan(
        tensor, tf.linalg.norm(tensor, axis=axis, keepdims=True)
    )


def gather_edges(edges, neighbor_idx):
    return tf.gather(edges, neighbor_idx, axis=2, batch_dims=2)


def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = tf.reshape(
        neighbor_idx, [tf.shape(neighbor_idx)[0], -1]
    )
    features = tf.gather(nodes, neighbors_flat, axis=1, batch_dims=1)
    output_shape = tf.concat(
        [tf.shape(neighbor_idx), [tf.shape(nodes)[-1]]], axis=0
    )
    return tf.reshape(features, output_shape)


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx, nv_nodes, nv_neighbors):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return vs_concat(h_neighbors, h_nodes, nv_neighbors, nv_nodes)
