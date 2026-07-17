"""Geometric vector perceptron layers used by the legacy PocketMiner model.

Attribute names and layer construction intentionally mirror the historical
TensorFlow implementation because TensorFlow object checkpoints are restored
through the tracked object graph.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Layer, LayerNormalization


def norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = tf.maximum(tf.math.reduce_sum(tf.math.square(x), axis, keepdims), eps)
    return tf.sqrt(out) if sqrt else out


class GVP(Model):
    def __init__(self, vi, vo, so, nlv=tf.math.sigmoid, nls=tf.nn.relu):
        super().__init__()
        if vi:
            self.wh = Dense(max(vi, vo))
        self.ws = Dense(so, activation=nls)
        if vo:
            self.wv = Dense(vo)
        self.vi = vi
        self.vo = vo
        self.so = so
        self.nlv = nlv

    def call(self, x, return_split=False):
        v, s = split(x, self.vi)
        if self.vi:
            vh = self.wh(v)
            vn = norm_no_nan(vh, axis=-2)
            out = self.ws(tf.concat([s, vn], -1))
        else:
            out = self.ws(s)
        if self.vo:
            vo = self.wv(vh)
            if self.nlv:
                vo *= self.nlv(norm_no_nan(vo, axis=-2, keepdims=True))
            out = (vo, out) if return_split else merge(vo, out)
        return out


class GVPDropout(Layer):
    """Drop vector and scalar channels independently."""

    def __init__(self, rate, nv):
        super().__init__()
        self.nv = nv
        self.vdropout = Dropout(rate, noise_shape=[1, nv])
        self.sdropout = Dropout(rate)

    def call(self, x, training=False):
        if not training:
            return x
        v, s = split(x, self.nv)
        return merge(self.vdropout(v, training=training), self.sdropout(s, training=training))


class GVPLayerNorm(Layer):
    def __init__(self, nv):
        super().__init__()
        self.nv = nv
        self.snorm = LayerNormalization()

    def call(self, x):
        v, s = split(x, self.nv)
        vn = norm_no_nan(v, axis=-2, keepdims=True, sqrt=False)
        vn = tf.sqrt(tf.math.reduce_mean(vn, axis=-1, keepdims=True))
        return merge(tf.math.divide_no_nan(v, vn), self.snorm(s))


def split(x, nv):
    """Split merged ``[..., 3*nv + ns]`` storage into vector/scalar parts."""

    static_shape = x.shape.as_list() if hasattr(x.shape, "as_list") else list(x.shape)
    target_shape = static_shape[:-1] + [3, nv]
    if any(dimension is None for dimension in target_shape[:-2]):
        target_shape = tf.concat([tf.shape(x)[:-1], [3, nv]], axis=0)
    v = tf.reshape(x[..., : 3 * nv], target_shape)
    s = x[..., 3 * nv :]
    return v, s


def merge(v, s):
    """Merge vector/scalar storage, preserving vector channels first."""

    static_shape = v.shape.as_list() if hasattr(v.shape, "as_list") else list(v.shape)
    target_shape = static_shape[:-2] + [3 * static_shape[-1]]
    if any(dimension is None for dimension in target_shape[:-1]):
        target_shape = tf.concat([tf.shape(v)[:-2], [3 * tf.shape(v)[-1]]], axis=0)
    return tf.concat([tf.reshape(v, target_shape), s], -1)


def vs_concat(x1, x2, nv1, nv2):
    v1, s1 = split(x1, nv1)
    v2, s2 = split(x2, nv2)
    return merge(tf.concat([v1, v2], -1), tf.concat([s1, s2], -1))


# Historical aliases used by some PocketMiner checkpoints/code releases.
Velu = GVP
VGDropout = GVPDropout
VGLayerNorm = GVPLayerNorm
