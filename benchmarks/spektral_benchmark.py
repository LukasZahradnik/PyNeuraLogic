import os

from tensorflow.python.keras.layers import Activation

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

import numpy as np
from keras import Model
from keras.layers import Dense

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.layers import GCNConv, GINConv, GraphSageConv
from spektral.layers.pooling import global_pool


class NetGCN(Model):
    def __init__(self, dim: int = 10):
        super().__init__()
        self.conv1 = GCNConv(dim, activation="relu", use_bias=False)
        self.conv2 = GCNConv(dim, use_bias=False)

        self.mean_pool = global_pool.get("avg")()
        self.fc1 = Dense(1, activation="sigmoid", use_bias=False)

    def call(self, inputs, training=None, mask=None):
        x, a, i, p = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])

        x = self.mean_pool(x)
        return self.fc1(x)


class NetGraphSage(Model):
    def __init__(self, dim: int = 10):
        super().__init__()
        self.conv1 = GraphSageConv(dim, activation="relu", use_bias=False)
        self.conv2 = GraphSageConv(dim, use_bias=False)

        self.mean_pool = global_pool.get("avg")()
        self.fc1 = Dense(1, activation="sigmoid", use_bias=False)

    def call(self, inputs, training=None, mask=None):
        x, a, i, p = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])

        x = self.mean_pool(x)
        return self.fc1(x)


class NetGIN(Model):
    def __init__(self, dim: int = 10):
        super().__init__()
        self.conv1 = GINConv(
            dim, mlp_batchnorm=False, activation="relu", epsilon=0, use_bias=False, mlp_hidden=[dim, dim]
        )
        self.conv2 = GINConv(
            dim, mlp_batchnorm=False, activation="relu", epsilon=0, use_bias=False, mlp_hidden=[dim, dim]
        )
        self.conv3 = GINConv(
            dim, mlp_batchnorm=False, activation="relu", epsilon=0, use_bias=False, mlp_hidden=[dim, dim]
        )
        self.conv4 = GINConv(
            dim, mlp_batchnorm=False, activation="relu", epsilon=0, use_bias=False, mlp_hidden=[dim, dim]
        )
        self.conv5 = GINConv(
            dim, mlp_batchnorm=False, activation="relu", epsilon=0, use_bias=False, mlp_hidden=[dim, dim]
        )

        self.mean_pool = global_pool.get("avg")()

        self.l1 = Dense(1, use_bias=False)
        self.l2 = Dense(1, use_bias=False)
        self.l3 = Dense(1, use_bias=False)
        self.l4 = Dense(1, use_bias=False)
        self.l5 = Dense(1, use_bias=False)

        self.act = Activation("sigmoid")

    def call(self, inputs, training=None, mask=None):
        x, a = inputs[0], inputs[1]

        x1 = self.conv1([x, a])
        x2 = self.conv2([x1, a])
        x3 = self.conv3([x2, a])
        x4 = self.conv4([x3, a])
        x5 = self.conv5([x4, a])

        m1 = self.mean_pool(x1)
        m2 = self.mean_pool(x2)
        m3 = self.mean_pool(x3)
        m4 = self.mean_pool(x4)
        m5 = self.mean_pool(x5)

        stacked = tf.stack([self.l1(m1), self.l2(m2), self.l3(m3), self.l4(m4), self.l5(m5)], axis=0)
        x = tf.reduce_sum(stacked, 0)

        return self.act(x)


def get_model(model):
    if model == "gcn":
        return NetGCN
    if model == "gsage":
        return NetGraphSage
    if model == "gin":
        return NetGIN
    raise NotImplementedError


def evaluate(model, dataset, steps, dataset_loc, dim):
    ds = TUDataset(dataset)
    loader = DisjointLoader(ds, batch_size=1, epochs=steps)

    model = get_model(model)(dim=dim)

    optimizer = Adam(1e-3)
    loss_fn = BinaryCrossentropy(from_logits=True)

    signature = loader.tf_signature()
    signature = (*signature[:-1], tf.TensorSpec(shape=(1, 1), dtype=tf.float64))

    @tf.function(input_signature=signature, experimental_relax_shapes=True)
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    times = []
    step = 0

    tm = 0
    for inputs, target in loader:
        target_ = tf.constant(np.argmax(target), shape=(1, 1), dtype=tf.float64)
        step += 1

        t = time.perf_counter()
        loss = train_step(inputs, target_)
        tm += time.perf_counter() - t

        if step == loader.steps_per_epoch:
            step = 0
            times.append(tm)
            tm = 0
    return times
