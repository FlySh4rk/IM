import numpy as np
import tensorflow as tf
# import tensorflow.keras.layers as layers
import keras

from utils import CrossProductLayer, PermLayer


def setup_model_v3(order=2, num_perm=400, with_squares=True, input_dim=(48,), num_not_lin=3,
                   units=32) -> keras.Model:
    input_layer = keras.layers.Input(input_dim)
    x = input_layer

    # layers
    perm = PermLayer(num_permutation=num_perm, flatten=False, groups=[(0, 8), (8, 16)])
    # x = perm(input_layer)

    # pool = keras.layers.GlobalAvgPool1D(data_format="channels_last")
    # x = pool(x)

    cross = CrossProductLayer(order=order, with_squares=with_squares)
    x = cross(x)

    dense1 = keras.layers.Dense(units=units)

    x = dense1(x)

    def not_lin(x):
        densen = keras.layers.Dense(units=units)
        actn = keras.layers.LeakyReLU()

        x = densen(x)
        x = actn(x)
        return x

    not_lins = [x] + [not_lin(x) for _ in range(num_not_lin)]

    concat = keras.layers.Concatenate(axis=-1)

    x = concat(not_lins)

    densef = keras.layers.Dense(units=7)
    x = densef(x)

    def _tan(x):
        if tf.is_symbolic_tensor(x):
            return tf.gather(x, axis=-1, indices=list(range(5)))
        x = x.numpy()
        a = np.arctan2(x[:, 0], x[:, 1])
        b = np.arctan2(x[:, 2], x[:, 3])
        r = tf.convert_to_tensor([x[:, 4], x[:, 5], x[:, 6], a, b])
        return tf.transpose(r, (1, 0))

    tan = keras.layers.Lambda(_tan)

    x = tan(x)
    output_layer = x

    # secondary output for implant selection
    # Will add this when required currently commented out
    # densec = keras.layers.Dense(units=5)
    # y = densec(x)
    # sm = keras.layers.Softmax()
    # y = sm(y)
    # output_layer = [output_layer, y]

    model = keras.models.Model(input_layer, output_layer)
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(
            learning_rate=keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.01,
                                                                         decay_steps=400,
                                                                         decay_rate=0.001)),
        loss=keras.losses.MeanSquaredError(),
        # keras.losses.MeanSquaredError(name="train_loss"),
    )

    return model


def setup_model_v4(order=2, with_squares=True, input_dim=(48,), num_not_lin=3,
                   units=32, lin_depth=1, dropout=0.05, angle_weight=10,
                   non_lin_activation=keras.layers.LeakyReLU) -> keras.Model:
    input_layer = keras.layers.Input(input_dim)

    # layers
    x = input_layer

    if order > 1:
        cross = CrossProductLayer(order=order, with_squares=with_squares, )
        x = cross(x)

    dense1 = keras.layers.Dense(units=units)

    x1 = dense1(x)

    drop = keras.layers.Dropout(rate=dropout)

    x = drop(x1)

    concat1 = keras.layers.Concatenate()
    x = concat1([x1, x])

    if num_not_lin:

        def not_lin(x):
            for i in range(lin_depth):
                densen = keras.layers.Dense(units=max(units / (2 ** i), 12))
                norm = keras.layers.BatchNormalization()
                actn = non_lin_activation()
                dout = keras.layers.Dropout(rate=dropout)

                x = densen(x)
                x = norm(x)
                x = actn(x)
                x = dout(x)
            return x

        not_lins = [x] + [not_lin(x) for _ in range(num_not_lin)]

        concat = keras.layers.Concatenate(axis=-1)

        x = concat(not_lins)

    densef = keras.layers.Dense(units=7)
    x = densef(x)

    # drop2 = keras.layers.Dropout(rate=dropout)
    # x = drop2(x)

    def norm_angles(i: tf.Tensor) -> tf.Tensor:
        # First angle
        c1 = tf.reshape(i[:, 3], (-1, 1))
        s1 = tf.reshape(i[:, 4], (-1, 1))
        n1 = tf.sqrt(c1 * c1 + s1 * s1)

        # Second angle
        c2 = tf.reshape(i[:, 5], (-1, 1))
        s2 = tf.reshape(i[:, 6], (-1, 1))
        n2 = tf.sqrt(c2 * c2 + s2 * s2)

        # Putting it all together
        r = tf.concat(
            [i[:, 0:3], c1 * angle_weight / n1, s1 * angle_weight / n1, c2 * angle_weight / n2,
             s2 * angle_weight / n2], axis=-1)

        return r

    norm_layer = keras.layers.Lambda(norm_angles)

    x = norm_layer(x)

    output_layer = x

    model = keras.models.Model(input_layer, output_layer)

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(
            learning_rate=keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.01,
                                                                         decay_steps=400,
                                                                         decay_rate=0.001)),
        loss=keras.losses.MeanSquaredError(),
        # keras.losses.MeanSquaredError(name="train_loss"),
    )

    return model


class MyLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        mean_vals = tf.gather(y_true, indices=[0, 2, 3, 4], axis=-1)
        cost = self.mean(mean_vals, y_pred)
        return cost
