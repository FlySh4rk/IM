from typing import List, Optional, Iterable

import tensorflow as tf
import keras


class Sampler:
    def __call__(self, *args, **kwargs) -> tf.Tensor:
        if (len(args)) > 0:
            (inputs,) = args
        else:
            inputs = kwargs['inputs']
        return self.call(inputs)

    def call(self, inputs) -> tf.Tensor:
        pass


class CubeSampler(Sampler):

    def call(self, inputs) -> tf.Tensor:
        if (tf.is_symbolic_tensor(inputs)):
            # Fake generate a tensor with the expected shape
            return tf.repeat(tf.repeat(tf.reduce_sum(inputs, axis=(1, 2), keepdims=True),
                                       ((self.density + 1) ** self.dim), axis=1), self.dim, axis=2)
        n = inputs.shape[0]

        def gen_vec(rank: int = 3) -> Iterable[List[int]]:
            xx = [i * self.factor / self.density - self.factor / 2 for i in range(self.density + 1)]
            c = [0] * rank
            for n in range((self.density + 1) ** rank):
                yield [xx[c[i]] for i in range(rank)]
                add = 1
                for i in range(rank):
                    c[i] += add
                    if c[i] >= len(xx):
                        c[i] = 0
                        add = 1
                    else:
                        add = 0

        r = tf.convert_to_tensor([
            vec for vec in gen_vec(self.dim)
        ])

        rr = tf.reshape(r, (1, *r.shape))
        rr = tf.broadcast_to(r, (n, *r.shape))
        return rr

    def __init__(self, density: int = 10, factor: float = 1., dim: int = 3):
        super().__init__()
        self.dim = dim
        self.density = density
        self.factor = factor


class OriginalSampler(Sampler):
    def __init__(self, spatial_dims: int = 3):
        super().__init__()
        self.spatial_dims = spatial_dims

    def call(self, inputs, *args, **kwargs):
        res, _ = tf.split(inputs, [self.spatial_dims, -1], axis=-1)
        return res


class IdentityMapper(keras.layers.Layer):

    def __init__(self, spatial_dims: int = 3):
        super().__init__(trainable=False)
        self.spatial_dims = spatial_dims

    def call(self, inputs, *args, **kwargs):
        res, _ = tf.split(inputs, [self.spatial_dims, -1], axis=-1)
        return res


class ScatteredConv(keras.layers.Layer):

    def __init__(self, num_points: int = 3,
                 weighter: Optional[List[keras.layers.Layer]] = None):
        """

        :param num_points: num of focal points
        :param weighter: the weighter layer
        """
        super().__init__()
        self.num_points = num_points
        self.points = None
        self.weighter = weighter

    def build(self, input_shape):
        points_shape, center_shape = input_shape
        # l = num of centers per each test
        # d = spacial dimensions
        (_, l, d) = center_shape
        self.points = self.add_weight(name="points",
                                      shape=(self.num_points, d),
                                      trainable=True,
                                      initializer=tf.random_normal_initializer)

    def call(self, inputs, *args, **kwargs):
        points, centers = inputs
        # Initial dumb implementation
        # l = num of centers (it's an input variable)
        # d = the spacial dimensions
        # m = num of focal points (num_points)
        m = self.num_points
        # d = spacial dimensions
        # inputs : set of centers, (:,l,d) broadcast to (:,l,m,d)
        # rel points: (m,d) broadcast to (:,l,m,d)
        (t, l, d) = centers.shape
        t = -1 if t is None else t
        # let's create the focal points
        focals = tf.reshape(self.points, (1, 1, m, d))
        focals = tf.broadcast_to(focals, (t, l, m, d))
        centers_p = tf.reshape(centers, (t, l, 1, d))
        centers_p = tf.broadcast_to(centers_p, (t, l, m, d))

        focals = focals + centers_p

        # Now split inputs into coord and features
        (_, n, d_and_f) = points.shape
        coords, feats = tf.split(points, [d, -1], axis=-1)

        coords = tf.reshape(coords, (t, 1, 1, n, d))
        focals = tf.reshape(focals, (t, l, m, 1, d))
        coords = tf.broadcast_to(coords, (t, l, m, n, d))
        focals = tf.broadcast_to(focals, (t, l, m, n, d))

        delta = coords - focals

        # compute norm

        norm = delta * delta
        norm = tf.reduce_sum(norm, -1)
        # breakpoint()
        # Compute features
        new_feats = [w(norm) for w in self.weighter]
        new_feats = tf.stack(new_feats, axis=-1)

        return new_feats


class CrossConvLayer(keras.layers.Layer):
    """
    Like ScatteredConv but the points are fixed
    """

    def __init__(self, weighter: Optional[List[keras.layers.Layer]] = None, dist: float = 3.0,
                 coeff: float = 0.1, avg_delta=True, do_norm=True):
        """

        :param num_points: num of focal points
        :param weighter: the weighter layer
        """
        super().__init__()
        self.coeff = coeff
        self.dist = dist
        self.points = None
        self.weighter = weighter
        self.avg_delta = avg_delta
        self.do_norm = do_norm

    def build(self, input_shape):
        points_shape, center_shape = input_shape
        import math
        # l = num of centers per each test
        # d = spacial dimensions
        (_, l, d) = center_shape
        angles = tf.convert_to_tensor(
            [[j * 0.125 - 0.125, i * 0.125 + (j - 1) * 0.0625] for j in range(0, 3) for i in
             range(0, 8)] + [[-0.25, 0], [0.25, 0]]) * math.pi * 2

        self.points = tf.convert_to_tensor(
            [[tf.sin(a), tf.cos(a) * tf.cos(b), tf.cos(a) * tf.sin(b)] for a, b in
             angles]) * self.dist

    def call(self, inputs, *args, **kwargs):
        points, centers = inputs
        # Initial dumb implementation
        # l = num of centers (it's an input variable)
        # d = the spacial dimensions
        # m = num of focal points (num_points)
        m = self.points.shape[0]
        # d = spacial dimensions
        # inputs : set of centers, (:,l,d) broadcast to (:,l,m,d)
        # rel points: (m,d) broadcast to (:,l,m,d)
        (t, l, d) = centers.shape
        t = -1 if t is None else t
        # let's create the focal points
        focals = tf.reshape(self.points, (1, 1, m, d))
        focals = tf.broadcast_to(focals, (t, l, m, d))
        centers_p = tf.reshape(centers, (t, l, 1, d))
        centers_p = tf.broadcast_to(centers_p, (t, l, m, d))

        focals = focals + centers_p

        # Now split inputs into coord and features
        (_, n, d_and_f) = points.shape
        coords, feats = tf.split(points, [d, -1], axis=-1)

        coords = tf.reshape(coords, (t, 1, 1, n, d))
        focals = tf.reshape(focals, (t, l, m, 1, d))
        coords = tf.broadcast_to(coords, (t, l, m, n, d))
        focals = tf.broadcast_to(focals, (t, l, m, n, d))

        delta = coords - focals

        # compute norm
        if self.avg_delta:
            delta = tf.reduce_mean(delta, axis=-2)

        if self.do_norm:
            norm = delta * delta
            norm = tf.reduce_sum(norm, -1)
        else:
            norm = tf.reduce_mean(delta, -2)

        # Mean over all points => (t, l, m)
        if not self.avg_delta:
            norm = tf.reduce_mean(norm, axis=-1)

        if self.do_norm:
            norm = self.coeff / (norm + self.coeff)

        norm_and_feats = tf.concat([norm, feats], axis=-1)

        # breakpoint()
        # Compute features => (t,l, w*self.weighter)
        new_feats = [w(norm_and_feats) for w in self.weighter]
        new_feats = tf.concat(new_feats, axis=-1)

        return new_feats


class CrossConvLayerV2(keras.layers.Layer):
    """
    Like ScatteredConv but the points are fixed
    """

    def __init__(self, weighter: Optional[List[keras.layers.Layer]] = None, dist: float = 3.0,
                 coeff: float = 0.1, avg_delta=True, do_norm=True):
        """

        :param num_points: num of focal points
        :param weighter: the weighter layer
        """
        super().__init__()
        self.coeff = coeff
        self.dist = dist
        self.points = None
        self.weighter = weighter
        self.avg_delta = avg_delta
        self.do_norm = do_norm

    def build(self, input_shape):
        points_shape, center_shape = input_shape
        import math
        # l = num of centers per each test
        # d = spacial dimensions
        (_, l, d) = center_shape
        angles = tf.convert_to_tensor(
            [[j * 0.125 - 0.125, i * 0.125 + (j - 1) * 0.0625] for j in range(0, 3) for i in
             range(0, 8)] + [[-0.25, 0], [0.25, 0]]) * math.pi * 2

        self.points = tf.convert_to_tensor(
            [[tf.sin(a), tf.cos(a) * tf.cos(b), tf.cos(a) * tf.sin(b)] for a, b in
             angles]) * self.dist

    def call(self, inputs, *args, **kwargs):
        points, centers = inputs
        # Initial dumb implementation
        # l = num of centers (it's an input variable)
        # d = the spacial dimensions
        # m = num of probes (num_points)
        m = self.points.shape[0]
        # d = spacial dimensions
        # inputs : set of centers, (:,l,d) broadcast to (:,l,m,d)
        # rel points: (m,d) broadcast to (:,l,m,d)
        (_, l, d) = centers.shape
        # let's create the focal points
        probes_per_center = tf.reshape(self.points, (-1, 1, m, d))
        centers_p = tf.reshape(centers, (-1, l, 1, d))

        probes_per_center = probes_per_center + centers_p

        # Now split inputs into coord and features
        (_, n, d_and_f) = points.shape
        coords, feats = tf.split(points, [d, -1], axis=-1)  # (t,n,d) , (t,n, old_f)

        coords = tf.reshape(coords, (-1, 1, 1, n, d))
        probes_per_center = tf.reshape(probes_per_center, (-1, l, m, 1, d))

        delta = coords - probes_per_center

        # compute norm
        norm = delta * delta
        norm = tf.reduce_sum(norm, -1)
        norm = self.coeff / (norm + self.coeff)  # (t,l,m,n)

        # How could we combine previous layer feats with the weight
        if feats.shape[-1] == 0:
            feats = tf.ones((1, 1, n, 1))
        else:
            feats = tf.reshape(feats, (-1, 1, n, d_and_f - d))

        norm = tf.linalg.matmul(norm,
                                feats) / n  # (t,l,m,n) x (1,1,n,old_f) => (t,l,m,old_f), old_f gets multiplied by each probe

        norm = tf.reshape(norm,
                          (-1, l, norm.shape[-1] * norm.shape[-2]))  # (t,l, m*old_f) flatten those

        # Mean over all target points => (t, l, m)
        # norm = tf.reduce_mean(norm, axis=-1)  # (t,l,m)

        # breakpoint()
        # Compute features => (t,l, w*self.weighter)
        new_feats = [w(norm)  # (t,l,m) => (t,l,f)
                     for w in self.weighter]
        new_feats = tf.concat(new_feats, axis=-1)

        return new_feats


class FixedNorm(keras.constraints.Constraint):
    """Constrains weight tensors to be centered around `ref_value`."""

    def __init__(self, dist: tf.float32):
        self.dist = dist

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        norm = tf.norm(w, axis=-1, keepdims=True)
        return w * self.dist / norm

    def get_config(self):
        return {'dist': self.dist}


class CrossConvLayerVariadic(keras.layers.Layer):
    """
    Like ScatteredConv but the points are fixed
    """

    def __init__(self, weighter: Optional[List[keras.layers.Layer]] = None, dist: float = 3.0,
                 coeff: float = 0.1, avg_delta=True, do_norm=True, num_points=16):
        """

        :param num_points: num of focal points
        :param weighter: the weighter layer
        """
        super().__init__()
        self.num_points = num_points
        self.coeff = coeff
        self.dist = dist
        self.points = None
        self.weighter = weighter
        self.avg_delta = avg_delta
        self.do_norm = do_norm

    def build(self, input_shape):
        points_shape, center_shape = input_shape
        # l = num of centers per each test
        # d = spacial dimensions
        (_, l, d) = center_shape

        self.points = self.add_weight(shape=(self.num_points, 2), dtype=tf.float32,
                                      trainable=True)

    def call(self, inputs, *args, **kwargs):
        points, centers = inputs
        # Initial dumb implementation
        # l = num of centers (it's an input variable)
        # d = the spacial dimensions
        # m = num of focal points (num_points)
        m = self.points.shape[0]
        # d = spacial dimensions
        # inputs : set of centers, (:,l,d) broadcast to (:,l,m,d)
        # rel points: (m,d) broadcast to (:,l,m,d)
        (t, l, d) = centers.shape
        t = -1 if t is None else t
        # let's create the focal points
        probes = tf.concat([tf.cos(self.points[:, 0]),
                            tf.sin(self.points[:, 0] * tf.cos(self.points[:, 1]),
                                   tf.sin(self.points[:, 0]) * tf.sin(self.points[:, 1]))],
                           axis=-1) * self.dists
        focals = tf.reshape(probes, (1, 1, m, d))
        focals = tf.broadcast_to(focals, (t, l, m, d))
        centers_p = tf.reshape(centers, (t, l, 1, d))
        centers_p = tf.broadcast_to(centers_p, (t, l, m, d))

        focals = focals + centers_p

        # Now split inputs into coord and features
        # TODO: add a dense layer / weighted mean reduction to summarize the points and features
        # into a single set of coordinates and features
        (_, n, d_and_f) = points.shape
        coords, feats = tf.split(points, [d, -1], axis=-1)

        coords = tf.reshape(coords, (t, 1, 1, n, d))
        focals = tf.reshape(focals, (t, l, m, 1, d))
        coords = tf.broadcast_to(coords, (t, l, m, n, d))
        focals = tf.broadcast_to(focals, (t, l, m, n, d))

        delta = coords - focals

        # compute norm
        if self.avg_delta:
            delta = tf.reduce_mean(delta, axis=-2)

        if self.do_norm:
            norm = delta * delta
            norm = tf.reduce_sum(norm, -1)
        else:
            norm = tf.reduce_mean(delta, -2)

        # Mean over all points => (t, l, m)
        if not self.avg_delta:
            norm = tf.reduce_mean(norm, axis=-1)

        if self.do_norm:
            norm = self.coeff / (norm + self.coeff)

        norm_and_feats = tf.concat([norm, feats], axis=-1)

        # breakpoint()
        # Compute features => (t,l, w*self.weighter)
        new_feats = [w(norm_and_feats) for w in self.weighter]
        new_feats = tf.concat(new_feats, axis=-1)

        return new_feats


class CrossConvLayerVariadic2(keras.layers.Layer):
    """
    Like ScatteredConv but the points are fixed
    """

    def __init__(self, weighter: Optional[List[keras.layers.Layer]] = None, dist: float = 3.0,
                 coeff: float = 10, num_points=16, enable_summaary_layer=True):
        """

        :param num_points: num of focal points
        :param weighter: the weighter layer
        """
        super().__init__()
        self.enable_summaary_layer = enable_summaary_layer
        self.num_points = num_points
        self.coeff = coeff
        self.dist = dist
        self.points = None
        self.weighter = weighter
        self.summary_layer = None

    def build(self, input_shape):
        points_shape, center_shape = input_shape
        # l = num of centers per each test
        # d = spacial dimensions
        (_, l, d) = center_shape
        if self.enable_summaary_layer:
            self.summary_layer = keras.Sequential([
                keras.layers.Dense(units=d),
                # keras.layers.Dropout(rate=0.10)
            ])

        self.points = self.add_weight(shape=(self.num_points, 2), dtype=tf.float32,
                                      trainable=True)

    def call(self, inputs, *args, **kwargs):
        points, centers = inputs
        # Initial dumb implementation
        # l = num of centers (it's an input variable)
        # d = the spacial dimensions
        # m = num of probes (num_points)
        # d = spacial dimensions
        # inputs : set of centers, (:,l,d) broadcast to (:,l,m,d)
        # rel points: (m,d) broadcast to (:,l,m,d)
        (_, l, d) = centers.shape
        # let's create the focal points
        probes = tf.concat([tf.expand_dims(tf.sin(self.points[:, 0]), -1),
                            tf.expand_dims(tf.cos(self.points[:, 0]) * tf.cos(self.points[:, 1]),
                                           -1),
                            tf.expand_dims(tf.cos(self.points[:, 0]) * tf.sin(self.points[:, 1]),
                                           -1)],
                           axis=-1) * self.dist
        focals = tf.reshape(probes, (1, 1, -1, d))
        # focals = tf.reshape(self.points, (1, 1, -1, d))
        # focals = tf.broadcast_to(focals, (t, l, m, d))
        centers_p = tf.reshape(centers, (-1, l, 1, d))
        # centers_p = tf.broadcast_to(centers_p, (t, l, m, d))

        focals = focals + centers_p  # (t,l,m,d)
        (_, n, d_and_f) = points.shape

        if self.enable_summaary_layer:
            coords = self.summary_layer(tf.reshape(points, (-1, n * d_and_f)))
            coords = tf.reshape(coords, (-1, 1, d))
        else:
            coords = tf.reduce_mean(points, axis=1, keepdims=True)  # (t,1,d)
        # coords = self.summary_level(coords)  # (t,d)

        coords, feats = tf.split(coords, (d, -1), axis=-1)
        coords = tf.reshape(coords, (-1, 1, 1, d))
        # coords = tf.broadcast_to(coords, (t, l, m, d))

        delta = coords - focals  # ( t,l,m,d)

        # compute norm

        norm = delta * delta
        norm = tf.reduce_sum(norm, -1)

        norm = self.coeff / (norm + self.coeff)
        # breakpoint()
        # Compute features => (t,l, w*self.weighter)
        new_feats = [w(norm) for w in self.weighter]
        new_feats = tf.concat(new_feats, axis=-1)

        return new_feats


class DirectAndInverseWeighter(keras.layers.Layer):
    """
    Takes as input (:,l,m,n) and outputs (:,l), where
    - l is number of centers
    - m is number of focal points
    - n is the number of original points
    """

    def __init__(self):
        super().__init__(trainable=True)
        self.w1 = None
        self.w2 = None

    def build(self, input_shape):
        (_, l, m, n) = input_shape
        self.w1 = self.add_weight(name="w1", shape=(m, 1), trainable=True)
        self.w2 = self.add_weight(name="w2", shape=(m, 1), trainable=True)

    def call(self, inputs, *args, **kwargs):
        (t, l, m, n) = inputs.shape
        t = -1 if t is None else t
        rev = 1 / (inputs + 0.01)

        # compute average per n
        focal_avg = tf.reduce_sum(inputs, axis=-1)  # (t,l,m)
        rev_avg = tf.reduce_sum(rev, axis=-1)  # (t,l,m)

        focal_dot = tf.squeeze(tf.matmul(focal_avg, self.w1), axis=2)  # (t,l)
        rev_dot = tf.squeeze(tf.matmul(rev_avg, self.w2), axis=2)  # (t,l)

        res = (focal_dot + rev_dot) / 2

        return res


class VConv(keras.layers.Layer):
    """
    Implements a Vector Convolution layer for pointnets.

    The main idea is to create a projection from the n dimensional target vector space to the
    m dimensional source vector space, then sampling the the smallest n-cube (or n-sphere) containing
    the pre image of the source set of points and applying a parametric convolution function centered
    in the image of the sampling points to the original points. The result is a set of new points in the
    target space.
    If the input has shape (l,m,f) (l points of m dimensions with f features) the output of
    k parallel convolutions of F-arity with a sample density of d will have shape (l*d, n, F, k) that
    can be flattened to (l*d, n, F*k) or (l*d, n+F, k)
    The convolution function is a function that depends on a point on the original space, called
    "the center", that computes a set of new features from a set of the input vectors and has trainable parameters.
    The mapping function maps from the target space to the original space may also have trainable
    parameters.
    Thus, there are three main components:
    - the mapping function from target to source
    - the sampling strategy
    - the convolution function

    For instance we can have:
    - mapping function : the identity function
    - sampling strategy : the original points (fixed density = 1)
    - convolution function: weighted of features sum of distances from three relative points around the center,
        the relative coordinates are the parameters

    Another option could be:
    - mapping function : a trainable linear trasformation from T to S
    - sampling strategy : equal spaced l*d points on a m-cube in the origin
    - convolution function: exponential of the invers of distance from n*2 points at fixed distance
      on each target axis, the parameters are the exponents coeff


    """

    def __init__(self, mapping: keras.layers.Layer,
                 sampler: Sampler,
                 conv: List[keras.layers.Layer]):
        super().__init__()
        self.mapping = mapping
        self.sampler = sampler
        self.conv = conv

    def call(self, inputs, *args, **kwargs):
        centers_orig = self.sampler(inputs)
        mapped = self.mapping(centers_orig)
        convs = tf.concat([c([inputs, mapped]) for c in self.conv], axis=-1)
        # breakpoint()
        if centers_orig.shape[1] != convs.shape[1]:
            return convs
        res = tf.concat([centers_orig, convs], axis=-1)
        return res


class OriginalConv(VConv):
    def __init__(self, num_weighters: int = 1, num_convs: int = 5):
        super().__init__(mapping=IdentityMapper(),
                         sampler=OriginalSampler(),
                         conv=[
                             ScatteredConv(
                                 weighter=[DirectAndInverseWeighter()
                                           for _ in range(num_weighters)])
                             for _ in range(num_convs)])


class CrossConv(VConv):
    def __init__(self, out_feats: int = 6, num_convs: int = 5, dist=3.0, coeff=10, ):
        super().__init__(mapping=IdentityMapper(),
                         sampler=OriginalSampler(),
                         conv=[
                             CrossConvLayer(dist=dist, coeff=coeff,
                                            weighter=[
                                                keras.Sequential([
                                                    keras.layers.Dense(units=out_feats),
                                                    keras.layers.Dropout(rate=0.10),
                                                    keras.layers.LeakyReLU()
                                                ])
                                            ])
                             for _ in range(num_convs)])


class CrossConvV2(VConv):
    def __init__(self, out_feats: int = 6, num_convs: int = 5, dist=3.0, coeff=10,
                 weighter_depth=1):
        super().__init__(mapping=IdentityMapper(),
                         sampler=OriginalSampler(),
                         conv=[
                             CrossConvLayerV2(dist=dist, coeff=coeff,
                                              weighter=[
                                                  keras.Sequential([layer for layers in [[
                                                      keras.layers.Dense(
                                                          units=out_feats),
                                                      keras.layers.Dropout(
                                                          rate=0.10),
                                                      keras.layers.LeakyReLU()
                                                  ] for _ in range(weighter_depth)] for layer in
                                                                       layers])
                                              ])
                             for _ in range(num_convs)])


class CrossConvVariadic(VConv):
    def __init__(self, out_feats: int = 6, num_convs: int = 5, dist=3.0, coeff=10, num_points=16):
        super().__init__(mapping=IdentityMapper(),
                         sampler=OriginalSampler(),
                         conv=[
                             CrossConvLayerVariadic(dist=dist,
                                                    coeff=coeff,
                                                    num_points=num_points,
                                                    weighter=[
                                                        keras.Sequential([
                                                            keras.layers.Dense(units=out_feats),
                                                            keras.layers.Dropout(rate=0.10),
                                                            keras.layers.LeakyReLU()
                                                        ])
                                                    ])
                             for _ in range(num_convs)])


class CrossConvVariadic2(VConv):
    def __init__(self, out_feats: int = 6, num_convs: int = 5, dist=3.0, coeff=10, num_points=16,
                 summary_level=True):
        super().__init__(mapping=IdentityMapper(),
                         sampler=OriginalSampler(),
                         conv=[
                             CrossConvLayerVariadic2(dist=dist,
                                                     coeff=coeff,
                                                     num_points=num_points,
                                                     enable_summaary_layer=summary_level,
                                                     weighter=[
                                                         keras.Sequential([
                                                             keras.layers.Dense(units=out_feats),
                                                             keras.layers.Dropout(rate=0.10),
                                                             keras.layers.LeakyReLU()
                                                         ])
                                                     ])
                             for _ in range(num_convs)])


class CubeConv(VConv):
    def __init__(self, num_weighters: int = 1, num_convs: int = 5, density: int = 10,
                 sampler_dim: int = 3, input_dim: int = 3):
        super().__init__(mapping=keras.layers.Dense(units=input_dim),
                         sampler=CubeSampler(density=density, dim=sampler_dim),
                         conv=[
                             ScatteredConv(
                                 weighter=[DirectAndInverseWeighter()
                                           for _ in range(num_weighters)])
                             for _ in range(num_convs)])
