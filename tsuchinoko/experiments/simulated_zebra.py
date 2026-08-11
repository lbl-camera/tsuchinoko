import copy
import inspect
import time
from functools import cached_property, partial
from uuid import uuid4

import dask
import numpy as np
import pandas
from PIL import Image, ImageDraw
from loguru import logger
from pyqtgraph.parametertree.parameterTypes import SimpleParameter
from scipy import linalg, ndimage

from gpcam.gp_optimizer import GPOptimizer
# from fvgp.gp_kernels import get_distance_matrix
from tsuchinoko.adaptive import Data
from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.adaptive.grid import Grid
from tsuchinoko.core import CoreState
from tsuchinoko.core.messages import FullDataResponse, StateResponse
from tsuchinoko.core.zmq_core import ZMQCore
from tsuchinoko.execution.simple import SimpleEngine
from tsuchinoko.graphs.common import Table, image_grid, GPCamHyperparameterPlot, Score, GPCamPosteriorMean, \
    GPCamAcquisitionFunction
from tsuchinoko.graphs.specialized import ReconstructionGraph, ProjectionOperatorGraph, ProjectionMask, ReconHistogram, \
    sirt, RealSpacePosteriorMean, SinogramSpaceGPCamAcquisitionFunction, RealSpacePosteriorVariance
from tsuchinoko.utils import threads

session_id = uuid4()

#         4    0.812    0.203    0.813    0.203 simulated_zebra.py:34(get_distance_matrix)
import numba as nb

# NOTES:
## Difficulties:
# - (A.T @ y)/N does not converge; it grows with number of points
# - Posterior mean continues to increase due to nature of tomographic reconstruction having no baseline
# - A.T @ y does not normalize for non-uniform distribution of points in sinogram
# - Training still slow
# - Although the binary_resolve acquisition function performed favorably for the pacman, its harder to compete
#   with the zebra stripes, likely due to the structure being more complex.
# - Comparative performance will also be affected after introducing cost function
# - An acquisition function that effectively behaves like gradient in sinogram space is one goal
#
## Things to try:
# - training -> "mcmc", iterations -> smaller, tolerance -> bigger
# - first hyperparameter -> square of expected signal value (real space) -> sinogram values / real space width
# - dynamic leveling scaling values to
# - increase noise variance to scale with value
# - use local with cost function?
# - less max iterations in ask
# - less pop size
# - is append mode on?


@nb.njit
def get_distance_matrix(x1, x2):
    """
    Function to calculate the pairwise distance matrix of
    points in x1 and x2.

    Parameters
    ----------
    x1 : np.ndarray
        Numpy array of shape (U x D).
    x2 : np.ndarray
        Numpy array of shape (V x D).

    Return
    ------
    distance matrix : np.ndarray
    """
    d = np.zeros((len(x1), len(x2)))
    for i in range(x1.shape[1]):
        d += (np.expand_dims(x1[:, i], 1) - x2[:, i]) ** 2
        # d += np.subtract.outer(x1[:, i], x2[:, i]) ** 2
    return np.sqrt(d)


frequent_distance_matrix = None
def memoized_distance_matrix(x1, x2):
    global frequent_distance_matrix
    if x1.shape == x2.shape and last_A[1] is not None and x1.shape[0] == last_A[1].shape[1] and frequent_distance_matrix is not None:
        return frequent_distance_matrix
    else:
        frequent_distance_matrix = get_distance_matrix(x1, x2)
        return frequent_distance_matrix



def trapez(y, y0, w):
    return np.clip(np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1 - c0) < abs(r1 - r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1 - r0) / (c1 - c0)

    # Adjust weight by the slope
    w *= np.sqrt(1 + np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1 + 1, dtype=float)
    y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w / 2)
    yy = (np.floor(y).reshape(-1, 1) + np.arange(-thickness - 1, thickness + 2).reshape(1, -1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1, 1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


# def weighted_line(y0, x0, y1, x1):
#     line = np.zeros((l_det, l_det), dtype=np.uint8)
#     cv2.line(line, (int(x0), int(y0)), (int(x1), int(y1)), (255,), lineType=cv2.LINE_AA)
#     return line/255.

# TODO: resolve data shape being 361 bins in phi

def projection_operator(x, phi, map_size, center=None, width=1, length=None):
    if not length:
        length = map_size * 2

    if not center:
        center = ((map_size - 1) / 2, (map_size - 1) / 2)

    # get a sampling over a slice through x, phi
    v = np.array([-np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))]) * (-x + (map_size - 1) / 2)
    c = np.array(center)
    f = np.array([-np.sin(np.deg2rad(phi)), -np.cos(np.deg2rad(phi))]) * length / 2

    f0 = v + c + f
    f1 = v + c - f

    x0, y0 = f0
    x1, y1 = f1

    # draw a 1px line (with AA) as a mask from f0 to f1
    # NOTE: this is correct when the beam width is the same as the x/y resolution; for other cases,
    # you could use scipy.ndimage.zoom()
    rows, cols, vals = weighted_line(y0, x0, y1, x1, w=0.25)
    xy_mask = (cols >= 0) & (cols < map_size) & (rows >= 0) & (rows < map_size)
    masked_cols, masked_rows, masked_vals = cols[xy_mask], rows[xy_mask], vals[xy_mask]
    (map_mask := np.zeros((map_size, map_size)))[masked_rows, masked_cols] = masked_vals

    return map_mask


# def get_distance_matrix(x1, x2):
#     """
#     Function to calculate the pairwise distance matrix of
#     points in x1 and x2.
#
#     Parameters
#     ----------
#     x1 : np.ndarray
#         Numpy array of shape (U x D).
#     x2 : np.ndarray
#         Numpy array of shape (V x D).
#
#     Return
#     ------
#     distance matrix : np.ndarray
#     """
#     d = np.zeros((len(x1), len(x2)))
#     for i in range(x1.shape[1]): d += (x1[:, i].reshape(-1, 1) - x2[:, i]) ** 2
#     return np.sqrt(d)

# A = None

@nb.njit(parallel=True)
def fast_k(d, hps):
    return hps[0] * np.exp(-np.square(d) / hps[1])


def kernel(x1_real, x2_real, hps, A_=None):
    if A_ is None:
        A_ = last_A[1]

    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    caller_name = calframe[1][3]
    code_context = calframe[1][4][calframe[1][5]]

    # print('caller name:', caller_name)
    # print('kernel:', hps, x1, x2)
    # with log_time('kernel', cumulative_key='kernel'):
    # this kernel takes elements from the real space the prior covariance in sinogram space
    # newstyle
    d = memoized_distance_matrix(x1_real, x2_real)
    # oldstyle
    # d = get_distance_matrix(x1, x2)
    # k = hps[0] * np.exp(-d ** 2 / hps[1])
    k = fast_k(d, hps)
    if (caller_name in ['posterior_covariance', 'posterior_mean'] and ' k = ' in code_context):
        kernel = np.repeat(A_ @ k, n_sinograms, 0)
    elif caller_name in ["_d_gp_kernel_dx"]:  # prediction
        kernel = k
    elif caller_name == 'posterior_covariance' and ' kk = ' in code_context:  # prediction
        kernel = k
    elif len(x1_real) == len(x2_real) == A_.shape[1]:  # training [actually initialize>prior_covariance]
        kernel = linalg.block_diag(*[A_ @ k @ A_.T] * n_sinograms)
        # kernel = A @ k @ A.T
    else:
        raise Exception('how?')
    # elif len(x1_real) == len(x2_real) != A.shape[1]:  # prediction
    #     kernel = k
    # elif len(x1_real) != len(x2_real):  # prediction [posterior_covariance]
    #     kernel = np.repeat(A @ k, n_sinograms, 0)

    return kernel


def noise(x_real, hps, variances=None):
    """
    Returns noise in sinogram space (which is conveniently exactly the noise variance from measurements)
    """
    # - add a hyperparameter for noise
    # - reasonable bounds
    # - return vector of repeating hyperparameter

    # this function takes elements from the real space and returns noise (co)variances in sinogram space
    if variances is None:
        variances = core.data.variances
    return np.asarray(variances).reshape(-1)


# newstyle
def mean_func(x_real, hps, A_=None):
    if A_ is None:
        A_ = last_A[1]
    # oldstyle
    # def mean_func(obj, x, hps):
    # this function takes elements from the real space and returns prior mean values in sinogram space
    # m = A @ np.zeros(len(x),)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    caller_name = calframe[1][3]
    code_context = calframe[1][4][calframe[1][5]]
    if caller_name in ['posterior_mean', 'posterior_mean_grad']:
        # m = np.zeros((len(x_real),))
        low = -1 # ThresholdResolve.binary_low.value()
        high = 1 # ThresholdResolve.binary_high.value()
        mid = (high - low) / 2 + low
        # m = np.ones((len(x_real),)) * mid
        m = np.zeros((len(x_real),))
    elif len(x_real) == A_.shape[1]:
        m = np.repeat(A_ @ np.zeros(len(x_real)), n_sinograms)
    else:
        raise Exception('WHAT?')
    return m


def cost(origin, x, arguments=None):
    print('cost:', origin, x)
    cost_x = 1 / 8  # cost is 1/velocity
    cost_phi = 1 / 30
    exposure_time = 1
    estimated_total_experiment_time = 60 * 60 * 1  # in seconds
    x_origin, phi_origin = origin
    cost = [(exposure_time + max(abs(ix - x_origin) * cost_x,
                                 abs(iphi - phi_origin) * cost_phi)) / estimated_total_experiment_time for
            ix, iphi in x]
    return cost


last_variance = (-1, None)
last_mean = (-1, None)
last_recon = (-1, None)
last_A = (-1, None)


def memoized_A(x):
    global last_A
    length_diff = len(x) - last_A[0]
    if length_diff:
        new_A = last_A[1]
        for position in x[-length_diff:]:
            A_vec = projection_operator(*position, l_x).reshape(1, l_x ** 2)
            if new_A is None:
                new_A = np.empty((0,A_vec.shape[1]))
            new_A = np.vstack([new_A, A_vec])
        last_A = len(x), new_A
    return last_A[1]

def memoized_posterior_variance(x, gp):
    global last_variance
    if len(gp.y_data) != last_variance[0]:
        last_variance = len(gp.y_data), gp.posterior_covariance(x, variance_only=True)["v(x)"]
    return last_variance[1]


def memoized_posterior_mean(x, gp):
    global last_mean
    if len(gp.y_data) != last_mean[0]:
        last_mean = len(gp.y_data), gp.posterior_mean(x)["f(x)"]
    return last_mean[1]


def memoized_recon(gp, init=None):
    global last_recon
    if len(gp.y_data) != last_recon[0]:
        # mid = (adaptive.binary_high.value() + adaptive.binary_low.value())/2
        last_recon = len(gp.y_data), sirt(gp.y_data, last_A[1], num_iterations=30, initial=init)
        gp.last_recon = last_recon[1]
    return last_recon[1]


def variance(x, gp):
    x = x.reshape(-1, gp.input_dim)

    projections = [projection_operator(*x_i, l_det) for x_i in x]
    grid_positions = np.fliplr(
        image_grid(((0, l_det), (0, l_det)), (l_det, l_det)))  # use positions from gp here!!!!!!!

    variances = memoized_posterior_variance(grid_positions, gp)
    res_s = [projection.ravel() * variances for projection in projections]
    res = np.sum(res_s, axis=1)
    return res


# TODO: look at res_s image; check why diagonals are preferred


# def variance(x, gp):
#     x = x.reshape(-1, gp.input_dim)
#
#     projections = [projection_operator(*x_i, l_det) for x_i in x]
#     x_reals = [np.asarray(np.nonzero(projection)).T for projection in projections]  # Each x_real is a collection of real-space points translated from the input point in sinogram space
#     x_coeffs = [projection[np.nonzero(projection)] for projection, x_real in zip(projections, x_reals)]
#
#     res_s = np.asarray([gp.posterior_covariance(x_real, variance_only=True)["v(x)"] for x_real, x_coeff in zip(x_reals, x_coeffs)])
#     res = np.sum(res_s, axis=1)
#     return res


def ucb(x, gp):
    x = x.reshape(-1, gp.input_dim)

    projections = [projection_operator(*x_i, l_det) for x_i in x]
    x_reals = [np.asarray(np.nonzero(projection)).T for projection in
               projections]  # Each x_real is a collection of real-space points translated from the input point in sinogram space
    x_coeffs = [projection[np.nonzero(projection)] for projection, x_real in zip(projections, x_reals)]

    v_s = np.asarray([gp.posterior_covariance(x_real, variance_only=True)["v(x)"] / x_coeff for x_real, x_coeff in
                      zip(x_reals, x_coeffs)])
    m_s = np.asarray([gp.posterior_mean(x_real)["f(x)"] / x_coeff for x_real, x_coeff in
                      zip(x_reals, x_coeffs)])

    v = np.average(v_s, axis=1)
    m = np.average(m_s, axis=1)

    return m + 3.0 * np.sqrt(v)


def binary_resolve(x, gp):
    x = x.reshape(-1, gp.index_set_dim)

    projections = [projection_operator(*x_i, l_det) for x_i in x]
    grid_positions = np.fliplr(image_grid(((0, l_det), (0, l_det)), (l_det, l_det)))

    # np.diag(A^-1 @ gp.posterior_covariance('S') @ A^-1) <- projection operator reversing from a grid of points in sinogram space

    v_s = memoized_posterior_variance(grid_positions, gp)
    m_s = memoized_posterior_mean(grid_positions, gp)
    recon = memoized_recon(gp)#, m_s)

    #
    # v = np.array([projection.ravel() * v_s for projection in projections])

    # normalize m_s by projection length

    # d = np.average(np.abs(np.abs(m)-1), axis=1) + 3.0 * np.sqrt(v)
    low = -1 # ThresholdResolve.binary_low.value()
    high = 1 # ThresholdResolve.binary_high.value()
    width = high - low
    mid = width / 2 + low

    # binary_levels_distance = np.abs(np.abs(m_s - mid) - width/2)
    binary_levels_distance = np.abs(np.abs(m_s-mid)-width/2) * np.logical_and(low < recon, recon < high)  # TODO: memoize
    m = np.array([projection.ravel() * binary_levels_distance for projection in projections])
    # cumulative_measurements = np.array([projection.ravel() * np.sum(last_A[1], axis=0) for projection in projections])
    v = np.array([projection.ravel() * v_s for projection in projections])

    normalized_mean = m / np.max(m)
    average_binary_distance = np.sum(m, axis=1)
    average_v = np.sum(v * normalized_mean, axis=1)
    # average_cumulative_measurements = np.maximum(np.sum(cumulative_measurements, axis=1), .0000001)

    # wasserstein distance
    # kl divergence
    #
    # distance to levels -> imaging sharp features: LBL
    # weighted uncertainty -> imaging smooth sample : question mark
    # average gradient -> imaging high detail : 
    # optimization -> should find maxima quickly for smooth sample: gaussian peaks

    d = np.max(m_s) * np.max(v_s) #average_cumulative_measurements
    # d = average_v

    return d


def bilinear_sample(pos, data):
    # pos = np.random.random((2,)) * np.array([32, 180])
    # print(f'measuring: x={pos[1]:.1f} y={pos[0]:.1f}')
    return pos, ndimage.map_coordinates(data, [[pos[0]], [pos[1]*(360/180)]], order=1)[0], [.001] * n_sinograms, {}
    # A = projection_operator(*pos, l_x).reshape(l_x, l_x)
    # return pos, np.sum(data.T @ A), [.00001] * n_sinograms, {}


def push_to_queue(pos):
    logger.critical(f'publishing: {pos}')
    decision_queue.publish([{'session': session_id,
                             'position': pos,
                             'measured': False}])
    while True:
        measurements = decision_queue.get()
        if len(measurements) > 1:
            logger.critical(f'More than 1 point retrieved from decision queue: {measurements}')
        elif measurements[0]['session'] != session_id:
            logger.critical('Stale data received. Clearing Queue...')
        else:
            break

    # return measurements[0]['position'], measurements[0]['value'], .0001, {}
    return (0, 0), 1, .001, {}


class BackgroundTraining(GPCAMInProcessEngine):
    def __init__(self, start_training_at: int = 10, *args, **kwargs):
        self.training_thread = None
        self.start_training_at = start_training_at
        super().__init__(*args, **kwargs)

    def train(self):
        if not self.training_thread or self.training_thread.done:
            if len(self.optimizer.y_data) >= self.start_training_at:
                # pull values from optimizer
                # newstyle
                x, y, v, A_ = self.optimizer.x_data.copy(), self.optimizer.y_data.copy(), copy.deepcopy(core.data.variances), self.optimizer.A.copy()
                # oldstyle
                # x, y, v, A = self.optimizer.x_data.copy(), self.optimizer.y_data.copy(), self.optimizer.variances.copy(), self.optimizer.A.copy()

                # pull parameters
                hyperparameters_bounds = np.asarray([[self.parameters[('hyperparameters', f'hyperparameter_{i}_{edge}')]
                                                      for edge in ['min', 'max']]
                                                     for i in range(self.num_hyperparameters)])
                hyperparameters = np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                              for i in range(self.num_hyperparameters)])
                parameter_bounds = np.asarray([[self.parameters[('bounds', f'axis_{i}_{edge}')]
                                                for edge in ['min', 'max']]
                                               for i in range(self.dimensionality)])

                self.training_thread = threads.QThreadFuture(self._background_train,
                                                             x,
                                                             y,
                                                             v,
                                                             A_,
                                                             parameter_bounds,
                                                             hyperparameters,
                                                             hyperparameters_bounds,
                                                             {'method': 'mcmc',
                                                              'tolerance': 0.1,
                                                              'max_iter': 1000,
                                                              # 'pop_size': 20, # unused
                                                              })
                self.training_thread.start()

        return True

    def _background_train(self, x, y, v, A_, parameter_bounds, hyperparameters, hyperparameter_bounds, training_kwargs):
        logger.info('Training asynchronously...')
        opts = self.gp_opts.copy()
        opts['gp_kernel_function'] = partial(opts['gp_kernel_function'], A_=A_)
        opts['gp_mean_function'] = partial(opts['gp_mean_function'], A_=A_)
        opts['gp_noise_function'] = partial(opts['gp_noise_function'], variances=v)
        # newstyle
        optimizer = GPOptimizer(x,
                                y,
                                init_hyperparameters=hyperparameters,
                                **opts)

        # oldstyle
        # optimizer = GPOptimizer(self.dimensionality, parameter_bounds)
        # optimizer.tell(x, y)
        # from cProfile import Profile
        # from pstats import SortKey, Stats
        # with Profile() as profile:
        optimizer.train(hyperparameter_bounds=hyperparameter_bounds, init_hyperparameters=hyperparameters,
                        **training_kwargs)
        # Stats(profile).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()

        self.optimizer.hyperparameters = optimizer.hyperparameters
        self.optimizer.set_hyperparameters(optimizer.hyperparameters)
        logger.info(f'Hyperparameters set from asynchronous training: {optimizer.hyperparameters}')


class ProjectionOperatorBuilderGPCamEngine(BackgroundTraining):
    default_retrain_locally_at = tuple()
    default_retrain_globally_at = tuple()  # 82

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphs = [
            # GPCamPosteriorCovariance(),
            # SinogramSpaceGPCamAcquisitionFunction(shape=(32, 32), real_space_bounds=((0, l_x), (0, l_x))),
            GPCamAcquisitionFunction(),
            ReconstructionGraph(shape=(l_x, l_x)),
            RealSpacePosteriorMean(shape=(l_x, l_x)),
            RealSpacePosteriorVariance(shape=(l_x, l_x)),
            ReconHistogram(),
            ProjectionMask(shape=(l_x, l_x)),
            ProjectionOperatorGraph(shape=(l_x, l_x)),
            # GPCamAverageCovariance(),
            GPCamHyperparameterPlot(),
            Table(),
            # Variance(),
            Score()
        ]

    def update_measurements(self, data: Data):
        global A
        with data.r_lock():  # quickly grab values within lock before passing to optimizer
            positions = data.positions.copy()
            scores = data.scores.copy()
            variances = data.variances.copy()

        scores = np.array(scores).ravel()
        variances = np.array(variances).ravel()

        self.optimizer.A = memoized_A(positions)
        self.optimizer.positions = positions

        # length_diff = len(positions) - self._positions_seen
        # if length_diff:
        #     for position in positions[-length_diff:]:
        #         A_vec = projection_operator(*position, l_x).reshape(1, l_x ** 2)
        #         self.optimizer.A = A = np.vstack([A, A_vec])
        #         self._positions_seen += 1
        #         # self.optimizer.positions = np.vstack([self.optimizer.positions, position])

        self.optimizer.tell(np.asarray(real_space_positions), np.asarray(scores))  # , np.asarray(variances))

        # oldstyle
        # if not self.optimizer.gp_initialized:
        #     hyperparameters = np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
        #                                   for i in range(self.num_hyperparameters)])
        #     opts = self.gp_opts.copy()
        #     # TODO: only fallback to numpy when packaged as an app
        #     if sys.platform == 'darwin':
        #         opts['compute_device'] = 'numpy'
        #
        #     self.init_gp(hyperparameters, **opts)

    def reset(self):
        global last_A, last_variance, last_mean, last_recon
        super().reset()
        self.optimizer.A = np.zeros((0, l_x ** 2))
        last_A = -1, None
        last_variance = -1, None
        last_mean = -1, None
        last_recon = -1, None
        last_A = -1, None
        self.optimizer.positions = np.zeros((0, 2))
        self._positions_seen = 0



class EmptyOptimizer():
    pass


# class ProjectionOperatorGrid(Grid):
#     def __init__(self, *args, **kwargs):
#         self.optimizer = EmptyOptimizer()
#         super().__init__(*args, **kwargs)
#         self.graphs = [ReconstructionGraph(),
#                        Table(),
#                        Score()
#                        ]
#
#     def update_measurements(self, data: Data):
#         global A
#         with data.r_lock():  # quickly grab values within lock before passing to optimizer
#             positions = data.positions.copy()
#
#         self.optimizer.A = memoized_A(positions)
#         # length_diff = len(positions) - self._positions_seen
#         # if length_diff:
#         #     for position in positions[-length_diff:]:
#         #         A_vec = projection_operator(*position, l_x).reshape(1, l_x ** 2)
#         #         self.optimizer.A = A = np.vstack([A, A_vec])
#         #         self._positions_seen += 1
#
#     def update_metrics(self, data: Data):
#         for graph in self.graphs:
#             try:
#                 graph.compute(data, self)
#             except Exception as ex:
#                 logger.exception(ex)
#
#     def reset(self):
#         global last_A
#         super().reset()
#         self.optimizer.A = A = np.zeros((0, l_x ** 2))
#         self._positions_seen = 0


class GridFirst(ProjectionOperatorBuilderGPCamEngine):
    def __init__(self, grid_until_N: int, parameter_bounds, *args, **kwargs):
        self.grid_until_N = grid_until_N
        self.grid_engine = Grid(parameter_bounds=parameter_bounds)
        super().__init__(*args, parameter_bounds=parameter_bounds, **kwargs)

    def request_targets(self, position):
        if not hasattr(self.optimizer, 'y_data') or len(self.optimizer.y_data) < self.grid_until_N:
            return self.grid_engine.request_targets(position)
        else:
            return super().request_targets(position)

    def reset(self):
        super().reset()

        self.grid_engine.reset()


class SessionReset(GPCAMInProcessEngine):
    def reset(self):
        global session_id
        session_id = uuid4()
        super().reset()


class ThresholdResolve(GridFirst, SessionReset):
    binary_low_percentile = SimpleParameter(title='Posterior Binary Low Percentile', name='binary_low', type='float', value=10, units='%', min=0, max=100)
    binary_high_percentile = SimpleParameter(title='Posterior Binary High Percentile', name='binary_high', type='float', value=90, units='%', min=0, max=100)

    @cached_property
    def parameters(self):
        parameters = super().parameters
        parameters.insertChild(0, self.binary_low_percentile)
        parameters.insertChild(0, self.binary_high_percentile)
        return parameters
    
    def update_measurements(self, data: Data):
        with data.r_lock():
            y = data.scores.copy()

        if hasattr(self, '_last_rescale'):
            last_scale, last_offset, reverse_to = self._last_rescale
            y[:reverse_to] = (np.asarray(y[:reverse_to]) + 1) / last_scale + last_offset
        else:
            last_scale = 1
            last_offset = 1

        if self.optimizer.gp:
            posterior_mean = memoized_posterior_mean(real_space_positions, self.optimizer.gp).reshape(l_x, l_x)
            posterior_low = np.percentile(posterior_mean, self.binary_low_percentile.value())
            posterior_high = np.percentile(posterior_mean, self.binary_high_percentile.value())
        else:
            posterior_low = 0
            posterior_high = 1

        scale = last_scale / (posterior_high-posterior_low)*2
        offset = np.mean(y)


        if posterior_low < posterior_high:
            y = (np.asarray(y) - offset) * scale-1
            data.scores = list(y)
            self._last_rescale = scale, offset, len(data)

        if self.optimizer.gp:
            print('last_offset, offset:', last_offset, offset)
            print('last_scale, scale  :', last_scale, scale)
            print('posterior avg:', np.average(posterior_mean))
            print('posterior high:', np.percentile(posterior_mean, self.binary_high_percentile.value()))
            print('posterior low :', np.percentile(posterior_mean, self.binary_low_percentile.value()))

        super().update_measurements(data)

    def reset(self):
        super().reset()
        if hasattr(self, '_last_rescale'):
            del self._last_rescale


class NoPartialResponseCore(ZMQCore):
    def respond_PartialDataRequest(self, request):
        if self.data and request.iteration <= len(self.data) and self.state == CoreState.Running:
            with self.data.r_lock():
                return FullDataResponse(self.data.as_dict())
        else:
            return StateResponse(self.state, self.compute_metrics)


# #### FOR KANUPRIYA's ####
# geom2d = Geometry2d(n_angles, np.array([l_det, l_det]), np.ones(2, ), l_det, 1.0)
# #########################

f = 'Zebra_df_peaks_all_subbg0.csv'
df = pandas.read_csv(f)
data_sort = df.sort_values(by=['pos_x', 'pos_phi'])
sino = copy.deepcopy(data_sort['sum001']).values
sino = np.reshape(sino, (67, 361))  # 67 x positions (0:1:66), 361 phis (0:0.5:180)

l_det = l_x = sino.shape[0] # sampling for ground truth sinograms

l_x = sino.shape[0]  ##l_x times l_x is the size of the real space image
n_sinograms = 1


if __name__ == "__main__":
    dask_client = dask.distributed.Client()

    real_space_positions = np.mgrid[:l_x, :l_x].T.reshape(-1, 2)
    # decision_queue = Queue_decision()

    # angles = np.linspace(0.0, np.pi, n_angles)

    # the following line will replaced by starting data, a sinogram with the right shape but only measured data non-zero
    # gt_A = np.array([projection_operator(x, phi, l_x) for x in range(l_det) for phi in range(n_angles)]).reshape(l_det * n_angles, -1)
    # domain_sinograms = {}
    # for domain_angle, domain_map in domain_maps.items():
    #     domain_sinograms[domain_angle] = (gt_A @ domain_map.ravel()).reshape(l_det, n_angles).T

    # name = r'C:\data\gitomo\run2-checkpoint1.yml'
    # data = Data(**load(open(name, 'r'), Loader=Loader))
    # positions = np.asarray(data.positions)
    # x, y = positions.T
    execution = SimpleEngine(measure_func=partial(bilinear_sample, data=sino))

    # print(f'values: {min(data.scores), max(data.scores)}')

    # Define a gpCAM adaptive engine with initial parameters
    adaptive = ThresholdResolve(dimensionality=2,
                                grid_until_N=9,
                                parameter_bounds=[(0, l_x - 1),
                                                  # NOTE: THIS -1 is important to avoid the projection weight not becoming empty
                                                  (0, 180)],
                                hyperparameters=[1, 3],
                                hyperparameter_bounds=[[1e-2, 1e2],  # signal variance
                                                       [1e-2, 1e2]],  # lengthscale
                                gp_opts=dict(gp_kernel_function=kernel,
                                             gp_mean_function=mean_func,
                                             gp_noise_function=noise,  # newstyle
                                             ),
                                ask_opts=dict(dask_client=dask_client,
                                              pop_size=10,
                                              max_iter=10,
                                              method='hgdl',
                                              #vectorized=False
                                              ),
                                acquisition_functions={'Binary Resolve': binary_resolve,
                                                       'Projected Variance': variance,
                                                       'Projected UCB': ucb})

    # adaptive = ProjectionOperatorGrid(parameter_bounds=[(0, l_x), (0, 180)])

    # x_data = np.empty((l_x ** 2, 2))
    # y_data = np.zeros((n_angles * l_det))

    # for i in range(l_x):
    #     for j in range(l_x):
    #         x_data[i+j*l_x] = np.array([float(i), float(j)])

    # v_data = 0.01 * np.ones((n_angles * l_det))

    # Construct a core server
    core = NoPartialResponseCore()
    core.set_adaptive_engine(adaptive)
    core.set_execution_engine(execution)
    # core.initialize_data(x_data, y_data, v_data)

    # Start the core server
    core.state = CoreState.Starting
    core.main()
