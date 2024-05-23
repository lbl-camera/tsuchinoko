from dataclasses import dataclass
from functools import partial, lru_cache, cached_property
import sys
from loguru import logger
from pyqtgraph.parametertree.parameterTypes import SimpleParameter
from yaml import load, Loader

import cv2

import fvgp
from fvgp.gp import GP

import numpy as np
from scipy import sparse, ndimage, fftpack, linalg
from gpcam.gp_optimizer import GPOptimizer
from tsuchinoko.adaptive import Data
from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.adaptive.grid import Grid
from tsuchinoko.core import ZMQCore, CoreState
from tsuchinoko.execution.replay import InterpolatingEngine
from tsuchinoko.execution.simple import SimpleEngine
from tsuchinoko.graphs import Location
from tsuchinoko.graphs.common import Table, Image, ImageViewBlendROI, image_grid, GPCamPosteriorMean, \
    GPCamHyperparameterPlot, Score, Variance, GPCamAcquisitionFunction
from tsuchinoko.utils import threads
from tsuchinoko.graphs.specialized import ReconstructionGraph, ProjectionOperatorGraph, \
    SinogramSpaceGPCamAcquisitionFunction, ProjectionMask, ReconHistogram, sirt
import PIL

from tsuchinoko.utils.zmq_queue import Queue_decision
from uuid import uuid4


session_id = uuid4()


def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

# def weighted_line(y0, x0, y1, x1):
#     line = np.zeros((l_det, l_det), dtype=np.uint8)
#     cv2.line(line, (int(x0), int(y0)), (int(x1), int(y1)), (255,), lineType=cv2.LINE_AA)
#     return line/255.


def projection_operator(x, phi, map_size, center=None, width=1, length=None):
    if not length:
        length = map_size * 2

    if not center:
        center = (map_size/2, map_size/2)

    # get a sampling over a slice through x, phi
    v = np.array([-np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))]) * (-x + map_size/2)
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
    # map_mask = weighted_line(y0,x0,y1,x1)

    return map_mask


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
    for i in range(x1.shape[1]): d += (x1[:, i].reshape(-1, 1) - x2[:, i]) ** 2
    return np.sqrt(d)


def kernel(x1,x2,hps,obj):
    # print('kernel:', hps, x1, x2)
    # with log_time('kernel', cumulative_key='kernel'):
    A = obj.A
    #this kernel takes elements from the real space the prior covariance in sinogram space
    # newstyle
    # d = obj.get_distance_matrix(x1, x2)
    # oldstyle
    d = get_distance_matrix(x1, x2)
    k = hps[0] * np.exp(-d**2/hps[1])
    if len(x1) == len(x2) == len(obj.x_data):  # training
        kernel = linalg.block_diag(*[A @ k @ A.T]*n_sinograms)
        # kernel = A @ k @ A.T
    elif len(x1) == len(x2) != len(obj.x_data):  # prediction
        kernel = k
    elif len(x1) != len(x2):  # prediction
        kernel = np.repeat(A @ k, n_sinograms, 0)
        # kernel = A @ k
    return kernel


def noise(x,hps,obj):
    A = obj.A
    #this function takes elements from the real space and returns noise (co)variances in sinogram space
    n = A @ (np.zeros((len(x))) + 0.01)
    return np.diag(n)

# newstyle
# def mean_func(x,hps,obj):
# oldstyle
def mean_func(obj, x, hps):
    A = obj.A
    #this function takes elements from the real space and returns prior mean values in sinogram space
    # m = A @ np.zeros(len(x),)
    if len(x) == len(obj.x_data):
        m = np.repeat(A @ np.zeros(len(x)), n_sinograms)
    else:
        m = np.zeros((len(x),))
    return m


def cost(origin, x, arguments=None):
    print('cost:', origin, x)
    cost_x = 1 / 8  # cost is 1/velocity
    cost_phi = 1 / 30
    exposure_time = 1
    estimated_total_experiment_time = 60*60*1  # in seconds
    x_origin, phi_origin = origin
    cost = [(exposure_time + max(abs(ix - x_origin) * cost_x, abs(iphi - phi_origin) * cost_phi))/estimated_total_experiment_time for
            ix, iphi in x]
    return cost


last_variance = (-1, None)
last_mean = (-1, None)
last_recon = (-1, None)


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


def memoized_recon(gp):
    global last_recon
    if len(gp.y_data) != last_recon[0]:
        last_recon = len(gp.y_data), sirt(gp.y_data, gp.A, num_iterations=30)
        gp.last_recon = last_recon[1]
    return last_recon[1]


def variance(x, gp):
    x = x.reshape(-1, gp.input_dim)

    projections = [projection_operator(*x_i, l_det) for x_i in x]
    grid_positions = np.fliplr(image_grid(((0, l_det), (0, l_det)), (l_det, l_det)))  # use positions from gp here!!!!!!!

    variances = memoized_posterior_variance(grid_positions[:-1], gp)
    res_s = [projection.ravel()[:-1] * variances for projection in projections]
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
    x = x.reshape(-1, gp.input_dim)

    projections = [projection_operator(*x_i, l_det) for x_i in x]
    grid_positions = np.fliplr(image_grid(((0, l_det), (0, l_det)), (l_det, l_det)))

    v_s = memoized_posterior_variance(grid_positions[:-1], gp)
    m_s = memoized_posterior_mean(grid_positions[:-1], gp)
    recon = memoized_recon(gp)

    v = np.array([projection.ravel()[:-1] * v_s for projection in projections])

    # normalize m_s by projection length

    # d = np.average(np.abs(np.abs(m)-1), axis=1) + 3.0 * np.sqrt(v)
    low = ThresholdResolve.binary_low.value()
    high = ThresholdResolve.binary_high.value()
    mid = (high-low)/2 + low
    width = high-mid

    binary_levels_distance = np.abs(np.abs(m_s - mid) - width)
    m = np.array([projection.ravel()[:-1] * binary_levels_distance for projection in projections])
    cumulative_measurements = np.array([projection.ravel() * np.sum(gp.A, axis=0) for projection in projections])

    average_binary_distance = np.average(m, axis=1)
    average_cumulative_measurements = np.maximum(np.average(cumulative_measurements, axis=1), .0000001)


    d = average_binary_distance / average_cumulative_measurements

    return d


def bilinear_sample(pos, data):
    # pos = np.random.random((2,)) * np.array([32, 180])
    # print(f'measuring: x={pos[1]:.1f} y={pos[0]:.1f}')
    # return pos, [ndimage.map_coordinates(sino, [[pos[1]], [pos[0]]], order=1)[0] for sino in data.values()], [.001] * n_sinograms, {}
    A = projection_operator(*pos, l_x).reshape(l_x, l_x)
    return pos, [np.sum(map * A) for map in data.values()], [.00001] * n_sinograms, {}


def push_to_queue(pos):
    logger.critical(f'publishing: {pos}')
    decision_queue.publish([{'session': session_id, 'position': pos}])
    while True:
        measurement = decision_queue.get()
        if measurement[0]['session'] != session_id:
            logger.critical('Stale data received. Clearing Queue...')
        else:
            break
    return measurement[0]['value']


class BackgroundTraining(GPCAMInProcessEngine):
    def __init__(self, start_training_at:int = 40, *args, **kwargs):
        self.training_thread = None
        self.start_training_at = start_training_at
        super().__init__(*args, **kwargs)

    def train(self):
        if not self.training_thread or self.training_thread.done:
            if len(self.optimizer.y_data) >= self.start_training_at:
                # pull values from optimizer
                # newstyle
                # x, y, v, A = self.optimizer.x_data.copy(), self.optimizer.y_data.copy(), self.optimizer.V.copy(), self.optimizer.A.copy()
                # oldstyle
                x, y, v, A = self.optimizer.x_data.copy(), self.optimizer.y_data.copy(), self.optimizer.variances.copy(), self.optimizer.A.copy()

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
                                                             A,
                                                             parameter_bounds,
                                                             hyperparameters,
                                                             hyperparameters_bounds,
                                                             {'method': 'global'})
                self.training_thread.start()

        return True

    def _background_train(self, x, y, v, A, parameter_bounds, hyperparameters, hyperparameter_bounds, training_kwargs):
        logger.info('Training asynchronously...')
        # newstyle
        # optimizer = GPOptimizer(x,
        #                         y,
        #                         noise_variances=v,
        #                         init_hyperparameters=hyperparameters,
        #                         hyperparameter_bounds=hyperparameter_bounds,
        #                         **self.gp_opts.copy())

        # oldstyle
        optimizer = GPOptimizer(self.dimensionality, parameter_bounds)
        optimizer.A = A.copy()
        optimizer.tell(x, y, v)
        opts = self.gp_opts.copy()
        optimizer.init_gp(init_hyperparameters=hyperparameters, **opts)
        optimizer.train(hyperparameter_bounds=hyperparameter_bounds, init_hyperparameters=hyperparameters, **training_kwargs)

        self.optimizer.hyperparameters = optimizer.hyperparameters
        logger.info(f'Hyperparameters set from asynchronous training: {optimizer.hyperparameters}')


class ProjectionOperatorBuilderGPCamEngine(BackgroundTraining):
    default_retrain_locally_at = tuple()
    default_retrain_globally_at = tuple() #82

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphs = [
            # GPCamPosteriorCovariance(),
            #            SinogramSpaceGPCamAcquisitionFunction(shape=(32, 32), real_space_bounds=((0, l_x), (0, l_x))),
                       ReconstructionGraph(shape=(l_x, l_x)),
                       ReconHistogram(),
                       ProjectionMask(shape=(l_x, l_x)),
                       ProjectionOperatorGraph(),
                       # GPCamAverageCovariance(),
                       GPCamHyperparameterPlot(),
                       Table(),
                       # Variance(),
                       Score()
        ]

    def init_optimizer(self):
        opts = self.gp_opts.copy()
        # TODO: only fallback to numpy when packaged as an app
        if sys.platform == 'darwin':
            opts['compute_device'] = 'numpy'




        # oldstyle
        parameter_bounds = np.asarray([[self.parameters[('bounds', f'axis_{i}_{edge}')]
                                        for edge in ['min', 'max']]
                                       for i in range(self.dimensionality)])
        self.optimizer = GPOptimizer(self.dimensionality, parameter_bounds)

        # newstyle
        # real_space_positions = np.mgrid[:l_x, :l_x].T.reshape(-1, 2) # swap in full real-space grid
        # self.optimizer = GPOptimizer(real_space_positions,
        #                              np.random.rand(2),
        #                              #noise_variances=np.ones((2,))*.01,
        #                              **opts)  # give hyperparameters!!!!!!!!!!!!!!!!!

    def update_measurements(self, data: Data):
        with data.r_lock():  # quickly grab values within lock before passing to optimizer
            positions = data.positions.copy()
            scores = data.scores.copy()
            variances = data.variances.copy()

        real_space_positions = np.mgrid[:l_x,:l_x].T.reshape(-1, 2)
        scores = np.array(scores).ravel()
        variances = np.array(variances).ravel()

        length_diff = len(positions) - self._positions_seen
        if length_diff:
            for position in positions[-length_diff:]:
                A_vec = projection_operator(*position, l_x).reshape(1, l_x **2)
                self.optimizer.A = np.vstack([self.optimizer.A, A_vec])
                self._positions_seen += 1
                # self.optimizer.positions = np.vstack([self.optimizer.positions, position])

        self.optimizer.tell(np.asarray(real_space_positions), np.asarray(scores), np.asarray(variances))

        # oldstyle
        if not self.optimizer.gp_initialized:
            hyperparameters = np.asarray([self.parameters[('hyperparameters', f'hyperparameter_{i}')]
                                          for i in range(self.num_hyperparameters)])
            opts = self.gp_opts.copy()
            # TODO: only fallback to numpy when packaged as an app
            if sys.platform == 'darwin':
                opts['compute_device'] = 'numpy'

            self.init_gp(hyperparameters, **opts)

    def reset(self):
        super().reset()
        self.optimizer.A = np.zeros((0, l_x ** 2))
        self.optimizer.positions = np.zeros((0, 2))
        self._positions_seen = 0
        if self.graphs:
            self.graphs[0].last_recon = None


class EmptyOptimizer():
    pass


class ProjectionOperatorGrid(Grid):
    def __init__(self, *args, **kwargs):
        self.optimizer = EmptyOptimizer()
        super().__init__(*args, **kwargs)
        self.graphs = [ReconstructionGraph(),
                       Table(),
                       Score()
                       ]

    def update_measurements(self, data: Data):
        with data.r_lock():  # quickly grab values within lock before passing to optimizer
            positions = data.positions.copy()

        length_diff = len(positions) - self._positions_seen
        if length_diff:
            for position in positions[-length_diff:]:
                A_vec = projection_operator(*position, l_x).reshape(1, l_x **2)
                self.optimizer.A = np.vstack([self.optimizer.A, A_vec])
                self._positions_seen += 1

    def update_metrics(self, data: Data):
        for graph in self.graphs:
            try:
                graph.compute(data, self)
            except Exception as ex:
                logger.exception(ex)

    def reset(self):
        super().reset()
        self.optimizer.A = np.zeros((0, l_x ** 2))
        self._positions_seen = 0
        if self.graphs:
            self.graphs[0].last_recon = None


class GridFirst(ProjectionOperatorBuilderGPCamEngine):
    def __init__(self, grid_until_N:int, parameter_bounds, *args, **kwargs):
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
    binary_low = SimpleParameter(title='Binary Low', name='binary_low', type='float', value=-1)
    binary_high = SimpleParameter(title='Binary High', name='binary_high', type='float', value=1)
    @cached_property
    def parameters(self):
        parameters = super().parameters
        parameters.insertChild(0, self.binary_low)
        parameters.insertChild(0, self.binary_high)
        return parameters

n_angles = 180  # number of angles
l_det = l_x = 32  # sampling for ground truth sinograms

# #### FOR KANUPRIYA's ####
# geom2d = Geometry2d(n_angles, np.array([l_det, l_det]), np.ones(2, ), l_det, 1.0)
# #########################

fn = "phantom.png"
image = PIL.Image.open(fn).convert('L')

#convert to angle (180) map; modulo wraps 255 back to 0
map = (np.asarray(image) / 255 * n_angles).astype(int) % n_angles

# print unique domain angles
print('angles:', np.unique(map))
l_x = map.shape[0]  ##l_x times l_x is the size of the real space image

domain_maps = {angle: (map == angle)*2-1 for angle in np.unique(map) if angle == 139}
# n_sinograms = len(domain_maps)
n_sinograms = 1

if __name__ == "__main__":
    decision_queue = Queue_decision()

    # angles = np.linspace(0.0, np.pi, n_angles)

    #the following line will replaced by starting data, a sinogram with the right shape but only measured data non-zero
    # gt_A = np.array([projection_operator(x, phi, l_x) for x in range(l_det) for phi in range(n_angles)]).reshape(l_det * n_angles, -1)
    # domain_sinograms = {}
    # for domain_angle, domain_map in domain_maps.items():
    #     domain_sinograms[domain_angle] = (gt_A @ domain_map.ravel()).reshape(l_det, n_angles).T

    # name = r'C:\data\gitomo\run2-checkpoint1.yml'
    # data = Data(**load(open(name, 'r'), Loader=Loader))
    # positions = np.asarray(data.positions)
    # x, y = positions.T
    execution = SimpleEngine(measure_func=push_to_queue)

    # print(f'values: {min(data.scores), max(data.scores)}')

    # Define a gpCAM adaptive engine with initial parameters
    adaptive = ThresholdResolve(dimensionality=2,
                         grid_until_N=4,
                         parameter_bounds=[(0, l_x-1),  # NOTE: THIS -1 is important to avoid the projection weight not becoming empty
                                           (0, 180)],
                         hyperparameters=[1, 2],
                         hyperparameter_bounds=[[.1, 1e3],  # signal variance
                                                [.1, 1e4]],  # lengthscale
                         gp_opts=dict(gp_kernel_function=kernel,
                                      gp_mean_function=mean_func,
                                      #gp_noise_function=noise, # newstyle
                                      ),
                         # ask_opts=dict(vectorized=False),
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
    core = ZMQCore()
    core.set_adaptive_engine(adaptive)
    core.set_execution_engine(execution)
    # core.initialize_data(x_data, y_data, v_data)

    # Start the core server
    core.state = CoreState.Starting
    core.main()
