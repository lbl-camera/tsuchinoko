from dataclasses import dataclass
from functools import partial

import numpy as np
from loguru import logger
import numba as nb
from scipy import linalg, sparse
from scipy.ndimage import median_filter

from tsuchinoko.graphics_items.mixins import YInvert, ClickRequester, BetterButtons, LogScaleIntensity, AspectRatioLock, \
    BetterAutoLUTRangeImageView, DomainROI
from tsuchinoko.graphs import Location
from tsuchinoko.graphs.common import Image, ImageViewBlendROI, image_grid, ImageViewBlend, Plot, Bar


class NonViridisBlend(YInvert,
                      ClickRequester,
                      BetterButtons,
                      AspectRatioLock,
                      BetterAutoLUTRangeImageView,
                      # DomainROI,
                      ):
    pass


#@nb.jit
def distance_from_line(x1, x2, pointsx, pointsy):
    """
    Calculate the perpendicular distance from an array of points to a line
    defined by two points, x1 and x2.

    Args:
        points (numpy.ndarray): An Nx2 array of points where each row is a point [p1, p2].
        x1 (numpy.ndarray): A 1x2 array representing the first point on the line [x1_1, x1_2].
        x2 (numpy.ndarray): A 1x2 array representing the second point on the line [x2_1, x2_2].

    Returns:
        numpy.ndarray: An N-element array where each element is the distance of a point to the line.
    """
    points = np.array([pointsx, pointsy]).T
    # Vector from x1 to x2 (direction of the line)
    line_vec = x2 - x1

    # Vectors from x1 to each point
    points_vec = points - x1

    # Cross product of line_vec and points_vec (only z component in 2D)
    cross_prod = np.abs(line_vec[0] * points_vec[:, :, 1] - line_vec[1] * points_vec[:, :, 0])

    # Magnitude of the line vector (length of the line)
    line_mag = np.linalg.norm(line_vec)

    # Distance from points to line (perpendicular distance)
    distances = cross_prod / line_mag

    return distances

def distance_from_line_array(p1, p2, shape):
    return np.fromfunction(partial(distance_from_line,p1,p2), shape)


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

    distance = distance_from_line_array(f0-.5, f1-.5, (map_size, map_size))
    projection = np.clip(width-distance, 0, None)

    return projection

@nb.jit()#nb.types.float32[:],(nb.types.float32[:],
                            # nb.types.float32[:, :],
                            # nb.types.optional(nb.types.int64),
                            # nb.types.optional(nb.types.float32[:, :]),
                            # nb.types.optional(nb.types.float32[:])))
def sirt(sinogram, projection_operator, num_iterations=10, inverse_operator=None, initial=None):
    R = np.diag(1 / np.sum(projection_operator, axis=1, dtype=np.float32))
    R = np.nan_to_num(R)
    C = np.diag(1 / np.sum(projection_operator, axis=0, dtype=np.float32))
    C = np.nan_to_num(C)

    if initial is None:
        x_rec = np.zeros(projection_operator.shape[1], dtype=np.float32)
    else:
        x_rec = initial

    for _ in range(num_iterations):
        if inverse_operator is not None:
            x_rec += C @ (inverse_operator @ (R @ (sinogram - projection_operator @ x_rec)))
        else:
            x_rec += C @ (projection_operator.T @ (R @ (sinogram - projection_operator @ x_rec)))

    return x_rec


@dataclass(eq=False)
class ReconstructionGraph(Image):
    compute_with = Location.AdaptiveEngine
    shape: tuple = (32, 32)
    data_key = 'Reconstruction'
    widget_class = NonViridisBlend
    transform_to_parameter_space = False

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        scores = np.array(data.scores)

        try:
            num_sinograms = len(scores[0])
        except TypeError:
            num_sinograms = 1

        # calculate domain maps
        # self.last_recon = sirt(scores.T.ravel(),
        #                        linalg.block_diag(*[engine.optimizer.A] * num_sinograms),
        #                        num_iterations=1,
        #                        initial=getattr(self, 'last_recon', None))
        try:
            last_recon = getattr(engine.optimizer, 'last_recon', None)
        except Exception:
            last_recon = None

        # assign to data object with lock
        if last_recon is not None:
            with data.w_lock():
                data.states[self.data_key] = np.rot90(last_recon.reshape(*self.shape))


@dataclass(eq=False)
class ProjectionMask(Image):
    compute_with = Location.AdaptiveEngine
    shape: tuple = (32, 32)
    data_key = 'Projection Mask'
    widget_class = NonViridisBlend
    transform_to_parameter_space = False

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = np.rot90(projection_operator(*data.positions[-1], self.shape[0]), 3)



@dataclass(eq=False)
class ProjectionOperatorGraph(Image):
    compute_with = Location.AdaptiveEngine
    shape: tuple = (32, 32)
    data_key = 'Projection Operator'
    # widget_class = NonViridisBlend
    transform_to_parameter_space = False

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        if not hasattr(self, 'A'):
            self.A = np.empty((0,np.prod(self.shape)))
            self._positions_seen = 0

        length_diff = len(data) - self._positions_seen
        if length_diff:
            for position in data.positions[-length_diff:]:
                A_vec = projection_operator(*position, self.shape[0]).reshape(1, self.shape[0] ** 2)
                self.A = np.vstack([self.A, A_vec])
                self._positions_seen += 1

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = np.rot90(np.sum(self.A, axis=0).reshape(*self.shape),3)


@dataclass(eq=False)
class SinogramSpaceGPCamAcquisitionFunction(Image):
    compute_with = Location.AdaptiveEngine
    shape: tuple = (50, 50)
    data_key = 'Acquisition Function'
    widget_class = ImageViewBlend
    real_space_bounds: tuple = (32, 32)
    transform_to_parameter_space = False

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        # if len(engine.optimizer.y_data) % 10:  # only compute every 10th measurement
        #     return
        from tsuchinoko.adaptive.gpCAM_in_process import gpcam_acquisition_functions  # avoid circular import

        grid_positions = image_grid(self.real_space_bounds, self.shape)

        # check if acquisition function is accessible
        if engine.parameters['acquisition_function'] not in gpcam_acquisition_functions:
            logger.exception(ValueError('The selected acquisition_function is not available for display.'))
            return

        # calculate acquisition function
        grid_positions = [grid_positions[:len(grid_positions)//2], grid_positions[len(grid_positions)//2:]]
        acquisition_function_value = np.hstack([engine.optimizer.evaluate_acquisition_function(p,
                                                                                    acquisition_function=
                                                                                    gpcam_acquisition_functions[
                                                                                        engine.parameters[
                                                                                            'acquisition_function']],
                                                                                    origin=engine.last_position) for p in grid_positions])

        try:
            acquisition_function_value = acquisition_function_value.reshape(*self.shape)
        except (ValueError, AttributeError):
            acquisition_function_value = np.array([[0]])

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = acquisition_function_value


@dataclass(eq=False)
class ReconHistogram(Bar):
    compute_with = Location.AdaptiveEngine
    data_key = 'Recon Histogram'
    name = "Recon Histogram"

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        scores = np.array(data.scores)

        try:
            num_sinograms = len(scores[0])
        except TypeError:
            num_sinograms = 1

        # calculate domain maps
        # self.last_recon = sirt(scores.T.ravel(),
        #                        linalg.block_diag(*[engine.optimizer.A] * num_sinograms),
        #                        num_iterations=1,
        #                        initial=getattr(self, 'last_recon', None))
        try:
            last_recon = getattr(engine.optimizer, 'last_recon', None)
        except Exception:
            last_recon = None

        if last_recon is not None:
            # calculate histogram
            y, x = np.histogram(last_recon, bins=100)

            # assign to data object with lock
            with data.w_lock():
                data.states[self.data_key] = [y, x]


@dataclass(eq=False)
class RealSpacePosteriorMean(Image):
    compute_with = Location.AdaptiveEngine
    shape:tuple = (50, 50)
    data_key = 'Posterior Mean'
    widget_class = ImageViewBlend
    transform_to_parameter_space = False

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        bounds = tuple(tuple(engine.parameters[('bounds', f'axis_{i}_{edge}')]
                   for edge in ['min', 'max'])
                  for i in range(engine.dimensionality))

        grid_positions = image_grid((bounds[0], bounds[0]), self.shape)
        shape = self.shape

        # if multi-task, extend the grid_positions to include the task dimension
        if hasattr(engine, 'output_number'):
            grid_positions = np.vstack([np.hstack([grid_positions, np.full((grid_positions.shape[0], 1), i)]) for i in range(engine.output_number)])
            shape = (*self.shape, engine.output_number)

        # calculate acquisition function
        posterior_mean_value = np.rot90(np.fliplr(engine.optimizer.posterior_mean(grid_positions)['f(x)'].reshape(*shape)),3)

        # assign to data object with lock
        with data.w_lock():
            data.states['Posterior Mean'] = posterior_mean_value


@dataclass(eq=False)
class RealSpacePosteriorVariance(Image):
    compute_with = Location.AdaptiveEngine
    shape:tuple = (50, 50)
    data_key = 'Posterior Variance'
    widget_class = ImageViewBlend
    transform_to_parameter_space = False

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        bounds = tuple(tuple(engine.parameters[('bounds', f'axis_{i}_{edge}')]
                   for edge in ['min', 'max'])
                  for i in range(engine.dimensionality))

        grid_positions = image_grid((bounds[0], bounds[0]), self.shape)
        shape = self.shape

        # if multi-task, extend the grid_positions to include the task dimension
        if hasattr(engine, 'output_number'):
            grid_positions = np.vstack([np.hstack([grid_positions, np.full((grid_positions.shape[0], 1), i)]) for i in range(engine.output_number)])
            shape = (*self.shape, engine.output_number)

        # calculate acquisition function
        posterior_variance_value = np.rot90(np.fliplr(engine.optimizer.posterior_covariance(grid_positions)['v(x)'].reshape(*shape)),3)

        # assign to data object with lock
        with data.w_lock():
            data.states['Posterior Variance'] = posterior_variance_value


@dataclass(eq=False)
class InvertedRealSpacePosteriorMean(Image):
    compute_with = Location.AdaptiveEngine
    shape:tuple = (64, 180)
    data_key = 'Posterior Mean'
    widget_class = ImageViewBlend
    transform_to_parameter_space = False
    A_inv:np.array = None

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        bounds = tuple(tuple(engine.parameters[('bounds', f'axis_{i}_{edge}')]
                   for edge in ['min', 'max'])
                  for i in range(engine.dimensionality))

        grid_positions = image_grid(((0,self.shape[0]), (0, 180)), self.shape)
        shape = self.shape

        # if multi-task, extend the grid_positions to include the task dimension
        if hasattr(engine, 'output_number'):
            grid_positions = np.vstack([np.hstack([grid_positions, np.full((grid_positions.shape[0], 1), i)]) for i in range(engine.output_number)])
            shape = (*self.shape, engine.output_number)

        # calculate posterior_mean
        posterior_mean = engine.optimizer.posterior_mean(grid_positions)['f(x)']

        # invert posterior_mean
        real_space_posterior_mean = np.rot90((self.A_inv @ posterior_mean).reshape(shape[0], shape[0]), 3)

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = real_space_posterior_mean


@dataclass(eq=False)
class InvertedRealSpacePosteriorVariance(Image):
    compute_with = Location.AdaptiveEngine
    shape:tuple = (64, 180)
    data_key = 'Posterior Variance'
    widget_class = ImageViewBlend
    transform_to_parameter_space = False
    A_inv:np.array = None

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        bounds = tuple(tuple(engine.parameters[('bounds', f'axis_{i}_{edge}')]
                   for edge in ['min', 'max'])
                  for i in range(engine.dimensionality))

        grid_positions = image_grid(((0,self.shape[0]), (0, 180)), self.shape)
        shape = self.shape

        # if multi-task, extend the grid_positions to include the task dimension
        if hasattr(engine, 'output_number'):
            grid_positions = np.vstack([np.hstack([grid_positions, np.full((grid_positions.shape[0], 1), i)]) for i in range(engine.output_number)])
            shape = (*self.shape, engine.output_number)

        # calculate posterior_ variance
        posterior_variance = engine.optimizer.posterior_covariance(grid_positions)['v(x)']

        # invert posterior_variance
        real_space_posterior_variance = np.rot90((self.A_inv @ posterior_variance).reshape(shape[0], shape[0]), 3)

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = real_space_posterior_variance

@dataclass(eq=False)
class InvertedSinoSpacePosteriorMean(Image):
    compute_with = Location.AdaptiveEngine
    shape:tuple = (64, 180)
    data_key = 'Sino Posterior Mean'
    widget_class = ImageViewBlend
    transform_to_parameter_space = False
    upsampling: int = 3

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        grid_shape = self.shape[0] * self.upsampling, self.shape[1] + 1
        grid_positions = image_grid(((0,self.shape[0]), (0, self.shape[1])), grid_shape)

        # # if multi-task, extend the grid_positions to include the task dimension
        # if hasattr(engine, 'output_number'):
        #     grid_positions = np.vstack([np.hstack([grid_positions, np.full((grid_positions.shape[0], 1), i)]) for i in range(engine.output_number)])
        #     shape = (*self.shape, engine.output_number)

        # calculate posterior_mean
        posterior_mean = engine.optimizer.posterior_mean(grid_positions)['f(x)'].reshape(*grid_shape)

        # invert posterior_mean
        real_space_posterior_mean = posterior_mean

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = real_space_posterior_mean


@dataclass(eq=False)
class InvertedSinoSpacePosteriorVariance(Image):
    compute_with = Location.AdaptiveEngine
    shape:tuple = (64, 180)
    data_key = 'Sino Posterior Variance'
    widget_class = ImageViewBlend
    transform_to_parameter_space = False
    upsampling: int = 3

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        grid_shape = self.shape[0] * self.upsampling, self.shape[1] + 1
        grid_positions = image_grid(((0,self.shape[0]), (0, self.shape[1])), grid_shape)

        # # if multi-task, extend the grid_positions to include the task dimension
        # if hasattr(engine, 'output_number'):
        #     grid_positions = np.vstack([np.hstack([grid_positions, np.full((grid_positions.shape[0], 1), i)]) for i in range(engine.output_number)])
        #     shape = (*self.shape, engine.output_number)

        # calculate posterior_mean
        posterior_variance = engine.optimizer.posterior_covariance(grid_positions)['v(x)'].reshape(*grid_shape)

        # invert posterior_mean
        real_space_posterior_mean = posterior_variance.reshape(self.shape)

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = real_space_posterior_mean

import tomopy


@dataclass(eq=False)
class InvertedRecon(Image):
    compute_with = Location.AdaptiveEngine
    shape:tuple = (64, 180)
    data_key = 'Reconstruction'
    widget_class = ImageViewBlend
    transform_to_parameter_space = False
    upsampling:int = 3
    algorithm:str = 'art'
    start_finding_center_at: int = 500
    find_center_every: int = 100

    def compute(self, data, engine: 'GPCAMInProcessEngine'):
        grid_shape = self.shape[0] * self.upsampling, self.shape[1] + 1
        grid_positions = image_grid(((0,self.shape[0]), (0, self.shape[1])), grid_shape)
        # shape = self.shape
        #
        # # if multi-task, extend the grid_positions to include the task dimension
        # if hasattr(engine, 'output_number'):
        #     grid_positions = np.vstack([np.hstack([grid_positions, np.full((grid_positions.shape[0], 1), i)]) for i in range(engine.output_number)])
        #     shape = (*self.shape, engine.output_number)

        # calculate posterior_mean
        sinogram = engine.optimizer.posterior_mean(grid_positions)['f(x)'].reshape(*grid_shape)

        # filter and clip
        sinogram = median_filter(sinogram, 3)
        sinogram = np.clip(sinogram, 3000, None)

        # reconstruct
        theta = np.deg2rad(np.linspace(0, self.shape[1], self.shape[1] + 1))
        last_center_found = data.states.get('recon_last_center_found', 0)
        if 'recon_center' in data.states:
            center = data.states.get('recon_center')
        elif len(data) > self.start_finding_center_at and len(data) - last_center_found > self.find_center_every:
            center = tomopy.find_center(sinogram.T[np.newaxis, :, :],
                                        theta,
                                        init=self.shape[0] * self.upsampling / 2,
                                        ind=0,
                                        tol=0.01,
                                        sinogram_order=True)
            data.states['recon_last_center_found'] = len(data)
        else:
            center = self.shape[0] / 2
        recon = tomopy.recon(sinogram.T[np.newaxis, :, :],
                             theta,
                             center=center * self.upsampling,
                             algorithm=self.algorithm,
                             sinogram_order=True)

        # mask
        recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = np.rot90(recon.reshape(self.shape[0]*self.upsampling,
                                                                self.shape[0]*self.upsampling), 3)