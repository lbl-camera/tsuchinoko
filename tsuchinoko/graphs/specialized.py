from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy import linalg, sparse

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
        if inverse_operator:
            x_rec += C @ (inverse_operator @ (R @ (sinogram.ravel() - projection_operator @ x_rec)))
        else:
            x_rec += C @ (projection_operator.T @ (R @ (sinogram.ravel() - projection_operator @ x_rec)))

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
            data.states[self.data_key] = np.flipud(np.rot90(engine.optimizer.A[-1].reshape(*self.shape),1))


@dataclass(eq=False)
class ProjectionOperatorGraph(Image):
    compute_with = Location.AdaptiveEngine
    shape: tuple = (32, 32)
    data_key = 'Projection Operator'
    # widget_class = NonViridisBlend
    transform_to_parameter_space = False

    def compute(self, data, engine: 'GPCAMInProcessEngine'):

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = np.flipud(np.rot90(np.sum(engine.optimizer.A, axis=0).reshape(*self.shape),1))


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