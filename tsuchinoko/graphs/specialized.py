from dataclasses import dataclass

import numpy as np
from scipy import linalg

from tsuchinoko.graphics_items.mixins import YInvert, ClickRequester, BetterButtons, LogScaleIntensity, AspectRatioLock, \
    BetterAutoLUTRangeImageView, DomainROI
from tsuchinoko.graphs import Location
from tsuchinoko.graphs.common import Image, ImageViewBlendROI, image_grid


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
class DomainGraph(Image):
    compute_with = Location.AdaptiveEngine
    shape = (32, 32)
    data_key = 'Domains'
    widget_class = NonViridisBlend
    transform_to_parameter_space = False

    def compute(self, data, engine: 'GPCAMInProcessEngine'):

        # calculate domain maps
        self.last_recon = sirt(np.array(data.scores).T.ravel(),
                               linalg.block_diag(*[engine.optimizer.A] * 3),
                               num_iterations=1,
                               initial=getattr(self, 'last_recon', None))

        # assign to data object with lock
        with data.w_lock():
            data.states[self.data_key] = np.fliplr(self.last_recon.reshape(3, 32, 32).T)