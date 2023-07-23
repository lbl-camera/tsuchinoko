import numpy as np
import pyqtgraph
from pyqtgraph import ImageView


# Fix for reaching last frame in ImageView timeline
class ImageView(ImageView):
    def timeIndex(self, slider):
        ## Return the time and frame index indicated by a slider
        if self.image is None:
            return (0, 0)

        t = slider.value()

        if not hasattr(self, "tVals"):
            return (0, 0)

        xv = self.tVals
        if xv is None:
            ind = int(t)
        else:
            if len(xv) < 2:
                return (0, 0)
            inds = np.argwhere(xv <= t)  # <- The = is import to reach the last value
            if len(inds) < 1:
                return (0, t)
            ind = inds[-1, 0]
        return ind, t

    def setCurrentIndex(self, ind):
        super(ImageView, self).setCurrentIndex(ind)
        (ind, time) = self.timeIndex(self.timeLine)
        self.sigTimeChanged.emit(ind, time)


pyqtgraph.__dict__["ImageView"] = ImageView


