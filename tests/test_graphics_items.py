import numpy as np
import pyqtgraph as pg
import scipy.misc

from tsuchinoko.graphics_items.clouditem import CloudItem


def test_cloud(qtbot, monkeypatch):
    pg.setConfigOption('useOpenGL', True)
    pg.setConfigOption('enableExperimental', True)

    # Create window with GraphicsView widget
    win = pg.GraphicsLayoutWidget()
    win.show()  # show widget alone in its own window
    win.setWindowTitle('CloudItem Example')
    view = win.addViewBox()

    # image = np.asarray(Image.open('test.jpeg'))
    image = scipy.misc.ascent()
    x, y = np.random.random((2, 10000))
    x *= image.shape[1]
    y *= image.shape[0]
    c = [np.average(image[-int(yi), int(xi)]) for xi, yi in zip(x, y)]
    x, y = list(x), list(y)

    cloud = CloudItem(size=1)
    histlut = pg.HistogramLUTWidget()
    histlut.setImageItem(cloud)

    n = 1000
    cloud.extendData([x.pop() for i in range(n)], [y.pop() for i in range(n)], [c.pop() for i in range(n)])

    view.addItem(cloud)

    qtbot.wait_exposed(win)
    qtbot.wait_exposed(histlut)

    with monkeypatch.context() as m:
        with qtbot.waitCallback() as cb:
            # monkeypatch paint method to capture call
            def paintGL(*args, **kwargs):
                ret = CloudItem.paintGL(cloud, *args, **kwargs)
                cb()
                return ret

            m.setattr(cloud, 'paintGL', paintGL)

    qtbot.add_widget(win)
    qtbot.add_widget(histlut)
