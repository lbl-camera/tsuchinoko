import numpy as np
from PySide2.QtCore import Qt
from astropy import modeling
from bluesky import Msg, plan_stubs as bps

from tsuchinoko.utils.threads import invoke_in_main_thread


def tune_max_and_fit(
        detectors, signal, motor,
        start, stop, min_step,
        num=10,
        step_factor=3.0,
        snake=False,
        name='primary',
        expected_spot_size=1,
        debug=False,
        sleep=0,
        dark_value=0,
        *, md=None):
    r"""
    plan: tune a motor to the centroid of signal(motor)

    Initially, traverse the range from start to stop with
    the number of points specified.  Repeat with progressively
    smaller step size until the minimum step size is reached.
    Rescans will be centered on the signal centroid
    (for $I(x)$, centroid$= \sum{I}/\sum{x*I}$)
    with original scan range reduced by ``step_factor``.

    Set ``snake=True`` if your positions are reproducible
    moving from either direction.  This will not necessarily
    decrease the number of traversals required to reach convergence.
    Snake motion reduces the total time spent on motion
    to reset the positioner.  For some positioners, such as
    those with hysteresis, snake scanning may not be appropriate.
    For such positioners, always approach the positions from the
    same direction.

    Note:  Ideally the signal has only one peak in the range to
    be scanned.  It is assumed the signal is not polymodal
    between ``start`` and ``stop``.

    Parameters
    ----------
    detectors : Signal
        list of 'readable' objects
    signal : string
        detector field whose output is to maximize
    motor : object
        any 'settable' object (motor, temp controller, etc.)
    start : float
        start of range
    stop : float
        end of range, note: start < stop
    min_step : float
        smallest step size to use.
    num : int, optional
        number of points with each traversal, default = 10
    step_factor : float, optional
        used in calculating new range after each pass

        note: step_factor > 1.0, default = 3
    snake : bool, optional
        if False (default), always scan from start to stop
    md : dict, optional
        metadata

    Examples
    --------
    Find the center of a peak using synthetic hardware.

    >>> from ophyd.sim import SynAxis, SynGauss
    >>> motor = SynAxis(name='motor')
    >>> det = SynGauss(name='det', motor, 'motor',
    ...                center=-1.3, Imax=1e5, sigma=0.05)
    >>> RE(tune_centroid([det], "det", motor, -1.5, -0.5, 0.01, 10))
    """
    if min_step <= 0:
        raise ValueError("min_step must be positive")
    if step_factor <= 1.0:
        raise ValueError("step_factor must be greater than 1.0")
    try:
        motor_name, = motor.hints['fields']
    except (AttributeError, ValueError):
        motor_name = motor.name
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motor': repr(motor),
                         'start': start,
                         'stop': stop,
                         'num': num,
                         'min_step': min_step, },
           'plan_name': 'tune_centroid',
           'hints': {},
           }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)

    low_limit = min(start, stop)
    high_limit = max(start, stop)

    initial_pos = motor.user_readback.get()

    # @bpp.stage_decorator(list(detectors) + [motor])
    def _tune_core(start, stop, num, signal):

        next_pos = start
        step = (stop - start) / (num - 1)
        peak_position = None
        cur_I = None
        sum_I = 0  # for peak centroid calculation, I(x)
        sum_xI = 0

        xs = []
        ys = []

        next_points = list(np.arange(low_limit, high_limit, step))

        while abs(step) >= min_step:
            yield Msg('checkpoint')
            next_pos = next_points.pop()
            yield from bps.mv(motor, next_pos)
            yield from bps.sleep(sleep)
            ret = (yield from bps.trigger_and_read(detectors + [motor], name=name))
            cur_I = ret[signal]['value']
            xs.append(next_pos)
            ys.append(cur_I)
            # sum_I += cur_I
            # position = ret[motor_name]['value']
            # sum_xI += position * cur_I

            if not next_points:

                arg_max = np.argmax(ys)
                if xs[arg_max] == min(xs):
                    next_points = [xs[arg_max] - step]
                elif xs[arg_max] == max(xs):
                    next_points = [xs[arg_max] + step]
                else:
                    step /= 2
                    next_points = [xs[arg_max] - step, xs[arg_max] + step]

        return xs, ys

    xs, ys = np.asarray((yield from _tune_core(start, stop, num, signal)))

    # fit gaussian to measured points

    model = modeling.models.Gaussian1D(amplitude=ys.max(), mean=xs[np.argmax(ys)], stddev=expected_spot_size) + modeling.models.Const1D(dark_value)
    fitter = modeling.fitting.SLSQPLSQFitter()
    fitted_model = fitter(model, xs, ys)
    if debug:  # and (fitted_model.stddev.value <= .1 or fitted_model.stddev.value >= 100):  # disabled; for debugging purposes
        import pyqtgraph as pg
        def plot_poor_fit(xs, ys):
            w = pg.plot(title=f'Scan of {motor_name}')
            w.addLegend()
            sorted_xs = sorted(xs)
            w.plot(sorted_xs, model(sorted_xs), pen='g', name='initial model')
            w.plot(xs, ys, pen=pg.mkPen(color='w', style=Qt.DashLine))
            w.plot(sorted_xs, fitted_model(sorted_xs), pen='r', name='fit')
            w.addItem(pg.ScatterPlotItem(x=xs, y=ys, size=10, pen=pg.mkPen(None), brush=pg.mkBrush('w')), name='measurement')
            w.show()

        invoke_in_main_thread(plot_poor_fit, xs=xs, ys=ys)

    if start < fitted_model.mean_0.value < stop and fitted_model.amplitude_0 > dark_value and fitted_model.stddev_0 < 10 * expected_spot_size:
        yield from bps.mov(motor, fitted_model.mean_0.value)

        return fitted_model

    else:
        # on failed fit, move back where we started
        yield from bps.mov(motor, initial_pos)
        return None
