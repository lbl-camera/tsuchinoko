from astropy import modeling
from bluesky import preprocessors as bpp, Msg
from bluesky import plan_stubs as bps
import numpy as np
from scipy import stats

from tsuchinoko.utils.threads import invoke_in_main_thread


def tune_centroid_and_fit(
        detectors, signal, motor,
        start, stop, min_step,
        num=10,
        step_factor=3.0,
        snake=False,
        name='primary',
        expected_spot_size=1,
        debug=False,
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

    # @bpp.stage_decorator(list(detectors) + [motor])
    def _tune_core(start, stop, num, signal):

        next_pos = start
        step = (stop - start) / (num - 1)
        peak_position = None
        cur_I = None
        sum_I = 0       # for peak centroid calculation, I(x)
        sum_xI = 0

        xs = []
        ys = []

        while abs(step) >= min_step and low_limit <= next_pos <= high_limit:
            yield Msg('checkpoint')
            yield from bps.mv(motor, next_pos)
            ret = (yield from bps.trigger_and_read(detectors + [motor], name=name))
            cur_I = ret[signal]['value']
            xs.append(next_pos)
            ys.append(cur_I)
            sum_I += cur_I
            position = ret[motor_name]['value']
            sum_xI += position * cur_I

            next_pos += step
            in_range = min(start, stop) <= next_pos <= max(start, stop)

            if not in_range:
                if sum_I == 0:
                    return
                peak_position = sum_xI / sum_I  # centroid
                sum_I, sum_xI = 0, 0    # reset for next pass
                new_scan_range = (stop - start) / step_factor
                start = np.clip(peak_position - new_scan_range/2,
                                low_limit, high_limit)
                stop = np.clip(peak_position + new_scan_range/2,
                               low_limit, high_limit)
                if snake:
                    start, stop = stop, start
                step = (stop - start) / (num - 1)
                next_pos = start
                # print("peak position = {}".format(peak_position))
                # print("start = {}".format(start))
                # print("stop = {}".format(stop))

        # finally, move to peak position
        if peak_position is not None:
            # improvement: report final peak_position
            # print("final position = {}".format(peak_position))
            yield from bps.mv(motor, peak_position)

        return xs, ys

    xs, ys = np.asarray((yield from _tune_core(start, stop, num, signal)))

    # fit gaussian to measured points

    model = modeling.models.Gaussian1D(amplitude=ys.max(), mean=xs[np.argmax(ys)], stddev=expected_spot_size)
    fitter = modeling.fitting.SLSQPLSQFitter()
    fitted_model = fitter(model, xs, ys)
    if debug and (fitted_model.stddev.value < .1 or fitted_model.stddev.value > 100):  # disabled; for debugging purposes
        import pyqtgraph as pg
        def plot_poor_fit(xs, ys):
            w = pg.plot()
            w.addLegend()
            w.plot(xs, ys, pen='w', name='measurement')
            w.plot(xs, fitted_model(xs),pen='r', name='fit')
            w.plot(xs, model(xs),pen='g', name='initial model')
            w.show()
        invoke_in_main_thread(plot_poor_fit, xs=xs, ys=ys)

    return fitted_model
