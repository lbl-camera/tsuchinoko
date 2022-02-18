import numpy as np
from caproto.server import PVGroup, pvproperty, template_arg_parser, run
from scipy.stats import multivariate_normal


class TestOpticsBeamline(PVGroup):
    optic_1 = pvproperty(value=0, dtype=float)
    optic_2 = pvproperty(value=0, dtype=float)
    optic_3 = pvproperty(value=3, dtype=float)
    optic_4 = pvproperty(value=4, dtype=float)

    scalar_sensor = pvproperty(value=0, dtype=float, read_only=True)

    target = [1, 2]
    sigma = np.array([1, 1])

    @scalar_sensor.getter
    async def scalar_sensor(self, instance):
        return np.multivariate_normal.pdf([self.optic_1.value, self.optic_2.value],
                                          self.target,
                                          np.diag(self.sigma ** 2)) + 1 * np.random.rand() * 1e-1


class TwoDScanBeamline(TestOpticsBeamline):
    scan_x = pvproperty(value=0, dtype=float)
    scan_y = pvproperty(value=0, dtype=float)

    scan_sensor = pvproperty(value=0, dtype=float, read_only=True)

    @scan_sensor.getter
    async def scan_sensor(self, instance):
        await self.scalar_sensor.read(self.scalar_sensor.data_type)  # update derived value
        return multivariate_normal.pdf([self.scan_x.value, self.scan_y.value],
                                       [0, 0],
                                       np.diag([.1, .1])) * self.scalar_sensor.value


def twod_scan_beamline():
    return TwoDScanBeamline()


if __name__ == "__main__":
    parser, split_args = template_arg_parser(default_prefix='test:', desc='beamline optics test')

    args = parser.parse_args()
    ioc_options, run_options = split_args(args)
    ioc = TwoDScanBeamline(**ioc_options)
    run_options['log_pv_names'] = True
    run(ioc.pvdb, **run_options)
