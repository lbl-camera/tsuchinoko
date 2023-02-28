import asyncio
import time

import numpy as np
from ophyd.sim import SynAxis, Device, Cpt, SynSignalRO, SynSignal
from bluesky.plan_stubs import mv, trigger_and_read, create, stage, checkpoint, mov
from PIL import Image
from scipy import ndimage

from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
from tsuchinoko.adaptive.random_in_process import RandomInProcess
from tsuchinoko.core import Core, ZMQCore
from tsuchinoko.execution.bluesky_in_process import BlueskyInProcessEngine

import numpy as np
import os
import time
import sys
import h5py
import zmq
import zlib
import pickle

from tsuchinoko.execution.threaded_in_process import ThreadedInProcessEngine

def val2ind(val, an_array):
    return np.argmin(abs(an_array-val), axis=0)

class ir_getdata:

    def __init__(self, fileName, ref_path, print_dir=False, dataName=r'./gpdata/Test'):
        h5_groups = []
        epsilon, b = 0.4, 0.29
        with h5py.File(fileName,'r') as f:
            root_name = list(f.keys())[0]
            print(root_name)
            h = f[root_name] #folder name
            self.xy = f[root_name + '/data/xy'][:,:]
            self.wav = f[root_name + '/data/wavenumbers'][:]
            self.N_w = len(self.wav)
            self.data_list = f[root_name + '/data/spectra'][:,:]
            self.data_cube = f[root_name + '/data/image/image_cube'][:, : ,:]
            self.data_list = -np.log10(self.data_list/100 + epsilon) + b
            self.data_cube = -np.log10(self.data_cube/100 + epsilon) + b
            h.visit(h5_groups.append)
            if print_dir:
                print(*h5_groups, sep='\n')
        self.n_spec = 0
        self.cum_spectra = np.zeros((0, self.N_w))
        path = os.path.dirname(dataName)
        self.cum_specName = os.path.join(path, 'all_specs.npy')
        self.ref_spec = np.loadtxt(ref_path, delimiter=',')
        print(f'ref_spec: {os.path.basename(ref_path)}')

        # get X, Y coordinates
        self.X = np.unique(self.xy[:,0])
        self.Y = np.unique(self.xy[:,1])

    def get_spec(self, pos):
        row, col = [], []
        pos = pos.reshape(-1,2)
        for i in range(len(pos)):
            row.append(val2ind(pos[i, 1], self.Y))  # align y coordinate to get row
            col.append(val2ind(pos[i, 0], self.X))  # align x coordinate to get col
        spec = self.data_cube[row, col, :]
        self.cum_spectra = np.append(self.cum_spectra, spec.reshape(1, -1), axis=0)
        np.save(self.cum_specName, self.cum_spectra)
        self.n_spec += 1

    def alignWithRef(self, spec, ref):
        if ref.ndim == 1:
            return spec, ref
        elif ref.ndim == 2:
            pos2, pos1 = val2ind(self.wav, ref[-1, 0]) + 1, val2ind(self.wav, ref[0, 0])
            if spec.ndim == 1: spec = spec.reshape(1, -1)
            if len(self.wav[pos1: pos2]) == ref.shape[0]:
                return spec[:, pos1: pos2], ref[:, 1]
            elif len(self.wav[pos1: pos2]) == ref[::2, :].shape[0]:
                return spec[:, pos1: pos2], ref[::2, 1]
            elif len(self.wav[pos1: pos2][::4]) == ref.shape[0]:
                return spec[:, pos1: pos2: 4], ref[:, 1]
            elif len(self.wav[pos1: pos2][::4]) == ref[::2, :].shape[0]:
                return spec[:, pos1: pos2: 4], ref[::2, 1]
            else:
                print("Length doesn't match")
                return None

    def gp_getCorrData(self, idx):
        spec = self.cum_spectra[idx, :]
        spec, ref = self.alignWithRef(spec, self.ref_spec)
        return np.corrcoef(spec, ref[None, :])[0, 1]

def instrument_synthetic(data):
    # load last saved data
    variance = 1e-3

    if (ir_data.n_spec == 0) and os.path.exists(ir_data.cum_specName) :
        ir_data.cum_spectra = np.load(ir_data.cum_specName)
        ir_data.n_spec = ir_data.cum_spectra.shape[0]

    for idx_data in range(len(data)):
        if data[idx_data]["measured"] == True: continue
        x1 = data[idx_data]["position"][0]
        x2 = data[idx_data]["position"][1]
        pos = np.array([x1, x2])
        ir_data.get_spec(pos)

    for idx_data in range(len(data)):
        if data[idx_data]["measured"] == True: continue
        data[idx_data]["value"] = ir_data.gp_getCorrData(-1)  # Modified for single-point operation idx_data)
        data[idx_data]["variance"] = variance
        data[idx_data]["measured"] = True
        data[idx_data]["time stamp"] = time.time()
    return data

fileName = 'E:\\data\\IR-gpcam\\20180216r_S1_tr_area1.h5'
ref_path = 'E:\\data\\IR-gpcam\\Lipopolysaccharides.CSV'
dataName = 'E:\\data\\IR-gpcam\\'
# ir_data = ir_getdata(fileName, ref_path, dataName=dataName)

## for beamline experiment

def alignWithRef(wav, spec, ref):
    if ref.ndim == 1:
        return spec, ref
    elif ref.ndim == 2:
        pos2, pos1 = val2ind(wav, ref[-1, 0]) + 1, val2ind(wav, ref[0, 0])
        if spec.ndim == 1: spec = spec.reshape(1, -1)
        if len(wav[pos1: pos2]) == ref.shape[0]:
            return spec[:, pos1: pos2], ref[:, 1]
        elif len(wav[pos1: pos2]) == ref[::2, :].shape[0]:
            return spec[:, pos1: pos2], ref[::2, 1]
        elif len(wav[pos1: pos2][::4]) == ref.shape[0]:
            return spec[:, pos1: pos2: 4], ref[:, 1]
        elif len(wav[pos1: pos2][::4]) == ref[::2, :].shape[0]:
            return spec[:, pos1: pos2: 4], ref[::2, 1]
        else:
            print("Length doesn't match")
            return None


def getCorrData(wav, spec, ref):
    spec, ref = alignWithRef(wav, spec, ref)
    return np.corrcoef(spec, ref[None, :])[0, 1]


class zmq_connect:

    def __init__(self, ref_file=r'./ref_spec/black_carbons_on_minerals.CSV', specs_file=r'./gpFiles/spectra.npy',
                 port="6888"):

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % port)
        self.ref_spec = np.loadtxt(ref_file, delimiter=',')
        self.cum_spectra = None
        self.specs_file = specs_file
        print(f'Reference spectrum: {os.path.basename(ref_file)}')
        print(f'Raw spectra are saved at: {specs_file}')

    def send_zipped_pickle(self, obj, flags=0, protocol=-1):
        """pack and compress an object with pickle and zlib."""
        pobj = pickle.dumps(obj, protocol)
        zobj = zlib.compress(pobj)
        #         print('zipped pickle is %i bytes' % len(zobj))
        self.socket.send(zobj, flags=flags)

    def recv_zipped_pickle(self, flags=0):
        """reconstruct a Python object sent with zipped_pickle"""
        zobj = self.socket.recv(flags)
        pobj = zlib.decompress(zobj)
        return pickle.loads(pobj)

    def comm_via_zmq_transform(self, data):
        self.send_zipped_pickle(data)
        np.save("data_last_sent", data)
        print("=================================")
        data = self.recv_zipped_pickle()
        for idx in range(len(data)):
            wav, spec = data[idx]["value"]
            if self.cum_spectra is None:
                self.cum_spectra = wav.reshape(1, -1)
            self.cum_spectra = np.append(self.cum_spectra, spec.reshape(1, -1), axis=0)
            data[idx]["value"] = getCorrData(wav, spec, self.ref_spec)
        print("gpCAM received data point#: ", self.cum_spectra.shape[0] - 1)
        np.save("data_last_received", data)
        np.save(self.specs_file, self.cum_spectra)
        return data

    def start(self):
        msg = self.recv_zipped_pickle()
        print(msg)

    def stop(self):
        self.socket.close()
        self.context.term()
        print('zmq connection is stopped.')


if __name__ == '__main__':

    zmq_con = zmq_connect()
    zmq_con.start()

    def measure_target(target, instrument=zmq_con):
        # data = instrument_synthetic([{'position': target, 'measured': False}])
        data = instrument.comm_via_zmq_transform([{'position': target, 'measured': False}])
        return data[0]['value'], data[0]['variance']


    adaptive = GPCAMInProcessEngine(dimensionality=2,
                                    parameter_bounds=[[-20590.4,  -20010.4], [3394.0, 3894.0]],
                                    hyperparameters=[1, 1, 1],
                                    hyperparameter_bounds=[[0.0001,10],[0.001,1e4],[0.001,1e4]])

    execution = ThreadedInProcessEngine(measure_target)

    core = ZMQCore()
    core.set_adaptive_engine(adaptive)
    core.set_execution_engine(execution)

    core.main()
