import numpy as np

from tsuchinoko.examples.gi_tomo_phantom import projection_operator, l_x, n_sinograms, domain_maps, bilinear_sample
from tsuchinoko.utils.zmq_queue import Queue_measure

measure_queue = Queue_measure()



if __name__ == "__main__":
    while True: # The loop that waits for new instructions...

        pos = measure_queue.get()  # Get measurement command
        print('received:', pos)
        pos[0]['value'] = bilinear_sample(pos[0]['position'], domain_maps)
        print('sending:', pos)
        measure_queue.publish(pos)  # Send new results for analysis

