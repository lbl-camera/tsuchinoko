# this is just to mock an analysis element. It effectively just relay's the
# measured value here
from tsuchinoko.utils.zmq_queue import Queue_analyze

q = Queue_analyze()

if __name__ == '__main__':
    while True:  # The loop that waits for new instructions...

        data = q.get()  # Get analysis command

        print("'Analyzing' data:", data)

        q.publish(data)  # Send new analysis results to gpCAM
