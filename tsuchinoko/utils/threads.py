import sys
import threading
import time
from functools import wraps

from loguru import logger
from qtpy.QtCore import QTimer, Signal, QThread, QObject, QEvent, QCoreApplication
from qtpy.QtWidgets import QApplication

from tsuchinoko.utils.coverage import coverage_resolve_trace


def log_error(exception: Exception, value=None, tb=None, **kwargs):
    """
    Logs an exception with traceback. All uncaught exceptions get hooked here

    """

    if not value:
        value = exception
    if not tb:
        tb = exception.__traceback__

    if "caller_name" not in kwargs:
        frame = sys._getframe()
        frame = getattr(frame, "f_back", frame) or frame
        kwargs["caller_name"] = frame.f_code.co_name

    caller_name = kwargs.pop('caller_name', '')

    logger.error("\n The following in {caller_name} was handled safely. It is displayed here for debugging.")
    logger.exception(exception)
    # try:
    #     logging.log(logging.ERROR, "\n" + ' '.join(traceback.format_exception(exception, value, tb)), extra={"caller_name": caller_name}, **kwargs)
    # except AttributeError:
    #     logging.log(logging.ERROR, "\n" + ' '.join(traceback.format_exception_only(exception, value)), extra={"caller_name": caller_name}, **kwargs)


show_busy = lambda *_: None
show_ready = lambda *_: None


# Justification for subclassing qthread: https://woboq.com/blog/qthread-you-were-not-doing-so-wrong.html
class QThreadFuture(QThread):
    """
    A future-like QThread, with many conveniences.
    """

    sigCallback = Signal()
    sigFinished = Signal()
    sigExcept = Signal(Exception)

    def __init__(
            self,
            method,
            *args,
            callback_slot=None,
            finished_slot=None,
            except_slot=None,
            interrupt_callable=None,
            # default_exhandle=True,
            # lock=None,
            # threadkey: str = None,
            showBusy=True,
            # keepalive=True,
            # cancelIfRunning=True,
            priority=QThread.InheritPriority,
            timeout=0,
            name=None,
            **kwargs,
    ):
        super(QThreadFuture, self).__init__()

        # Auto-Kill other threads with same threadkey
        # if threadkey and cancelIfRunning:
        #     for thread in manager.threads:
        #         if thread.threadkey == threadkey:
        #             thread.cancel()
        # self.threadkey = threadkey

        self.name = name
        self.callback_slot = callback_slot
        self.except_slot = except_slot
        # if callback_slot: self.sigCallback.connect(callback_slot)
        self.interrupt_callable = interrupt_callable
        if finished_slot:
            self.sigFinished.connect(finished_slot)
        if except_slot:
            self.sigExcept.connect(except_slot)
        if QApplication.instance():
            # QApplication.instance().aboutToQuit.connect(self.quit)
            QApplication.instance().aboutToQuit.connect(self.interrupt)
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout  # ms

        self.cancelled = False
        self.exception = None
        self._purge = False
        self.thread = None
        self.priority = priority
        self.showBusy = showBusy

        # if keepalive:
        #     manager.append(self)

    @property
    def done(self):
        return self.isFinished()

    @property
    def running(self):
        return self.isRunning()

    def start(self):
        """
        Starts the thread
        """
        if self.running:
            raise ValueError("Thread could not be started; it is already running.")

        super(QThreadFuture, self).start(self.priority)
        if self.timeout:
            self._timeout_timer = QTimer.singleShot(self.timeout, self.cancel)

    @coverage_resolve_trace
    def run(self, *args, **kwargs):
        """
        Do not call this from the main thread; you're probably looking for start()
        """
        # if self.threadkey:
        #     threading.current_thread().name = self.threadkey

        threading.current_thread().name = self.name
        self.cancelled = False
        self.exception = None
        if self.showBusy:
            show_busy()
        try:
            value = [None]
            runner = self._run(*args, **kwargs)
            while not self.isInterruptionRequested():
                try:
                    value = next(runner)
                except StopIteration as ex:
                    value = ex.value
                    if not isinstance(value, tuple):
                        value = (value,)
                    if isinstance(self, QThreadFutureIterator) and self.callback_slot:
                        self.callback_slot(*value)
                    self._result = value
                    break
                if not isinstance(value, tuple):
                    value = (value,)
                if isinstance(self, QThreadFutureIterator) and self.yield_slot:
                    invoke_in_main_thread(self.yield_slot, *value)
                elif isinstance(self, QThreadFuture) and self.callback_slot:
                    invoke_in_main_thread(self.callback_slot, *value)

        except Exception as ex:
            self.exception = ex
            self.sigExcept.emit(ex)
            logger.error(f"Error in thread: "
                         f'Method: {getattr(self.method, "__name__", "UNKNOWN")}\n'
                         f"Args: {self.args}\n"
                         f"Kwargs: {self.kwargs}", )
            log_error(ex)
        else:
            self.sigFinished.emit()
        finally:
            if self.showBusy:
                show_ready()

            self.quit()
            if QApplication.instance():
                try:
                    # QApplication.instance().aboutToQuit.disconnect(self.quit)
                    QApplication.instance().aboutToQuit.disconnect(self.interrupt)
                # Somehow the application never had its aboutToQuit connected to quit...
                except (TypeError, RuntimeError) as e:
                    # msg.logError(e)
                    ...

    def _run(self, *args, **kwargs):  # Used to generalize to QThreadFutureIterator
        yield self.method(*self.args, **self.kwargs)

    def result(self):
        if not self.running:
            self.start()
        while not self.done and not self.exception:
            time.sleep(0.01)
        if self.exception:
            return self.exception
        return self._result

    def cancel(self):
        self.cancelled = True
        if self.except_slot:
            invoke_in_main_thread(self.except_slot, InterruptedError("Thread cancelled."))
        self.requestInterruption()
        self.quit()
        self.wait()

    def interrupt(self):
        self.requestInterruption()
        if self.interrupt_callable:
            self.interrupt_callable()
        self.wait()


class QThreadFutureIterator(QThreadFuture):
    """
    Same as QThreadFuture, but emits to the callback_slot for every yielded value of a generator
    """

    def __init__(self, *args, yield_slot=None, **kwargs):
        super(QThreadFutureIterator, self).__init__(*args, **kwargs)
        self.yield_slot = yield_slot

    def _run(self, *args, **kwargs):
        return (yield from self.method(*self.args, **self.kwargs))


class InvokeEvent(QEvent):
    """
    Generic callable containing QEvent
    """

    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, fn, *args, **kwargs):
        QEvent.__init__(self, InvokeEvent.EVENT_TYPE)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class Invoker(QObject):
    def event(self, event):
        try:
            if hasattr(event.fn, "signal"):  # check if invoking a signal or a callable
                event.fn.emit(*event.args, *event.kwargs.values())
            else:
                event.fn(*event.args, **event.kwargs)
            return True
        except Exception as ex:
            logger.error("QThreadFuture callback could not be invoked.")
            log_error(ex)
        return False


_invoker = Invoker()


def invoke_in_main_thread(fn, *args, force_event=False, **kwargs):
    """
    Invoke a callable in the main thread. Use this for making callbacks to the gui where signals are inconvenient.
    """
    # co_name = sys._getframe().f_back.f_code.co_name
    # msg.logMessage(f"Invoking {fn} in main thread from {co_name}")

    if not force_event and is_main_thread():
        # we're already in the main thread; just do it!
        fn(*args, **kwargs)
    else:
        QCoreApplication.postEvent(_invoker, InvokeEvent(fn, *args, **kwargs))


def invoke_as_event(fn, *args, **kwargs):
    """Invoke a callable as an event in the main thread."""
    # co_name = sys._getframe().f_back.f_code.co_name
    # msg.logMessage(f"Invoking {fn} in main thread from {co_name}")
    invoke_in_main_thread(fn, *args, force_event=True, **kwargs)


def is_main_thread():
    return threading.current_thread() is threading.main_thread()


def method(
        callback_slot=None,
        finished_slot=None,
        except_slot=None,
        # default_exhandle=True,
        # lock=None,
        # threadkey: str = None,
        showBusy=True,
        priority=QThread.InheritPriority,
        # keepalive=True,
        # cancelIfRunning=True,
        timeout=0,
        block=False,
        name=None,
):
    """
    Decorator for functions/methods to run as RunnableMethods on background QT threads
    Use it as any python decorator to decorate a function with @decorator syntax or at runtime:
    decorated_method = threads.method(callback_slot, ...)(method_to_decorate)
    then simply run it: decorated_method(*args, **kwargs)
    Parameters
    ----------
    callback_slot : function
        Function/method to run on a background thread
    finished_slot : QtCore.Slot
        Slot to call with the return value of the function
    except_slot : QtCore.Slot
        Function object (qt slot), slot to receive exception type, instance and traceback object
    default_exhandle : bool
        Flag to use the default exception handle slot. If false it will not be called
    lock : mutex/semaphore
        Simple lock if multiple access needs to be prevented
    Returns
    -------
    wrap_runnable_method : function
        Decorated function/method
    """

    def wrap_runnable_method(func):
        @wraps(func)
        def _runnable_method(*args, **kwargs):
            future = QThreadFuture(
                func,
                *args,
                callback_slot=callback_slot,
                finished_slot=finished_slot,
                except_slot=except_slot,
                # default_exhandle=default_exhandle,
                # lock=lock,
                # threadkey=threadkey,
                showBusy=showBusy,
                priority=priority,
                # keepalive=keepalive,
                # cancelIfRunning=cancelIfRunning,
                timeout=timeout,
                name=name,
                **kwargs,
            )
            future.start()
            if block:
                future.result()
            return future

        return _runnable_method

    return wrap_runnable_method


def iterator(
        yield_slot=None,
        callback_slot=None,
        finished_slot=None,
        interrupt_signal=None,
        except_slot=None,
        # default_exhandle=True,
        # lock=None,
        # threadkey: str = None,
        showBusy=True,
        priority=QThread.InheritPriority,
        name=None,
        # keepalive=True,
):
    """
    Decorator for iterators/generators to run as RunnableIterators on background QT threads
    Use it as any python decorator to decorate a function with @decorator syntax or at runtime:
    decorated_iterator = threads.iterator(callback_slot, ...)(iterator_to_decorate).
    then simply run it: decorated_iterator(*args, **kwargs)

    Parameters
    ----------
    callback_slot : function
        Function/method to run on a background thread
    finished_slot : QtCore.Slot
        Slot to call with the return value of the function
    interrupt_signal : QtCore.Signal
        Signal to break out of iterator loop prematurely
    except_slot : QtCore.Slot
        Function object (qt slot), slot to receive exception type, instance and traceback object
    lock : mutex/semaphore
        Simple lock if multiple access needs to be prevented

    Returns
    -------
    wrap_runnable_iterator : function
        Decorated iterator/generator
    """

    def wrap_runnable_method(func):
        @wraps(func)
        def _runnable_method(*args, **kwargs):
            future = QThreadFutureIterator(
                func,
                *args,
                yield_slot=yield_slot,
                callback_slot=callback_slot,
                finished_slot=finished_slot,
                except_slot=except_slot,
                # default_exhandle=default_exhandle,
                # lock=lock,
                # threadkey=threadkey,
                showBusy=showBusy,
                priority=priority,
                name=name,
                # keepalive=keepalive,
                **kwargs,
            )
            future.start()

        return _runnable_method

    return wrap_runnable_method
