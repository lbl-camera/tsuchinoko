from qtpy.QtWidgets import QSlider, QWidget, QHBoxLayout, QLabel
from qtpy.QtCore import Signal


class ValueDoubleSlider(QWidget):
    sigFloatValueChanged = Signal(float)
    '''Shows current value'''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.slider = DoubleSlider(*args, **kwargs)
        self.label = QLabel()
        self.label.setFixedWidth(40)

        self.slider.sigFloatValueChanged.connect(self.sigFloatValueChanged)
        self.slider.sigFloatValueChanged.connect(self.update_label)

        ## Explicitly wrap methods from DoubleSlider
        for m in ['value', 'setMinimum', 'setMaximum', 'setInterval']:
            setattr(self, m, getattr(self.slider, m))

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.slider)
        self.update_label(self.slider.value())

    def update_label(self, value):
        self.label.setText(str(value))


class DoubleSlider(QSlider):
    sigFloatValueChanged = Signal(float)

    def __init__(self, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._min = 0
        self._max = 99
        self.interval = 1
        self.valueChanged.connect(self._emit_value)

    def setValue(self, value):
        index = round((value - self._min) / self.interval)
        return super(DoubleSlider, self).setValue(index)

    def value(self):
        return self.index * self.interval + self._min

    @property
    def index(self):
        return super(DoubleSlider, self).value()

    def setIndex(self, index):
        return super(DoubleSlider, self).setValue(index)

    def setMinimum(self, value):
        self._min = value
        self._range_adjusted()

    def setMaximum(self, value):
        self._max = value
        self._range_adjusted()

    def setInterval(self, value):
        # To avoid division by zero
        if not value:
            raise ValueError('Interval of zero specified')
        self.interval = value
        self._range_adjusted()

    def _range_adjusted(self):
        number_of_steps = int((self._max - self._min) / self.interval)
        super(DoubleSlider, self).setMaximum(number_of_steps)

    def _emit_value(self):
        self.sigFloatValueChanged.emit(self.value())