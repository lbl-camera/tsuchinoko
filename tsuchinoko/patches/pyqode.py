import enum
import struct

from pyqode.core.api.client import JsonTcpClient, comm  # Must be a late import
from qtpy import QtGui, QtCore, QtWidgets


def _read_header(self):
    comm('reading header')
    self._header_buf += self.read(4)
    if len(self._header_buf) == 4:
        self._header_complete = True
        try:
            if hasattr(self._header_buf, 'data'):
                raise TypeError  # The following line unforgivingly causes access violation on PySide2, skip to doing it right
            header = struct.unpack('=I', self._header_buf)
        except TypeError:
            # pyside
            header = struct.unpack('=I', self._header_buf.data())
        self._to_read = header[0]
        self._header_buf = bytes()
        comm('header content: %d', self._to_read)


JsonTcpClient._read_header = _read_header

# fix enum promotion in qtpy/pyside6. This looks wacky, but it actually does something.
for module in [QtCore, QtWidgets, QtGui]:
    class_names = [name for name in dir(module) if name.startswith("Q")]
    for class_name in class_names:
        klass = getattr(module, class_name)
        attrib_names = [name for name in dir(klass) if name[0].isupper()]
        for attrib_name in attrib_names:
            attrib = getattr(klass, attrib_name)
            if not isinstance(attrib, enum.EnumMeta):
                continue
            for name, value in attrib.__members__.items():
                setattr(klass, name, value)
