from pathlib import Path

from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import QAction, QStyle, QMessageBox, QFileDialog, QVBoxLayout
from pyqode.python.folding import PythonFoldDetector
from qtpy.QtWidgets import QWidget, QApplication
from qtpy.QtCore import Qt
from pyqode.core import panels, api, modes
from pyqode.python import widgets, panels as pypanels, modes as pymodes
from pyqode.python.backend import server

from tsuchinoko import assets
from tsuchinoko.examples import server_demo
from pyqode.python.backend.workers import defined_names
from pyqode.python import managers as pymanagers

import inspect

from tsuchinoko.widgets.debugmenubar import DebuggableMenuBar


class PythonEditor(widgets.PyCodeEditBase):
    def __init__(self):
        super(PythonEditor, self).__init__()

        # starts the default pyqode.python server (which enable the jedi code
        # completion worker).
        self.backend.start(server.__file__)

        # some other modes/panels require the analyser mode, the best is to
        # install it first
        self.modes.append(modes.OutlineMode(defined_names))

        # --- core panels
        self.panels.append(panels.FoldingPanel())
        self.panels.append(panels.LineNumberPanel())
        self.panels.append(panels.CheckerPanel())
        self.panels.append(panels.SearchAndReplacePanel(),
                           panels.SearchAndReplacePanel.Position.BOTTOM)
        self.panels.append(panels.EncodingPanel(), api.Panel.Position.TOP)
        # add a context menu separator between editor's
        # builtin action and the python specific actions
        self.add_separator()

        # --- python specific panels
        self.panels.append(pypanels.QuickDocPanel(), api.Panel.Position.BOTTOM)

        # --- core modes
        self.modes.append(modes.CaretLineHighlighterMode())
        self.modes.append(modes.CodeCompletionMode())
        self.modes.append(modes.ExtendedSelectionMode())
        self.modes.append(modes.FileWatcherMode())
        self.modes.append(modes.OccurrencesHighlighterMode())
        self.modes.append(modes.RightMarginMode())
        self.modes.append(modes.SmartBackSpaceMode())
        self.modes.append(modes.SymbolMatcherMode())
        self.modes.append(modes.ZoomMode())
        # self.modes.append(modes.PygmentsSyntaxHighlighter(self.document()))

        # ---  python specific modes
        self.modes.append(pymodes.CommentsMode())
        self.modes.append(pymodes.CalltipsMode())
        self.modes.append(pymodes.PyFlakesChecker())
        self.modes.append(pymodes.PEP8CheckerMode())
        self.modes.append(pymodes.PyAutoCompleteMode())
        self.modes.append(pymodes.PyAutoIndentMode())
        self.modes.append(pymodes.PyIndenterMode())
        self.modes.append(pymodes.PythonSH(self.document()))
        self.syntax_highlighter.fold_detector = PythonFoldDetector()

        self.syntax_highlighter.color_scheme = api.ColorScheme('darcula')
        self.save_on_focus_out = True


class ServerEditor(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window
        self.editor = PythonEditor()

        menubar = DebuggableMenuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction('&New', self.new)
        open_action = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), 'Open...', parent=file_menu)
        save_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save',
                                   parent=file_menu)
        save_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_S))
        save_as_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save As...',
                                   parent=file_menu)
        server_menu = menubar.addMenu("&Server")
        start_server = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), 'Start Server', parent=server_menu)
        stop_server = QAction(self.style().standardIcon(QStyle.SP_MediaStop), 'Stop Server', parent=server_menu)

        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addAction('E&xit', self.exit)
        server_menu.addAction(start_server)
        server_menu.addAction(stop_server)

        open_action.triggered.connect(self.open)
        save_action.triggered.connect(self.save)
        save_as_action.triggered.connect(self.save_as)
        start_server.triggered.connect(self.start_server)
        stop_server.triggered.connect(self.stop_server)

        self.setWindowTitle('Tsuchinoko Server Editor')
        self.setWindowIcon(QIcon(assets.path('tsuchinoko.png')))
        self.resize(1700, 1000)

        self.has_unsaved_changes = False
        self.editor.textChanged.connect(self._set_has_changes)

        QApplication.instance().aboutToQuit.connect(self.close_by_quit)  # TODO: use this approach in Xi-cam

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(menubar)
        layout.addWidget(self.editor)
        self.setLayout(layout)

        self.new()
        self.editor.setPlainText(inspect.getsource(server_demo))

    def _set_has_changes(self):
        self.has_unsaved_changes = True

    def _prompt_unsaved_changes(self, buttons=QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel):
        result = None
        if self.has_unsaved_changes:
            result = QMessageBox.question(self,
                                          'Unsaved changes',
                                          f"Do you want to save changes to {Path(self.editor.file.path).name or 'Untitled'}?",
                                          buttons=QMessageBox.StandardButtons(buttons),
                                          defaultButton=QMessageBox.Yes)

            if not result:
                raise ValueError("WHAT?!")

        return result

    def _prompt_save_path(self):
        name, filter = QFileDialog.getSaveFileName(filter=("Python (*.py)"))
        if not name:
            return False
        return name

    def new(self):
        if self._prompt_unsaved_changes() == QMessageBox.Cancel:
            return False
        self.editor.file = pymanagers.PyFileManager(self.editor)
        self.editor.setPlainText('')
        self.has_unsaved_changes = False
        return True

    def open(self):
        if self._prompt_unsaved_changes() == QMessageBox.Cancel:
            return False
        name, filter = QFileDialog.getOpenFileName(filter=("Python (*.py)"))
        if not name:
            return False
        self.editor.file = pymanagers.PyFileManager(self.editor)
        self.editor.file.open(name)
        self.has_unsaved_changes = False
        return True
        # with open(name, 'r') as f:
        #     state = f.read()
        # self.editor.setPlainText(state)

    def save(self):
        path = self.editor.file.path
        return self.save_as(path)

    def save_as(self, path=None):
        if not path:
            path = self._prompt_save_path()
            if not path: return False

        # state = self.editor.toPlainText()
        # with open(path, 'w') as f:
        #     f.write(state)

        self.editor.file.save(path)
        self.has_unsaved_changes = False
        # self.path = path
        return True

    def exit(self, buttons=QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel):
        if self._prompt_unsaved_changes(buttons) == QMessageBox.Cancel:
            return False

        self.editor.file.close()
        self.editor.backend.stop()
        self.has_unsaved_changes = False
        self.close()

        return True

    def closeEvent(self, event):
        if self.exit():
            event.accept()  # let the window close
        else:
            event.ignore()

    def close_by_quit(self):
        self.exit(buttons=QMessageBox.Yes | QMessageBox.No)

    def start_server(self):
        if self._prompt_unsaved_changes(buttons=QMessageBox.Yes | QMessageBox.Cancel) == QMessageBox.Cancel:
            return False
        self.save()
        self.main_window.start_server(path=self.editor.file.path)

    def stop_server(self):
        self.main_window.stop_server(confirm=True)
