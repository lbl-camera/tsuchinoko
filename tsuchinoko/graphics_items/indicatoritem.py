from pyqtgraph import CurvePoint, CurveArrow


class BetterCurvePoint(CurvePoint):
    def event(self, ev):
        try:
            return super(BetterCurvePoint, self).event(ev)
        except IndexError:
            return True


class BetterCurveArrow(BetterCurvePoint, CurveArrow):
    pass
