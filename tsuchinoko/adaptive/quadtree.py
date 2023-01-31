import time
from typing import List

import numpy as np

from tsuchinoko.adaptive import Data
from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine, acquisition_functions
from tsuchinoko.utils.logging import log_time


class Point:
    """A point located at (x,y) in 2D space.

    Each Point object may be associated with a payload object.

    """

    def __init__(self, x, y, value=None, variance=None, metrics=None, acq_func_value=None):
        self.x, self.y = x, y
        self.value = value
        self.variance = variance
        self.metrics = metrics
        self.acq_func_value = acq_func_value

    def __repr__(self):
        return '{}: {}'.format(str((self.x, self.y)), repr([self.value, self.variance, self.metrics, self.acq_func_value]))
    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.x, self.y)

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)

class Rect:
    """A rectangle centred at (cx, cy) with width w and height h."""

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.west_edge, self.east_edge = cx - w/2, cx + w/2
        self.north_edge, self.south_edge = cy - h/2, cy + h/2

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                    self.north_edge, self.east_edge, self.south_edge)

    def contains(self, point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""

        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (point_x >= self.west_edge and
                point_x <  self.east_edge and
                point_y >= self.north_edge and
                point_y < self.south_edge)

    def intersects(self, other):
        """Does Rect object other interesect this Rect?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def draw(self, ax, c='k', lw=1, **kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)

    @property  # TODO: memoize
    def area(self):
        return self.w * self.h

    @property
    def center(self):
        return self.cx, self.cy


class QuadTree:
    """A class implementing a quadtree."""

    def __init__(self, boundary, max_points=4, depth=0):
        """Initialize this node of the quadtree.

        boundary is a Rect object defining the region from which points are
        placed into this node; max_points is the maximum number of points the
        node can hold before it must divide (branch into four more nodes);
        depth keeps track of how deep into the quadtree this node lies.

        """

        self.boundary = boundary
        self.max_points = max_points
        self.points = []
        self.depth = depth
        # A flag to indicate whether this node has divided (branched) or not.
        self.divided = False

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + '\n'
        s += sp + ', '.join(str(point) for point in self.points)
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
                sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def smallest_containing_quad(self, point:Point) -> 'QuadTree':
        if self.divided:
            for div in [self.nw, self.ne, self.se, self.sw]:
                if div.boundary.contains(point):
                    return div.smallest_containing_quad(point)
        elif self.boundary.contains(point):
            return self
        else:
            return None

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""

        cx, cy = self.boundary.cx, self.boundary.cy
        w, h = self.boundary.w / 2, self.boundary.h / 2
        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.
        self.nw = QuadTree(Rect(cx - w/2, cy - h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.ne = QuadTree(Rect(cx + w/2, cy - h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.se = QuadTree(Rect(cx + w/2, cy + h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.sw = QuadTree(Rect(cx - w/2, cy + h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.divided = True
        return self.divisions

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""

        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False
        if len(self.points) < self.max_points:
            # There's room for our point without dividing the QuadTree.
            self.points.append(point)
            return True

        # No room: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def query(self, boundary, found_points):
        """Find the points in the quadtree that lie within boundary."""

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary ...
        for point in self.points:
            if boundary.contains(point):
                found_points.append(point)
        # ... and if this node has children, search them too.
        if self.divided:
            self.nw.query(boundary, found_points)
            self.ne.query(boundary, found_points)
            self.se.query(boundary, found_points)
            self.sw.query(boundary, found_points)
        return found_points


    def query_circle(self, boundary, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre.

        boundary is a Rect object (a square) that bounds the search circle.
        There is no need to call this method directly: use query_radius.

        """

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary
        # and also lie within a circle of given radius around the centre point.
        for point in self.points:
            if (boundary.contains(point) and
                    point.distance_to(centre) <= radius):
                found_points.append(point)

        # Recurse the search into this node's children.
        if self.divided:
            self.nw.query_circle(boundary, centre, radius, found_points)
            self.ne.query_circle(boundary, centre, radius, found_points)
            self.se.query_circle(boundary, centre, radius, found_points)
            self.sw.query_circle(boundary, centre, radius, found_points)
        return found_points

    def query_radius(self, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre."""

        # First find the square that bounds the search circle as a Rect object.
        boundary = Rect(*centre, 2*radius, 2*radius)
        return self.query_circle(boundary, centre, radius, found_points)


    def __len__(self):
        """Return the number of points in the quadtree."""

        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
        return npoints

    def draw(self, ax):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""

        self.boundary.draw(ax)
        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)

    @property
    def divisions(self):
        if self.divided:
            return self.nw, self.ne, self.se, self.sw
        else:
            return []

    @property
    def children_points(self):
        yield from self.points
        for div in self.divisions:
            yield from div.children_points

class QuadTreeEngine(GPCAMInProcessEngine):

    def __init__(self, parameter_bounds, hyperparameters, hyperparameter_bounds, **kwargs):
        super(QuadTreeEngine, self).__init__(2, parameter_bounds, hyperparameters, hyperparameter_bounds, **kwargs)

    def reset(self):
        parameter_bounds = np.asarray([[self.parameters[('bounds', f'axis_{i}_{edge}')]
                                        for edge in ['min', 'max']]
                                       for i in range(self.dimensionality)])

        cx = (parameter_bounds[0][1] + parameter_bounds[0][0]) / 2
        cy = (parameter_bounds[1][1] + parameter_bounds[1][0]) / 2

        width = parameter_bounds[0][1] - parameter_bounds[0][0]
        height = parameter_bounds[1][1] - parameter_bounds[1][0]

        domain = Rect(cx, cy, width, height)
        self.quadtree = QuadTree(domain)
        self.target_queue: List[QuadTree] = [self.quadtree]
        super(QuadTreeEngine, self).reset()
        self._invalidate_all = False

        self._update_counter = 0

    def request_targets(self, position):
        if not self.target_queue:
            points = list(self.quadtree.children_points)
            acq_func_values = [point.acq_func_value for point in points]  # TODO: avoid list-comp
            # n_largest = np.argpartition(acq_func_values, -n)[-n:]
            largest_point = points[np.argmin(acq_func_values)]  # TODO: do nones affect this?
            target_quad = self.quadtree.smallest_containing_quad(largest_point)
            target_divisions = target_quad.divide()
            target_divisions = [div for div in target_divisions if not div.boundary.contains(largest_point)]

            self.target_queue = target_divisions

        return [(np.random.uniform(div.boundary.west_edge, div.boundary.east_edge),
                 np.random.uniform(div.boundary.south_edge, div.boundary.north_edge)) for div in self.target_queue]

    def update_measurements(self, data: Data):
        with data.r_lock():  # quickly grab values within lock before passing to optimizer
            positions = data.positions.copy()
            scores = data.scores.copy()
            variances = data.variances.copy()

        self.optimizer.tell(positions, scores, variances)

        for target_division in self.target_queue:
            for i, position in enumerate(positions[-4:]):  # only check last 3 points (dim**2 - 1)
                if target_division.boundary.contains(Point(*position)):
                    self.quadtree.insert(Point(*position, scores[-i-1], variances[-i-1]))
                    self.target_queue.remove(target_division)
                    break

        # TODO: this could be limited in range to only points nearby updated points



        positions = [(point.x, point.y) for point in self.quadtree.children_points]

        # calculate acquisition function
        with log_time('updating acq_func values', cumulative_key='updating acq_func values'):

            acquisition_function_values = self.optimizer.evaluate_acquisition_function(positions,
                                                                                      acquisition_function=
                                                                                      acquisition_functions[
                                                                                          self.parameters[
                                                                                              'acquisition_function']])

        for i, point in enumerate(self.quadtree.children_points):
            point.acq_func_value = acquisition_function_values[i]*self.quadtree.smallest_containing_quad(point).boundary.area

        if not len(data)%100:
            import matplotlib.pyplot as plt

            DPI = 72
            # np.random.seed(60)

            # width, height = image.shape[1], image.shape[0]

            N = 15000
            # coords = np.random.randn(N, 2) * height / 3 + (width / 2, height / 2)
            # values = [bilinear_sample(pos) for pos in coords]
            # points = [Point(*coord, payload=value) for coord, value in zip(coords, values)]

            # domain = Rect(width / 2, height / 2, width, height)
            # qtree = QuadTree(domain, 3)
            # for point in points:
            #     qtree.insert(point)

            # print('Number of points in the domain =', len(qtree))

            fig = plt.figure(figsize=(700 / DPI, 500 / DPI), dpi=DPI)
            ax = plt.subplot()
            # ax.set_xlim(0, width)
            # ax.set_ylim(0, height)
            self.quadtree.draw(ax)

            points = list(self.quadtree.children_points)
            ax.scatter([p.x for p in points], [p.y for p in points], s=400, c=[p.value for p in points], cmap='gray')
            # ax.set_xticks([])
            # ax.set_yticks([])

            # centre, radius = (width / 2, height / 2), 120
            # found_points = []
            # qtree.query_radius(centre, radius, found_points)
            # print('Number of found points =', len(found_points))

            # ax.scatter([p.x for p in found_points], [p.y for p in found_points],
            #            facecolors='none', edgecolors='r', s=32)

            # circle = plt.Circle(centre, radius, ec='r')
            # Rect(*centre, 2 * radius, 2 * radius).draw(ax, c='r')

            # ax.invert_yaxis()
            plt.tight_layout()
            # plt.savefig('search-quadtree-circle.png')#, DPI=72)
            plt.show()

    def update_metrics(self, data: Data):
        if self._update_counter>10:
            super().update_metrics(data)
            self._update_counter = 0
        else:
            self._update_counter += 1

    def train(self):
        self._invalidate_all = super().train()



if __name__ == '__main__':
    from scipy import ndimage
    from PIL import Image
    from pathlib import Path

    # Load data from a jpg image to be used as a luminosity map
    image = np.flipud(np.asarray(Image.open(Path(__file__).parent.parent.parent / 'examples' / 'sombrero_pug.jpg')))
    luminosity = np.average(image, axis=2)


    # Bilinear sampling will be used to effectively smooth pixel edges in source data
    def bilinear_sample(pos):

        return pos, ndimage.map_coordinates(luminosity, [[pos[1]], [pos[0]]], order=1)[0], 1, {}


    # Poisson noise applied to measured value; variance is set in accordance with poisson statistics
    def noisy_sample(pos):
        pos, value, variance, extra = bilinear_sample(pos)
        return pos, np.random.poisson(value), value, extra


    def gaussian_sample(pos):
        pos, value, variance, extra = bilinear_sample(pos)
        return pos, np.random.standard_normal(1)[0] * 30 + value, value, extra



    import numpy as np
    import matplotlib.pyplot as plt
    from quadtree import Point, Rect, QuadTree
    from matplotlib import gridspec

    DPI = 72
    # np.random.seed(60)

    width, height = image.shape[1], image.shape[0]

    N = 15000
    coords = np.random.randn(N, 2) * height / 3 + (width / 2, height / 2)
    values = [bilinear_sample(pos) for pos in coords]
    points = [Point(*coord, payload=value) for coord, value in zip(coords, values)]

    domain = Rect(width / 2, height / 2, width, height)
    qtree = QuadTree(domain, 3)
    for point in points:
        qtree.insert(point)

    print('Number of points in the domain =', len(qtree))

    fig = plt.figure(figsize=(700 / DPI, 500 / DPI), dpi=DPI)
    ax = plt.subplot()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    qtree.draw(ax)

    ax.scatter([p.x for p in points], [p.y for p in points], s=400, c=[p.payload[1] for p in points], cmap='gray')
    # ax.set_xticks([])
    # ax.set_yticks([])

    centre, radius = (width / 2, height / 2), 120
    found_points = []
    qtree.query_radius(centre, radius, found_points)
    print('Number of found points =', len(found_points))

    ax.scatter([p.x for p in found_points], [p.y for p in found_points],
               facecolors='none', edgecolors='r', s=32)

    circle = plt.Circle(centre, radius, ec='r')
    Rect(*centre, 2 * radius, 2 * radius).draw(ax, c='r')

    # ax.invert_yaxis()
    plt.tight_layout()
    # plt.savefig('search-quadtree-circle.png')#, DPI=72)
    plt.show()
