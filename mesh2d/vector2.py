import math
from itertools import izip, chain
from operator import itemgetter
from .utils import pairs


# vector af arbitrary size
class vec(object):
    tolerance = 1e-8
    tolSq = tolerance * tolerance

    def __init__(self, *comps):
        self.comps = tuple(float(c) for c in comps)

    def dot(self, right):
        return sum((c0 * c1 for (c0, c1) in izip(self.comps, right.comps)))

    def normSq(self):
        return self.dot(self)

    def norm(self):
        return math.sqrt(self.normSq())

    # multiply by a scalar on the right
    def __mul__(self, right_scalar):
        return vec(*(c * right_scalar for c in self.comps))

    # multiply by a scalar on the left
    def __rmul__(self, left_scalar):
        return vec(*(left_scalar * c for c in self.comps))

    # negate
    def __neg__(self):
        return -1. * self

    # add vector
    def __add__(self, right):
        return vec(*(c0 + c1 for (c0, c1) in izip(self.comps, right.comps)))

    # subtract vector
    def __sub__(self, right):
        return self + (-right)

    # scalar division
    def __div__(self, right_scalar):
        return (1. / right_scalar) * self

    # [] getter
    def __getitem__(self, key):
        return self.comps[key]

    # equality test
    def __eq__(self, right):
        return hash(self) == hash(right)

    # inequality test
    def __ne__(self, right):
        return not self == right

    # hashing support
    def __hash__(self):
        return hash(self.comps)

    def __len__(self):
        return len(self.comps)

    def __repr__(self):
        return self.comps.__repr__()

    def normalized(self):
        return self / self.norm()

    def append(self, *tail):
        return vec(*chain(self.comps, tail))

    def prepend(self, *head):
        return vec(*chain(head, self.comps))

    def is_closer(self, other, dist = tolerance):
        return (self - other).norm() < dist


    # returns min and max corners of the axis-aligned bounding box of points
    @staticmethod
    def aabb(*points):
        ndim = len(points[0].comps)
        min_corner, max_corner = [], []
        for dim in range(ndim):
            min_corner.append(min(points, key = itemgetter(dim)) [dim])
            max_corner.append(max(points, key = itemgetter(dim)) [dim])
        return (vec(*min_corner), vec(*max_corner))


    # returns 3d vector
    # requires at least 3d vectors
    @staticmethod
    def cross3(u, v):
        return vec(
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])

    # returns scalar
    # requires at least 2d vectors
    @staticmethod
    def cross2(u, v):
        return u[0] * v[1] - u[1] * v[0]


# some functions for 2d geometry
class Geom2:

    @staticmethod
    def signed_area(v1, v2, v3):
        return .5 * vec.cross2(v2 - v1, v3 - v1)


    @staticmethod
    def poly_signed_area(vertices):
        area = 0.
        vertices = chain(vertices, vertices[:1])

        for vert1, vert2 in pairs(vertices):
            area += (vert1[0] - vert2[0]) * (vert1[1] + vert2[1])

        return .5 * area


    @staticmethod
    def are_points_ccw(v1, v2, v3):
        return Geom2.signed_area(v1, v2, v3) > 0


    @staticmethod
    def point_inside(v0, v1, v2, v3):
        """
        Returns True iff v0 is inside the [v1, v2, v3] triangle
        """
        triangle_ccw = Geom2.are_points_ccw(v1, v2, v3)
        return (Geom2.are_points_ccw(v0, v1, v2) == triangle_ccw) and \
               (Geom2.are_points_ccw(v0, v2, v3) == triangle_ccw) and \
               (Geom2.are_points_ccw(v0, v3, v1) == triangle_ccw)


    # returns scalar parameter of projected point
    # line must be a Ray-like object
    @staticmethod
    def project_to_line(point, line):
        line_start = line[0]
        line_guide = line[1]
        return (point - line_start).dot(line_guide) / line_guide.dot(line_guide)


    # line must be a Ray-like object
    @staticmethod
    def point_to_line_distSq(point, line):
        coef = Geom2.project_to_line(point, line)
        return (line[0] + (coef * line[1]) - point).normSq()


    # line must be a Ray-like object
    @staticmethod
    def point_to_line_dist(point, line):
        return math.sqrt(Geom2.point_to_line_distSq(point, line))


    @staticmethod
    def cos_angle(vect1, vect2):
        cosine = vect1.normalized().dot(vect2.normalized())
        return min(max(cosine, -1.), 1.)

    @staticmethod
    def sin_angle(vect1, vect2):
        sine = vec.cross2(vect1.normalized(), vect2.normalized())
        return min(max(sine, -1.), 1.)


    @staticmethod
    def mul_mtx_2x2(mtx, vector):
        return vec(
            mtx[0]*vector[0] + mtx[1]*vector[1],
            mtx[2]*vector[0] + mtx[3]*vector[1])


    # returns a tuple (coef1, coef2, distSq)
    # coef1 and coef2 are scalars that define the intersection point on line1 and line2 respectively
    # if lines are parallel, coef1 and coef2 are NaN, but distSq is still correct
    # if lines are non-parallel, distSq is 0
    # line1 and line2 must be Ray-like objects
    @staticmethod
    def lines_intersect(line1, line2, angle_tolerance = 1e-8):
        '''
        s1 = line1[0]
        r1 = line1[1]

        s2 = line2[0]
        r2 = line2[1]

        Need to find coef a where intersection = s1 + a*r1
        Need to find coef b where intersection = s2 + b*r2

        s1 + a*r1 = s2 + b*r2
        [r1 x s1] = [r1 x s2] + b * [r1 x r2]
        [r2 x s1] + a * [r2 x r1] = [r2 x s2]

        a = ([r2 x s2] - [r2 x s1]) / [r2 x r1]
        b = ([r1 x s1] - [r1 x s2]) / [r1 x r2]
        '''

        r2s2 = vec.cross2(line2[1], line2[0])
        r2s1 = vec.cross2(line2[1], line1[0])
        r2r1 = vec.cross2(line2[1], line1[1])

        r1s1 = vec.cross2(line1[1], line1[0])
        r1s2 = vec.cross2(line1[1], line2[0])
        r1r2 = - r2r1

        appr_angle = abs(r1r2) / (line1[1].norm() * line2[1].norm())

        # if lines are parallel
        if appr_angle < angle_tolerance:
            return (float('nan'), float('nan'), Geom2.point_to_line_distSq(line1[0], line2))

        else:
            a = (r2s2 - r2s1) / r2r1
            b = (r1s1 - r1s2) / r1r2
            return (a, b, 0)
