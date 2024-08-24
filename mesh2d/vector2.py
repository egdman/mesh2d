import math
from itertools import chain, tee

try:
    from itertools import izip as zip
except ImportError:
    pass

def pairs(iterable):
    a, b = tee(iterable, 2)
    first = next(b, None)
    return zip(a, chain(b, [first]))


class vec:
    """
    vector of arbitrary size
    """
    def __init__(self, *comps):
        self.comps = tuple(float(c) for c in comps)

    def dot(self, right):
        return sum((c0 * c1 for (c0, c1) in zip(self.comps, right.comps)))

    def normSq(self):
        return self.dot(self)

    def norm(self):
        return math.sqrt(self.dot(self))

    # multiply by a scalar on the right
    def __mul__(self, right_scalar):
        return vec(*(c * right_scalar for c in self.comps))

    # multiply by a scalar on the left
    def __rmul__(self, left_scalar):
        return vec(*(left_scalar * c for c in self.comps))

    # negate
    def __neg__(self):
        return vec(*(-c for c in self.comps))

    # add vector
    def __add__(self, right):
        return vec(*(c0 + c1 for (c0, c1) in zip(self.comps, right.comps)))

    # subtract vector
    def __sub__(self, right):
        return vec(*(c0 - c1 for (c0, c1) in zip(self.comps, right.comps)))

    # scalar division
    def __truediv__(self, right_scalar):
        a = 1. / right_scalar
        return vec(*(a * c for c in self.comps))

    # [] getter
    def __getitem__(self, key):
        return self.comps[key]

    # equality test
    def __eq__(self, right):
        return self.comps == right.comps

    # inequality test
    def __ne__(self, right):
        return self.comps != right.comps

    # hashing support
    def __hash__(self):
        return hash(self.comps)

    def __len__(self):
        return len(self.comps)

    def __repr__(self):
        return self.comps.__repr__()

    def normalized(self):
        a = 1. / math.sqrt(self.dot(self))
        return vec(*(a * c for c in self.comps))

    def append(self, *tail):
        return vec(*chain(self.comps, tail))

    def prepend(self, *head):
        return vec(*chain(head, self.comps))

    @staticmethod
    def aabb(points):
        """
        returns min and max corners of the axis-aligned bounding box of points
        """
        points = iter(points)
        p_min = list(next(points).comps)
        p_max = p_min[:]
        ndim = len(p_min)

        for point in points:
            for dim in range(ndim):
                component = point[dim]
                p_min[dim] = min(p_min[dim], component)
                p_max[dim] = max(p_max[dim], component)
        return vec(*p_min), vec(*p_max)



    @staticmethod
    def cross3(u, v):
        """
        returns 3d vector
        requires at least 3d vectors
        """
        return vec(
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])



    @staticmethod
    def cross2(u, v):
        """
        returns scalar
        requires at least 2d vectors
        """
        return u[0] * v[1] - u[1] * v[0]



class Geom2:
    """
    some functions for 2d geometry
    """

    @staticmethod
    def poly_signed_area(vertices):
        area = 0.
        for a, b in pairs(vertices):
            area += (a[0] - b[0]) * (a[1] + b[1])

        return .5 * area


    @staticmethod
    def project_to_line(point, line):
        """
        returns scalar parameter of projected point
        line must be a Ray-like object
        """
        line_start, line_guide = line
        return (point - line_start).dot(line_guide) / line_guide.dot(line_guide)


    @staticmethod
    def cos_angle(vect1, vect2):
        cosine = vect1.normalized().dot(vect2.normalized())
        return min(max(cosine, -1.), 1.)


    @staticmethod
    def sin_angle(vect1, vect2):
        sine = vec.cross2(vect1.normalized(), vect2.normalized())
        return min(max(sine, -1.), 1.)


    @staticmethod
    def is_origin_inside_polyline(polyline):
        inside = False

        polyline = iter(polyline)
        a = next(polyline, (0, 0))

        for b in chain(polyline, (a,)):
            # see if (a, b) intersects the {x>0, y=0} ray
            if a[1] >= 0:
                if b[1] < 0 and a[0] * b[1] < a[1] * b[0]:
                    inside = not inside

            elif b[1] >= 0 and a[0] * b[1] > a[1] * b[0]:
                inside = not inside

            a = b

        return inside


    @staticmethod
    def is_point_inside_polyline(point, polyline):
        return is_origin_inside_polyline((p - point for p in polyline))
