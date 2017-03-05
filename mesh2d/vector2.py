import math


class LineRelation(object):
    def __init__(self, intersection, identical):
        self.intersection = intersection
        self.identical = identical


class Vector2:
    tolerance = 0.00001

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        # this is for multiplication with a Matrix
        self.shape = (3, 1)


    def copy(self):
        return Vector2(self.x, self.y)


    def __getitem__(self, key):
        if key == 0:
            return self.x
        if key == 1:
            return self.y
        if key == 2:
            return 1.0
        else:
            raise IndexError(key)


    # Matrix interface
    def row(self, row_num):
        if row_num > 2:
            raise ValueError("Vector2: cannot get row {}".format(row_num))
        return (self[row_num],)

    def column(self, col_num):
        if col_num != 0:
            raise ValueError("Vector2: cannot get column {}".format(col_num))
        return (self.x, self.y, 1.0)



    def __iter__(self):
        return iter((self.x, self.y))


    def __str__(self):
        return "{0},{1}".format(self.x, self.y)

    def __add__(self, right_operand):
        return Vector2(self.x + right_operand.x, self.y + right_operand.y)

    def __sub__(self, right_operand):
        return Vector2(self.x - right_operand.x, self.y - right_operand.y)


    # scalar mul with scalar on the right
    def __mul__(self, right_scalar):
        return Vector2(self.x * right_scalar, self.y * right_scalar)


    # scalar division
    def __div__(self, right_operand):
        return Vector2(self.x / right_operand, self.y / right_operand)


    # scalar mul with scalar on the left
    def __rmul__(self, left_scalar):
        return Vector2(self.x * left_scalar, self.y * left_scalar)


    # negation
    def __neg__(self):
        return Vector2(-self.x, -self.y)


    # equality
    def __eq__(self, right_operand):
        # return self.x == right_operand.x and self.y == right_operand.y
        return abs(self.x - right_operand.x) < Vector2.tolerance and \
            abs(self.y - right_operand.y) < Vector2.tolerance



    # inequality
    def __ne__(self, right_operand):
        # return self.x != right_operand.x or self.y != right_operand.y
        return not self == right_operand


    # for hashing support
    def __hash__(self):
        return hash((self.x, self.y))



    def length(self):
        return math.sqrt(self.dot_product(self))


    def snap_to(self, other):
        self.x = other.x
        self.y = other.y


    def dot_product(self, other):
        return self.x * other.x + self.y * other.y





    @staticmethod
    def cross(v1, v2):
        return v1.x * v2.y - v2.x * v1.y



    @staticmethod
    def double_signed_area(v1, v2, v3):
        return Vector2.cross(v2 - v1, v3 - v1) / 2.0



    @staticmethod
    def poly_signed_area(vertices):
        area = 0.0
        vertices = iter(vertices)
        begin_vrt = next(vertices)
        vert1 = begin_vrt
        for vert2 in vertices:
            area += (vert1.x - vert2.x) * (vert1.y + vert2.y)
            vert1 = vert2

        # wrap for last segment:
        vert2 = begin_vrt
        area += (vert1.x - vert2.x) * (vert1.y + vert2.y)
        return area / 2.0



    @staticmethod
    def are_points_ccw(v1, v2, v3):
        return Vector2.double_signed_area(v1, v2, v3) > 0


    @staticmethod
    def point_inside(v0, v1, v2, v3):
        """
        Returns True iff v0 is inside the [v1, v2, v3] triangle
        """
        triangle_ccw = Vector2.are_points_ccw(v1, v2, v3)
        return (Vector2.are_points_ccw(v0, v1, v2) == triangle_ccw) and \
               (Vector2.are_points_ccw(v0, v2, v3) == triangle_ccw) and \
               (Vector2.are_points_ccw(v0, v3, v1) == triangle_ccw)



    @staticmethod
    def distance(v0, v1):
        dx = v0.x - v1.x
        dy = v0.y - v1.y
        return math.sqrt(dx*dx + dy*dy)



    @staticmethod
    def project_to_line(vert, line1, line2):
        line_span = line2 - line1
        coef = (vert - line1).dot_product(line_span) / (line_span.dot_product(line_span))
        return line1 + (line_span * coef)



    @staticmethod
    def vertex_to_line_dist(vert, line1, line2):
        proj = Vector2.project_to_line(vert, line1, line2)
        return Vector2.distance(proj, vert)



    @staticmethod
    def angle(vect1, vect2):
        cos_angle = vect1.dot_product(vect2) / (vect1.length() * vect2.length())
        return math.acos(cos_angle)


    @staticmethod
    def signed_angle(vect1, vect2):
        signed_area = Vector2.cross(vect1, vect2)
        sin_angle = signed_area / (vect1.length() * vect2.length())
        return math.asin(sin_angle)



    @staticmethod
    def mul_mtx(matrix, vector):
        if len(matrix) != 4:
            raise ValueError("Matrix must be 2x2")
        x = matrix[0]*vector.x + matrix[1]*vector.y
        y = matrix[2]*vector.x + matrix[3]*vector.y
        return Vector2(x, y)



    @staticmethod
    def vertex_to_segment_dist(vert, seg1, seg2):
        proj_point = Vector2.project_to_line(vert, seg1, seg2)
        if Vector2.point_between(proj_point, seg1, seg2):
            dist = Vector2.distance(proj_point, vert)

        else:
            dist1 = Vector2.distance(seg1, vert)
            dist2 = Vector2.distance(seg2, vert)
            dist = min(dist1, dist2)
            proj_point = seg1 if dist1 < dist2 else seg2

        return dist, proj_point



    @staticmethod
    def point_between(vert, vert1, vert2):
        '''
        Tells whether vert is between vert1 and vert2.
        Assumes they are on the same straight line.
        This is exclusive version:
        if vert == vert1 or vert == vert2, returns False.
        '''
        if abs(vert1.x - vert2.x) > abs(vert1.y - vert2.y):
            xmin = min(vert1.x, vert2.x)
            xmax = max(vert1.x, vert2.x)
            return vert.x > xmin and vert.x < xmax

        else:
            ymin = min(vert1.y, vert2.y)
            ymax = max(vert1.y, vert2.y)
            return vert.y > ymin and vert.y < ymax
            



    @staticmethod
    def point_between_inclusive(vert, vert1, vert2):
        '''
        Tells whether vert is between vert1 and vert2.
        Assumes they are on the same straight line.
        This is inclusive version:
        If vert == vert1 or vert == vert2, returns True.
        '''
        if abs(vert1.x - vert2.x) > abs(vert1.y - vert2.y):
            xmin = min(vert1.x, vert2.x)
            xmax = max(vert1.x, vert2.x)
            return vert.x >= xmin and vert.x <= xmax

        else:
            ymin = min(vert1.y, vert2.y)
            ymax = max(vert1.y, vert2.y)
            return vert.y >= ymin and vert.y <= ymax




    @staticmethod
    def vertex_on_ray(vert, ray_tip, ray_target):
        return Vector2.point_between_inclusive(vert, ray_tip, ray_target) or \
        Vector2.point_between_inclusive(ray_target, ray_tip, vert)



    @staticmethod
    def lines_intersect(l11, l12, l21, l22):

        # starting points
        s1 = l11
        s2 = l21

        # direction vectors
        r1 = l12 - l11
        r2 = l22 - l21

        '''
        Need to find coef b where intersection = s2 + b*r2

        Solve w.r.t. b:
        b|r1 x r2| = |r1 x s1| - |r1 x s2|
        '''

        r1r2 = Vector2.cross(r1, r2)

        r1s1 = Vector2.cross(r1, s1)
        r1s2 = Vector2.cross(r1, s2)

        # this means that lines are parallel
        if r1r2 == 0:

            # this means that s2 lies on l1
            if r1s1 == r1s2:
                # the lines are identical (they overlap)
                return LineRelation(intersection=None, identical=True)
            else:
                # the lines are parallel
                return LineRelation(intersection=None, identical=False)

        # if lines are not parallel
        else:
            b = (r1s1 - r1s2) / r1r2
            return LineRelation(intersection=s2 + b*r2, identical=False)



    @staticmethod
    def where_segments_cross_inclusive(seg11, seg12, seg21, seg22):
        line_rel = Vector2.lines_intersect(seg11, seg12, seg21, seg22)

        line_x = line_rel.intersection
        lines_overlap =  line_rel.identical

        # if lines intersect
        if line_x is not None:
            # if intersection lies inside both segments (INCLUDING ENDPOINTS)
            if Vector2.point_between_inclusive(line_x, seg11, seg12) and \
               Vector2.point_between_inclusive(line_x, seg21, seg22):
                return line_x
            else:
                return None

        # if lines overlap fully
        elif lines_overlap:
            # order vertices by x
            if seg11.x > seg12.x: seg11, seg12 = seg12, seg11
            if seg21.x > seg22.x: seg21, seg22 = seg22, seg21

            # if segments only touch at endpoints
            if seg12 == seg21:
                return seg12

            elif seg11 == seg22:
                return seg11
                
            else:
                return None

        # if lines are parallel
        else:
            return None




    @staticmethod
    def where_segments_cross_exclusive(seg11, seg12, seg21, seg22):
        line_rel = Vector2.lines_intersect(seg11, seg12, seg21, seg22)

        line_x = line_rel.intersection
        lines_overlap =  line_rel.identical

        # if lines intersect
        if line_x is not None:
            # if intersection lies inside both segments (EXCLUDING ENDPOINTS)
            if Vector2.point_between(line_x, seg11, seg12) and \
               Vector2.point_between(line_x, seg21, seg22):
                return line_x
            else:
                return None



    @staticmethod
    def where_segment_crosses_ray(seg1, seg2, ray1, ray2):
        line_rel = Vector2.lines_intersect(seg1, seg2, ray1, ray2)

        line_x = line_rel.intersection
        lines_overlap =  line_rel.identical

        # if lines intersect
        if line_x is not None:
            # first check if X is inside the segment
            if Vector2.point_between_inclusive(line_x, seg1, seg2):

                # then check if X is on the ray
                if Vector2.vertex_on_ray(line_x, ray1, ray2):
                    return line_x
                else:
                    return None

            # if the X is outside the segment
            else:
                return None

        # if no intersection
        else:
            # if lines overlap (are identical)
            if lines_overlap:

                '''
                There is only one case in which single-point intersection happens:
                when they overlap exactly at the tip and nowhere else.
                '''
                if seg1 == ray1 and \
                    Vector2.point_between_inclusive(ray1, seg2, ray2):
                    return ray1.copy()

                if seg2 == ray1 and \
                    Vector2.point_between_inclusive(ray1, seg1, ray2):
                    return ray1.copy()


                # In all other cases they either overlap too much or not at all
                return None

            # if lines are parallel
            else:
                return None




class ZeroSegmentError(StandardError):
    def __init__(self, message, segment):
        self._mes = message
        self._seg = segment

    def message(self):
        return self._mes

    def segment(self):
        return self._seg