import math


class MatrixIndexer(object):
    def __init__(self, mtx):
        self.mtx = mtx


    def __getitem__(self, location):
        row = location[0]
        col = location[1]
        if row >= self.mtx.shape[0] or col >= self.mtx.shape[1]:
            raise IndexError("Index out of range: {}".format(location))

        return self.mtx.values[col + row*self.mtx.shape[1]]
    

    def __setitem__(self, location, value):
        row = location[0]
        col = location[1]
        if row >= self.mtx.shape[0] or col >= self.mtx.shape[1]:
            raise IndexError("Index out of range: {}".format(location))

        self.mtx.values[col + row*self.mtx.shape[1]] = float(value)



class Matrix(object):

    def __init__(self, shape, values):
        self.values = list(float(v) for v in values)

        self.shape = shape

        num_values = shape[0]*shape[1]

        self.loc = MatrixIndexer(self)

        if len(self.values) > num_values:
            raise ValueError("Matrix: too many values for shape {}".format(shape))

        if len(self.values) < num_values:
            raise ValueError("Matrix: too few values for shape {}".format(shape))


    def row(self, row_num):
        if row_num >= self.shape[0]:
            raise ValueError("Matrix: cannot get row {} for shape {}".format(
                row_num, self.shape))

        return tuple(self.values[row_num*self.shape[1] : (1+row_num)*self.shape[1]])



    def column(self, col_num):
        if col_num >= self.shape[1]:
            raise ValueError("Matrix: cannot get column {} for shape {}".format(
                col_num, self.shape))

        return tuple(self.values[idx] for idx in xrange(col_num, len(self.values), self.shape[1]))



    def list_view(self):
        return self.values



    def __str__(self):
        st = ""
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                st += "{0:>12}".format(round(self.values[self.shape[1]*row + col], 5))
            st += "\n"
        return st[:-1]




    def multiply(self, right_operand):
        if right_operand.shape[0] != self.shape[1]:
            raise ValueError("Cannot multiply objects with shapes {} and {}".format(
                self.shape, right_operand.shape))

        res_shape = (self.shape[0], right_operand.shape[1])
        res_values = []

        for rown in range(self.shape[0]):
            for coln in range(right_operand.shape[1]):
                left_row = self.row(rown)
                right_col = right_operand.column(coln)
                val = 0.
                for idx in range(len(left_row)):
                    val += left_row[idx] * right_col[idx]

                res_values.append(val)

        return Matrix(res_shape, res_values)




    @staticmethod
    def identity(size):
        values = [0.] * (size*size)
        for idx in range(size):
            values[size*idx + idx] = 1.
        return Matrix((size, size), values)


    @staticmethod
    def zeros(size):
        values = [0.] * (size*size)
        return Matrix((size, size), values)


    @staticmethod
    def ones(size):
        values = [1.] * (size*size)
        return Matrix((size, size), values)




    @staticmethod
    def translate2d(vect):
        values = [
            1., 0., vect[0],
            0., 1., vect[1],
            0., 0., 1.
        ]
        return Matrix((3,3), values)



    @staticmethod
    def rotate2d(rotation_center, angle):
        cosa = math.cos(angle)
        sina = math.sin(angle)
        Cx = rotation_center[0]
        Cy = rotation_center[1]

        values = [
            cosa, -sina, -Cx*cosa + Cy*sina + Cx,
            sina, cosa, -Cx*sina - Cy*cosa + Cy,
            0, 0, 1.
        ]
        return Matrix((3,3), values)



    @staticmethod
    def scale2d(scale_center, scale_coef):
        Cx = scale_center[0]
        Cy = scale_center[1]

        Sx = scale_coef[0]
        Sy = scale_coef[1]

        values = [
            Sx, 0, Cx*(1. - Sx),
            0, Sy, Cy*(1. - Sy),
            0., 0., 1.
        ]
        return Matrix((3,3), values)



def add_rotation_to_mtx(mtx, angle):
    sinb = math.sin(angle)
    cosb = math.cos(angle)

    cosa = mtx.loc[(0, 0)]
    sina = mtx.loc[(0, 1)]

    # cos(a+b)
    cos_ab = cosa*cosb - sina*sinb

    # sin(a+b)
    sin_ab = sina*cosb + cosa*sinb

    mtx.loc[(0, 0)] = cos_ab
    mtx.loc[(0, 1)] = sin_ab
    mtx.loc[(1, 0)] = -sin_ab
    mtx.loc[(1, 1)] = cos_ab