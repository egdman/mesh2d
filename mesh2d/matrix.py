import math
import numpy as np

class Matrix(object):
    @staticmethod
    def column_vec(vector):
        return np.resize(np.array(vector), (len(vector), 1))


    @staticmethod
    def row_vec(vector):
        return np.resize(np.array(vector), (1, len(vector)))


    @staticmethod
    def identity(size):
        return np.identity(size)


    @staticmethod
    def zeros(size):
        return np.zeros((size, size))


    @staticmethod
    def ones(size):
        return np.ones((size, size))


    @staticmethod
    def translate2d(vect):
        return np.array([
            [1., 0., vect[0]],
            [0., 1., vect[1]],
            [0., 0.,      1.]
        ])


    @staticmethod
    def rotate2d(rotation_center, angle):
        cosa = math.cos(angle)
        sina = math.sin(angle)
        Cx, Cy = rotation_center

        return np.array([
            [cosa, -sina, -Cx*cosa + Cy*sina + Cx],
            [sina,  cosa, -Cx*sina - Cy*cosa + Cy],
            [   0,     0,                       1]
        ])


    @staticmethod
    def scale2d(scale_center, scale_coef):
        Cx, Cy = scale_center
        Sx, Sy = scale_coef

        return np.array([
            [Sx, 0., Cx*(1. - Sx)],
            [0., Sy, Cy*(1. - Sy)],
            [0., 0.,           1.]
        ])
