import math

class Matrix(object):

	def __init__(self, shape, values):
		self.values = list(float(v) for v in values)

		self.shape = shape

		num_values = shape[0]*shape[1]

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


	def __getitem__(self, key):
		return self.values[key]


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
	def transform2d(translation, rotation):
		cosr = math.cos(rotation)
		sinr = math.sin(rotation)
		values = [
			cosr, -sinr, translation[0],
			sinr, cosr, translation[1],
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