import math

class Matrix(object):

	def __init__(self, shape, values):
		self.values = list(float(v) for v in values)

		self.height = shape[0]
		self.width = shape[1]
		self.shape = shape

		num_values = self.height*self.width

		if len(self.values) > num_values:
			raise ValueError("Matrix: too many values for shape {}".format(shape))

		if len(self.values) < num_values:
			raise ValueError("Matrix: too few values for shape {}".format(shape))


	def row(self, row_num):
		if row_num >= self.height:
			raise ValueError("Matrix: cannot get row {} for shape {}".format(
				row_num, self.height, self.width))

		return tuple(self.values[row_num*self.width : (1+row_num)*self.width])



	def column(self, col_num):
		if col_num >= self.width:
			raise ValueError("Matrix: cannot get column {} for shape {}".format(
				col_num, self.height, self.width))

		return tuple(self.values[idx] for idx in xrange(col_num, len(self.values), self.width))



	def list_view(self):
		return self.values



	def __str__(self):
		st = ""
		for row in range(self.height):
			for col in range(self.width):
				st += "{0:>10.5f}".format(self.values[self.width*row + col])
			st += "\n"
		return st[:-1]



	def multiply(self, right_operand):
		right_h = right_operand.shape[0]
		if right_h != self.width:
			raise ValueError("Cannot multiply objects with shapes {} and {}".format(
				self.shape, right_operand.shape))

		res_shape = (self.height, right_operand.width)
		res_values = []

		for rown in range(self.height):
			for coln in range(right_operand.width):
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