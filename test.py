def merge_loops(loop1, loop2, index1, index2):
	'''
	Assumes that index1 in loops1, index2 in loops2
	'''
	# shift loop1
	index1_at = loop1.index(index1)
	loop1 = loop1[index1_at:] + loop1[:index1_at] + [index1]

	print loop1

	index2_at = loop2.index(index2)
	res = loop2[:index2_at+1] + loop1 + loop2[index2_at:]
	print res
	return res




merge_loops([0, 11, 1, 2, 3], [7, 8, 9, 10], 3, 10)