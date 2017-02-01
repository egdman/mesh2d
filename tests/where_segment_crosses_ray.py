import sys
import os

# import pudb; pu.db

here = os.path.abspath(os.path.dirname(__file__))

module_root = os.path.join(here, "..")
sys.path.append(module_root)

from mesh2d import Polygon2d, Vector2


num_fails = 0


def vector_equal(result_point, correct_point):
	if correct_point is None: return result_point is None
	if result_point is None: return correct_point is None
	return correct_point == result_point



def test_case(msg, seg1, seg2, ray1, ray2, crosses, cross_point):
	global num_fails
	report = ""
	
	def error_out():
		return "seg1 = {}\nseg2 = {}\nray tip = {}\nray target = {}".format(
			seg1, seg2, ray1, ray2)

	
	passed = True

	report_results = ""




	# report_results += "      segment_crosses_ray: "
	# try:
	# 	crosses_result = Vector2.segment_crosses_ray(seg1, seg2, ray1, ray2)
	# 	report_results += str(crosses_result) + "\n"

	# 	if crosses_result != crosses:
	# 		passed = False
	# 		report_results = report_results[:-1] + " (must be {})\n".format(crosses)


	# except StandardError as er:
	# 	report_results += "ERROR: {}".format(er) + "\n"
	# 	passed = False






	report_results += "where_segment_crosses_ray: "
	try:
		cross_point_result = Vector2.where_segment_crosses_ray(seg1, seg2, ray1, ray2)
		report_results += str(cross_point_result) + "\n"

		if not vector_equal(cross_point_result, cross_point):
			passed = False 
			report_results = report_results[:-1] + " (must be {})\n".format(cross_point)


	except StandardError as er:
		report_results += "ERROR: {}".format(er) + "\n"
		passed = False





	# make report
	report += "------------------------------------------------------------------------------\n"
	report += msg + "\n"

	if not passed:
		num_fails += 1

		report += "FAIL\n"
		report += error_out() + "\n"

	else:
		report += "OK\n"

	report += report_results
	report += "------------------------------------------------------------------------------\n\n"

	print report



# seg1 = Vector2(0.75, 1.0)
# seg2 = Vector2(0.5, 0.5)
# tip = Vector2(0.5, 0.5)
# left = Vector2(-0.3, 1.3)
# right = Vector2(1.3, 1.3)

# print(Vector2.where_segment_crosses_ray(seg1, seg2, tip, left))
# print(Vector2.where_segment_crosses_ray(seg1, seg2, tip, right))


test_case(
"check what happens when segment lies on the ray behind the tip:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.15, 0.15),
seg2 = Vector2(0.45, 0.45),

crosses = False,
cross_point = None
)


test_case(
"check what happens when segment lies on the ray behind the tip (seg flipped):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.45, 0.45),
seg2 = Vector2(0.15, 0.15),

crosses = False,
cross_point = None
)


test_case(
"check what happens when segment lies on the ray in front of the tip:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.55, 0.55),
seg2 = Vector2(0.75, 0.75),

crosses = False,
cross_point = None
)


test_case(
"check what happens when segment lies on the ray in front of the tip (seg flipped):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.75, 0.75),
seg2 = Vector2(0.55, 0.55),

crosses = False,
cross_point = None
)



test_case(
"check what happens when segment lies on the ray and hits the ray tip exactly:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.75, 0.75),

crosses = False,
cross_point = None
)


test_case(
"check what happens when segment lies on the ray and hits the ray target exactly:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.55, 0.55),
seg2 = Vector2(1.5, 1.5),

crosses = False,
cross_point = None
)



test_case(
"check what happens when segment lies on the ray and hits both the ray tip and target exactly:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(1.5, 1.5),

crosses = False,
cross_point = None
)


print("########################## ########################## ##########################")

test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly \
(segment to the left, sloped forward):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.75, 1.0),

crosses = True,
cross_point = Vector2(0.5, 0.5)
)



test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly \
(segment to the left, vertical):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.5, 1.0),

crosses = True,
cross_point = Vector2(0.5, 0.5)
)



test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly \
(segment to the left, sloped backwards):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.25, 1.0),

crosses = True,
cross_point = Vector2(0.5, 0.5)
)


test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly \
(segment to the left, horizontal):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.0, 0.5),

crosses = True,
cross_point = Vector2(0.5, 0.5)
)



test_case(
"check what happens when segment lies on the ray, hits the ray tip exactly, \
and goes in the exact opposite direction from the ray:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.0, 0.0),

crosses = True,
cross_point = Vector2(0.5, 0.5)
)


test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly \
(segment to the right, vertical):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.5, 0.0),

crosses = True,
cross_point = Vector2(0.5, 0.5)
)


test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly \
(segment to the right, sloped backwards):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.75, 0.0),

crosses = True,
cross_point = Vector2(0.5, 0.5)
)


test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly \
(segment to the right, horizontal):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(1.0, 0.5),

crosses = True,
cross_point = Vector2(0.5, 0.5)
)


test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly \
(segment to the right, sloped forward):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(1.0, 0.75),

crosses = True,
cross_point = Vector2(0.5, 0.5)
)



print("########################## ########################## ##########################")

test_case(
"check what happens when one of segment endpoints lies on the ray\
(segment to the left, sloped forward):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(1., 1.),
seg2 = Vector2(1.25, 1.5),

crosses = True,
cross_point = Vector2(1., 1.)
)


test_case(
"check what happens when one of segment endpoints lies on the ray\
(segment to the left, vertical):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(1., 1.),
seg2 = Vector2(1., 1.5),

crosses = True,
cross_point = Vector2(1., 1.)
)


test_case(
"check what happens when one of segment endpoints lies on the ray\
(segment to the right, vertical):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(1., 1.),
seg2 = Vector2(1., 0.),

crosses = True,
cross_point = Vector2(1., 1.)
)



test_case(
"check what happens when one of segment endpoints lies on the ray\
(segment to the right, sloped backwards):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(1., 1.),
seg2 = Vector2(1.5, 0.),

crosses = True,
cross_point = Vector2(1., 1.)
)


test_case(
"check what happens when one of segment endpoints lies on the ray\
(segment to the right, sloped forward):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(1., 1.),
seg2 = Vector2(1.5, 1.25),

crosses = True,
cross_point = Vector2(1., 1.)
)

## NOW THE SAME WITH SHORTER RAY
print("########################## ########################## ##########################")


test_case(
"check what happens when one of segment endpoints lies on the ray\
(segment to the left, sloped forward) (shorter ray):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(0.6, 0.6), # target
seg1 = Vector2(1., 1.),
seg2 = Vector2(1.25, 1.5),

crosses = True,
cross_point = Vector2(1., 1.)
)


test_case(
"check what happens when one of segment endpoints lies on the ray\
(segment to the left, vertical) (shorter ray):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(0.6, 0.6), # target
seg1 = Vector2(1., 1.),
seg2 = Vector2(1., 1.5),

crosses = True,
cross_point = Vector2(1., 1.)
)


test_case(
"check what happens when one of segment endpoints lies on the ray\
(segment to the right, vertical) (shorter ray):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(0.6, 0.6), # target
seg1 = Vector2(1., 1.),
seg2 = Vector2(1., 0.),

crosses = True,
cross_point = Vector2(1., 1.)
)



test_case(
"check what happens when one of segment endpoints lies on the ray\
(segment to the right, sloped backwards) (shorter ray):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(0.6, 0.6), # target
seg1 = Vector2(1., 1.),
seg2 = Vector2(1.5, 0.),

crosses = True,
cross_point = Vector2(1., 1.)
)


test_case(
"check what happens when one of segment endpoints lies on the ray\
(segment to the right, sloped forward) (shorter ray):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(0.6, 0.6), # target
seg1 = Vector2(1., 1.),
seg2 = Vector2(1.5, 1.25),

crosses = True,
cross_point = Vector2(1., 1.)
)


print("########################## ########################## ##########################")


test_case(
"check what happens when segment partially overlaps with the ray (ray tip lies between seg1 and seg2):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.7, 0.7),
seg2 = Vector2(0.3, 0.3),

crosses = False,
cross_point = None
)


test_case(
"check what happens when intersection point hits the ray tip exactly, but does not coincide with seg1 or seg2:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.2, 0.8),
seg2 = Vector2(0.9, 0.1),

crosses = True,
cross_point = Vector2(0.5, 0.5)
)



test_case(
"check what happens when the continuation of the segment intersects the ray exactly at the tip:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.2, 0.8),
seg2 = Vector2(0.3, 0.7),

crosses = False,
cross_point = None
)


test_case(
"check what happens when segment and ray are both vertical and they do not coincide:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(0.5, 1.5), # target
seg1 = Vector2(0.6, 0.0),
seg2 = Vector2(0.6, 1.0),

crosses = False,
cross_point = None
)


test_case(
"check what happens when segment and ray are both vertical and they coincide:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(0.5, 1.5), # target
seg1 = Vector2(0.5, 0.0),
seg2 = Vector2(0.5, 1.0),

crosses = False,
cross_point = None
)




print("{} TESTS FAILED".format(num_fails))