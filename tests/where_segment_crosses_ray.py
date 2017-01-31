import sys
import os

# import pudb; pu.db

here = os.path.abspath(os.path.dirname(__file__))

module_root = os.path.join(here, "..")
sys.path.append(module_root)

from mesh2d import Polygon2d, Vector2


def test_case(msg, seg1, seg2, ray1, ray2):
	print("------------------------------------------------------------------------------")
	print(msg)
	print("seg1 = {}\nseg2 = {}\nray tip = {}\nray target = {}".format(
		seg1, seg2, ray1, ray2))
	print("")
	try:
		print(Vector2.where_segment_crosses_ray(seg1, seg2, ray1, ray2))

	except StandardError as er:
		print("ERROR: {}".format(er))

	print("------------------------------------------------------------------------------\n")



seg1 = Vector2(0.75, 1.0)
seg2 = Vector2(0.5, 0.5)
tip = Vector2(0.5, 0.5)
left = Vector2(-0.3, 1.3)
right = Vector2(1.3, 1.3)

print(Vector2.where_segment_crosses_ray(seg1, seg2, tip, left))
print(Vector2.where_segment_crosses_ray(seg1, seg2, tip, right))


test_case(
"check what happens when segment lies on the ray behind the tip:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.15, 0.15),
seg2 = Vector2(0.45, 0.45),
)



test_case(
"check what happens when segment lies on the ray in front of the tip:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.55, 0.55),
seg2 = Vector2(0.75, 0.75)
)




test_case(
"check what happens when segment lies on the ray and hits the ray tip exactly:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.75, 0.75)
)


test_case(
"check what happens when segment lies on the ray and hits the ray target exactly:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.55, 0.55),
seg2 = Vector2(1.5, 1.5)
)



test_case(
"check what happens when segment lies on the ray and hits both the ray tip and target exactly:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(1.5, 1.5)
)



test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly (#1):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.5, 1.0)
)



test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly (#2):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.25, 1.0)
)


test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly (#3):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.0, 1.0)
)



test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly (#4):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.0, 0.5)
)



test_case(
"check what happens when segment does not lie on the ray and hits the ray tip exactly (#5):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.5, 0.5),
seg2 = Vector2(0.0, 0.1)
)



test_case(
"check what happens when segment lies on the ray, hits the ray tip exactly, \
and goes in the exact opposite direction from the ray:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.0, 0.0),
seg2 = Vector2(0.5, 0.5)
)



test_case(
"check what happens when segment partially coincides with the ray (ray tip lies between seg1 and seg2):",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.7, 0.7),
seg2 = Vector2(0.3, 0.3)
)


test_case(
"check what happens when intersection point hits the ray tip exactly, but does not coincide with seg1 or seg2:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.2, 0.8),
seg2 = Vector2(0.9, 0.1)
)


test_case(
"check what happens when intersection point hits the ray tip almost exactly:",
ray1 = Vector2(0.5, 0.5), # tip
ray2 = Vector2(1.5, 1.5), # target
seg1 = Vector2(0.2, 0.8),
seg2 = Vector2(0.9, 0.099999999999999929224)
)






