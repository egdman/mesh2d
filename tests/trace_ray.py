import sys
import os
from random import random

here = os.path.abspath(os.path.dirname(__file__))

module_root = os.path.join(here, "..")
sys.path.append(module_root)

from mesh2d import Mesh2d, Vector2


def test_case(NW, SE):
	start = NW
	end = SE

	v0 = start
	v1 = Vector2(end.x, start.y)
	v2 = end
	v3 = Vector2(start.x, end.y)

	width = end.x - start.x
	height = end.y - start.y
	cntr_x = start.x + width / 2.
	cntr_y = start.y + height / 2.
	cntr = Vector2(cntr_x, cntr_y)

	sz = min(height, width) / 4.

	h10 = (Vector2(.1, .9) - Vector2(.5, .5) )*sz + cntr #4
	h11 = (Vector2(.1, .1) - Vector2(.5, .5) )*sz + cntr #5
	h12 = (Vector2(.4, .1) - Vector2(.5, .5) )*sz + cntr #6
	h13 = (Vector2(.4, .9) - Vector2(.5, .5) )*sz + cntr #7

	h20 = (Vector2(.9, .9) - Vector2(.5, .5) )*sz + cntr #8
	h21 = (Vector2(.5, .5) - Vector2(.5, .5) )*sz + cntr #9
	h22 = (Vector2(.9, .1) - Vector2(.5, .5) )*sz + cntr #10


	poly = Mesh2d([v0, v1, v2, v3], range(4))
	poly.add_hole([h10, h11, h12, h13])
	poly.add_hole([h20, h21, h22])

	ray1 = poly.vertices[10].copy()
	# ray2 = ray1 + Vector2(0., -1.)

	ray1 = Vector2(54.8, -124.8)
	ray2 = Vector2(54.8, -313.0)


	inters = poly.trace_ray(ray1, ray2)

	edge = inters[1]
	if edge != (0,1): return False, inters
	return True, inters





nw = Vector2(-211, -313)
se = Vector2(232, 152)

result, inters = test_case(nw, se)
print("inters: {}, on edge {}".format(inters[0], inters[1]))




# do test after test forever

# rnd_size = 100.

# min_size = 0.1

# start_nw = 0
# end_nw = rnd_size + min_size


# count = 0
# while True:
# 	count += 1
# 	start_pt = Vector2(start_nw + random()*rnd_size, start_nw + random()*rnd_size)
# 	end_pt = Vector2(end_nw + random()*rnd_size, end_nw + random()*rnd_size)

# 	result, inters = test_case(start_pt, end_pt)
# 	if not result:

# 		print("NW = {}, SE = {}".format(start_pt, end_pt))
# 		print("inters: {}, on edge {}".format(inters[0], inters[1]))

# 	if count % 10000 == 0: print("ran {} tests".format(count))

