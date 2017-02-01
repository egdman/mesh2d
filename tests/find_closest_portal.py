import sys
import os


here = os.path.abspath(os.path.dirname(__file__))

module_root = os.path.join(here, "..")
sys.path.append(module_root)

from mesh2d import Polygon2d, Vector2


indices = [0, 1, 2, 5, 6, 3, 7, 4]

vertices = [
	Vector2(0., 0.),
	Vector2(0.5, 0.5),
	Vector2(1., 0.),
	Vector2(1., 1.),
	Vector2(0., 1.),
	Vector2(1., 0.5),
	Vector2(1., 0.75),
	Vector2(0.75, 1.),
]

poly = Polygon2d(vertices, indices)


# should be outside
portal1 = {
	'start_index': 1,
	'end_index': 5,
	'end_point': vertices[5]
}

portal1r = {
	'start_index': 5,
	'end_index': 1,
	'end_point': vertices[1]
}



# should be outside
portal2 = {
	'start_index': 1,
	'end_index': 6,
	'end_point': vertices[6]
}

portal2r = {
	'start_index': 6,
	'end_index': 1,
	'end_point': vertices[1]
}



# border case
portal3 = {
	'start_index': 1,
	'end_index': 3,
	'end_point': vertices[3]
}

portal3r = {
	'start_index': 3,
	'end_index': 1,
	'end_point': vertices[1]
}




# should be inside
portal4 = {
	'start_index': 1,
	'end_index': 7,
	'end_point': vertices[7]
}

portal4r = {
	'start_index': 7,
	'end_index': 1,
	'end_point': vertices[1]
}


tip = vertices[1]
left = tip + Vector2(-1., 1.)
right = tip + Vector2(1., 1.)

print("my version:")
print("left: {}".format(left))
print("right: {}".format(right))


print("automatic:")
conevec1, conevec2 = poly.get_anticone(0, 1, 2, 0.0)
left = tip + conevec2
right = tip + conevec1
print("left: {}".format(left))
print("right: {}".format(right))


print("\n============================================================")
print("portal 1")
closest_portal, closest_portal_point, closest_portal_dst = \
	poly.find_closest_portal(left, tip, right, [portal1])

print(closest_portal)
print(closest_portal_point)
print(closest_portal_dst)


print("\n============================================================")
print("portal 1 reversed")
closest_portal, closest_portal_point, closest_portal_dst = \
	poly.find_closest_portal(left, tip, right, [portal1r])

print(closest_portal)
print(closest_portal_point)
print(closest_portal_dst)


print("\n============================================================")
print("portal 2")
closest_portal, closest_portal_point, closest_portal_dst = \
	poly.find_closest_portal(left, tip, right, [portal2])

print(closest_portal)
print(closest_portal_point)
print(closest_portal_dst)

print("\n============================================================")
print("portal 2 reversed")
closest_portal, closest_portal_point, closest_portal_dst = \
	poly.find_closest_portal(left, tip, right, [portal2r])

print(closest_portal)
print(closest_portal_point)
print(closest_portal_dst)

# import pudb; pu.db

print("\n============================================================")
print("portal 3")
closest_portal, closest_portal_point, closest_portal_dst = \
	poly.find_closest_portal(left, tip, right, [portal3])

print(closest_portal)
print(closest_portal_point)
print(closest_portal_dst)

print("\n============================================================")
print("portal 3 reversed")
closest_portal, closest_portal_point, closest_portal_dst = \
	poly.find_closest_portal(left, tip, right, [portal3r])

print(closest_portal)
print(closest_portal_point)
print(closest_portal_dst)


print("\n============================================================")
print("portal 4")
closest_portal, closest_portal_point, closest_portal_dst = \
	poly.find_closest_portal(left, tip, right, [portal4])

print(closest_portal)
print(closest_portal_point)
print(closest_portal_dst)

print("\n============================================================")
print("portal 4 reversed")
closest_portal, closest_portal_point, closest_portal_dst = \
	poly.find_closest_portal(left, tip, right, [portal4r])

print(closest_portal)
print(closest_portal_point)
print(closest_portal_dst)