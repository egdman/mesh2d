import sys
import os

here = os.path.abspath(os.path.dirname(__file__))

module_root = os.path.join(here, "..")
sys.path.append(module_root)

from mesh2d import Mesh2d, Vector2


def poly_repr(poly):
	outl = poly.outline_coordinates()

	crds = list((outl[2*pos], outl[2*pos+1]) for pos in range(len(outl) / 2))
	st = ''
	st += 'indices:  ' + repr(poly.outline)
	st += '\n'
	st += 'outline:  ' + repr(crds)
	return st 


a = Vector2(0., 0.)
b = Vector2(1., 0.)
c = Vector2(1., 1.)
d = Vector2(0., 1.)

i1 = Vector2(0.5, 0.25)
i2 = Vector2(0.75, 0.5)
i3 = Vector2(0.5, 0.75)
i4 = Vector2(0.25, 0.5)

vertices = [a, b, c, d]

poly = Mesh2d(vertices[:], range(len(vertices)))

print(poly_repr(poly))
print("////////////////////////////////////////////////////////////////////\n\n")




poly = Mesh2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_outline(i1, (0,1))
print(poly_repr(poly))
print("----------------------------------")

poly = Mesh2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_outline(i1, (1,0))
print(poly_repr(poly))
print("==================================\n")




poly = Mesh2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_outline(i2, (1,2))
print(poly_repr(poly))
print("----------------------------------")

poly = Mesh2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_outline(i2, (2,1))
print(poly_repr(poly))
print("==================================\n")




poly = Mesh2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_outline(i3, (2,3))
print(poly_repr(poly))
print("----------------------------------")

poly = Mesh2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_outline(i3, (3,2))
print(poly_repr(poly))
print("==================================\n")




poly = Mesh2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_outline(i4, (3,0))
print(poly_repr(poly))
print("----------------------------------")

poly = Mesh2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_outline(i4, (0,3))
print(poly_repr(poly))
print("==================================\n")




# test area of polygon
print("Testing area of polygon")
a = Vector2(54., 154.)
b = Vector2(55., 154.)
c = Vector2(55., 155.)
d = Vector2(54., 155.)
vertices = [a, b, c, d]

print ("Area = {} (must be 1.0)".format(Vector2.poly_signed_area(vertices)))
print ("----------------------------------")

a = Vector2(-2., -2.)
b = Vector2(-2., 0.)
c = Vector2(0., 0.)
d = Vector2(0., -2.)
vertices = [a, b, c, d]

print ("Area = {} (must be -4.0)".format(Vector2.poly_signed_area(vertices)))
print ("----------------------------------")

# check that works with generator
print ("Area = {} (must be 4.0)".format(Vector2.poly_signed_area(
	(v for v in reversed(vertices))
)))
print ("----------------------------------")


# TESTING split index buffer
print("TESTING split index buffer")
ib = [8, 10, 5, 1, 15, 19, 6]


try:
	p1, p2 = Mesh2d._split_index_buffer(ib, 5, 19)
	print(p1)
	print(p2)
	print("-----------------")

except ValueError as ve:
	print (ve)


try:
	p1, p2 = Mesh2d._split_index_buffer(ib, 8, 10)
	print(p1)
	print(p2)
	print("-----------------")
except ValueError as ve:
	print (ve)


try:
	p1, p2 = Mesh2d._split_index_buffer(ib, 5, 5)
	print(p1)
	print(p2)
	print("-----------------")
except ValueError as ve:
	print (ve)


try:
	p1, p2 = Mesh2d._split_index_buffer(ib, 5, 100)
	print(p1)
	print(p2)
	print("-----------------")
except ValueError as ve:
	print (ve)