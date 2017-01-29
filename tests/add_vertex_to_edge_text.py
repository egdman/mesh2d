import sys
import os

here = os.path.abspath(os.path.dirname(__file__))

module_root = os.path.join(here, "..")
sys.path.append(module_root)

from mesh2d import Polygon2d, Vector2


def poly_repr(poly):
	# crds = list((poly.vertices[ind].x, poly.vertices[ind].y) for ind in poly.indices)

	outl = poly.outline_coordinates()

	crds = list((outl[2*pos], outl[2*pos+1]) for pos in range(len(outl) / 2))
	st = ''
	st += 'indices:  ' + repr(poly.indices)
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

poly = Polygon2d(vertices[:], range(len(vertices)))

print(poly_repr(poly))
print("////////////////////////////////////////////////////////////////////\n\n")




poly = Polygon2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_edge(i1, (0,1))
print(poly_repr(poly))
print("----------------------------------")

poly = Polygon2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_edge(i1, (1,0))
print(poly_repr(poly))
print("==================================\n")




poly = Polygon2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_edge(i2, (1,2))
print(poly_repr(poly))
print("----------------------------------")

poly = Polygon2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_edge(i2, (2,1))
print(poly_repr(poly))
print("==================================\n")




poly = Polygon2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_edge(i3, (2,3))
print(poly_repr(poly))
print("----------------------------------")

poly = Polygon2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_edge(i3, (3,2))
print(poly_repr(poly))
print("==================================\n")




poly = Polygon2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_edge(i4, (3,0))
print(poly_repr(poly))
print("----------------------------------")

poly = Polygon2d(vertices[:], range(len(vertices)))
poly.add_vertex_to_edge(i4, (0,3))
print(poly_repr(poly))
print("==================================\n")




# test area of polygon
a = Vector2(54., 154.)
b = Vector2(55., 154.)
c = Vector2(55., 155.)
d = Vector2(54., 155.)
vertices = [a, b, c, d]

poly = Polygon2d(vertices[:], range(len(vertices)))
print(poly_repr(poly))
print ("Area = {}".format(Polygon2d.signed_area(poly.vertices, poly.indices)))


