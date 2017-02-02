import sys
import os

# import pudb; pu.db

here = os.path.abspath(os.path.dirname(__file__))

module_root = os.path.join(here, "..")
sys.path.append(module_root)

from mesh2d import Vector2

# check equality and inequality operations

print("\nshould be (0,4)")
rel1 = Vector2.lines_intersect(
	Vector2(0., 0.), Vector2(0., 5.),
	Vector2(2., 4.), Vector2(-2., 4.),)

print(rel1.intersection)
print(rel1.identical)



print("\nshould be (0,5)")
rel1 = Vector2.lines_intersect(
	Vector2(0., 0.), Vector2(0., 5.),
	Vector2(2., 5.), Vector2(-2., 5.),)

print(rel1.intersection)
print(rel1.identical)


print("\nshould be (0,7)")
rel1 = Vector2.lines_intersect(
	Vector2(0., 0.), Vector2(0., 5.),
	Vector2(2., 7.), Vector2(-2., 7.),)

print(rel1.intersection)
print(rel1.identical)


print("\nshould be (3,3)")
rel1 = Vector2.lines_intersect(
	Vector2(0., 0.), Vector2(5., 5.),
	Vector2(0., 3.), Vector2(-1., 3.),)

print(rel1.intersection)
print(rel1.identical)


print("\nshould be (1.5, 1.5)")
rel1 = Vector2.lines_intersect(
	Vector2(0., 0.), Vector2(5., 5.),
	Vector2(0., 3.), Vector2(3., 0.),)

print(rel1.intersection)
print(rel1.identical)



print("\nshould be parallel")
rel1 = Vector2.lines_intersect(
	Vector2(0., 0.), Vector2(2., 5.),
	Vector2(8., 3.), Vector2(8.+2., 3.+5.),)

print(rel1.intersection)
print(rel1.identical)


print("\nshould be identical")
rel1 = Vector2.lines_intersect(
	Vector2(0., 0.), Vector2(2., 5.),
	Vector2(-4., -10.), Vector2(-1., -2.5),)

print(rel1.intersection)
print(rel1.identical)



print("\nshould be identical")
rel1 = Vector2.lines_intersect(
	Vector2(0., 0.), Vector2(0., 5.),
	Vector2(0., -10.), Vector2(0., -2.5),)

print(rel1.intersection)
print(rel1.identical)


print("\nline undefined (same point twice)")
rel1 = Vector2.lines_intersect(
	Vector2(0., 0.), Vector2(2, 4.),
	Vector2(0., -10.), Vector2(0., -10.),)

print(rel1.intersection)
print(rel1.identical)