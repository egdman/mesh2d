import sys
import os

# import pudb; pu.db

here = os.path.abspath(os.path.dirname(__file__))

module_root = os.path.join(here, "..")
sys.path.append(module_root)

from mesh2d import Polygon2d, Vector2

# check equality and inequality operations

a = Vector2(0.5, 0.5)

b = Vector2(0.5, 0.5)

assert((a == b) == True)
assert((a != b) == False)


b = Vector2(0.5, 0.45)


assert((a == b) == False)
assert((a != b) == True)


b = Vector2(0.45, 0.5)

assert((a == b) == False)
assert((a != b) == True)

b = Vector2(0.45, 0.45)

assert((a == b) == False)
assert((a != b) == True)


b = a*5.
assert(b.x == 2.5 and b.y == 2.5)


b = 5.*a
assert(b.x == 2.5 and b.y == 2.5)


b = Vector2(-0.5, -0.5)
b = a + b
assert(b.x == 0 and b.y == 0)
