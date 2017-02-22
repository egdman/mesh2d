import os
import sys

here = os.path.abspath(os.path.dirname(__file__))

module_root = os.path.join(here, "..")
sys.path.append(module_root)

from mesh2d import Vector2 as v

print (v.vertex_on_ray(v(54.800000001, -36.2), v(54.8000000, -124.8), v(54.8000000, -313.0)))
