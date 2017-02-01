import yaml
import sys
import os

here = os.path.abspath(os.path.dirname(__file__))

module_root = os.path.join(here, "..")
sys.path.append(module_root)

from mesh2d import Polygon2d, Vector2


with open(os.path.join(module_root, "debug_dump.yaml"), 'r') as debf:
	polygon = yaml.load(debf)



portals = polygon.break_into_convex(10.0)

print("num portals = {}".format(len(portals)))
print("num pieces  = {}".format(len(polygon.pieces)))