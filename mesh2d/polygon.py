from itertools import chain
from operator import itemgetter

from .vector2 import Geom2

class Loops(object):
    def __init__(self):
        self.loops = []
        self.next = []
        self.prev = []


    def add_loop(self, how_many_nodes):
        loop_start = len(self.next)
        ids = range(loop_start, loop_start + how_many_nodes)

        self.loops.append(loop_start)
        self.next.extend(ids[1:] + ids[:1])
        self.prev.extend(ids[-1:] + ids[:-1])
        return loop_start


    def loop_iterator(self, loop_start):
        yield loop_start
        idx = self.next[loop_start]
        while idx != loop_start:
            yield idx
            idx = self.next[idx]


    def all_nodes_iterator(self):
        return chain(*(self.loop_iterator(loop) for loop in self.loops))


    def insert_node(self, edge_to_split):
        new_idx = len(self.next)
        e0, e1 = edge_to_split

        self.next[e0] = self.prev[e1] = new_idx
        self.next.append(e1)
        self.prev.append(e0)
        return new_idx


class Polygon2d(object):
    def __init__(self, vertices):
        # ensure CCW order - outline must be CCW
        if Geom2.poly_signed_area(vertices) > 0:
            self.vertices = list(vertices)
        else:
            self.vertices = list(vertices[::-1])

        self.graph = Loops()
        self.graph.add_loop(len(self.vertices))


    def add_hole(self, vertices):
        # ensure CW order - holes must be CW
        if Geom2.poly_signed_area(vertices) < 0:
            self.vertices.extend(vertices)
        else:
            self.vertices.extend(vertices[::-1])

        self.graph.add_loop(len(vertices))


    def insert_vertex(self, vertex, edge_to_split):
        new_idx = self.graph.insert_node(edge_to_split)
        assert new_idx == len(self.vertices)
        self.vertices.append(vertex)
        return new_idx


    def add_vertices_to_border(self, edge, vertex_params):
        '''
        Add given list of vertices to the given edge of the polygon border.
        The exact loop that contains the edge is determined automatically.
        This function returns a list of new indices for the list of vertices in the same order.
        '''
        new_ids = [None] * len(vertex_params)
        ascend_params = sorted(enumerate(vertex_params), key = itemgetter(1))

        e0, e1 = edge
        ray = (self.vertices[e0], self.vertices[e1] - self.vertices[e0])
        idx = e0
        for (position_before_sorted, param) in ascend_params:
            idx = self.insert_vertex(ray[0] + param * ray[1], (idx, e1))
            new_ids[position_before_sorted] = idx

        return new_ids


    def point_inside_loop(self, point, loop_start):
        # transform vertices so that query point is at the origin, append start vertex at end to wrap
        verts = (self.vertices[idx] - point for idx in self.graph.loop_iterator(loop_start))
        return Geom2.is_origin_inside_polyline(verts)


    def point_inside(self, point):
        # first check if inside outline
        if not self.point_inside_loop(point, self.graph.loops[0]):
            return False

        # now check if inside a hole
        for hole in self.graph.loops[1:]:
            if self.point_inside_loop(point, hole):
                return False

        return True
