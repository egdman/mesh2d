from itertools import chain, cycle, repeat
from functools import partial
from collections import defaultdict
from copy import deepcopy

try:
    from itertools import izip as zip
except ImportError:
    pass

from .polygon import Polygon2d
from .vector2 import Geom2
from .mesh2d import get_area_calculator, signed_area

class Union       : pass
class Subtraction : pass

class Outline     : pass
class Hole        : pass

class Inside      : pass
class Outside     : pass


def cut_loop_into_pieces(loop, cuts):
    firstPiece = []
    piece = []
    wrap = True
    for elem in loop:
        if elem in cuts:
            if wrap:
                firstPiece = piece
                wrap = False
            else:
                yield piece
            piece = [elem]
        else:
            piece.append(elem)
    piece.extend(firstPiece)
    yield piece


def split_poly_boundaries(this_poly, intersect_ids, other_poly, backwards):
    def _cut_loop(loop):
        pieces = list(cut_loop_into_pieces(loop, intersect_ids))
        ref_idx, ref_piece = next((n, piece) for n, piece in enumerate(pieces) if len(piece) > 1)
        ref_vertex = this_poly.vertices[ref_piece[1]]
        ref_inside = other_poly.point_inside(ref_vertex)
        ref_even = ref_idx & 1 == 0
        if ref_inside == ref_even:
            in_out_flags = cycle((Inside, Outside))
        else:
            in_out_flags = cycle((Outside, Inside))
        return zip(pieces, in_out_flags)

    outline = this_poly.graph.loop_iterator(this_poly.graph.loops[0])
    holes = (this_poly.graph.loop_iterator(hole) for hole in this_poly.graph.loops[1:])

    if backwards:
        outline = reversed(list(outline))
        holes = (reversed(list(hole)) for hole in holes)

    for piece, location in _cut_loop(outline):
        yield piece, location, Outline

    for hole in holes:
        for piece, location in _cut_loop(hole):
            yield piece, location, Hole



def _bool_impl(A, B, op, canvas=None):
    # Copy the original polygons because they will be changed by this algorithm.
    A = deepcopy(A)
    B = deepcopy(B)

    A_contacts, B_contacts = _add_intersections_to_polys(A, B)

    A_pieces = list(split_poly_boundaries(A, A_contacts, B, False))
    B_pieces = list(split_poly_boundaries(B, B_contacts, A, op == Subtraction))

    def next_idx(idx, poly):
        return poly.graph.next[idx]

    def prev_idx(idx, poly):
        return poly.graph.prev[idx]

    next_in_A = partial(next_idx, poly = A)
    next_in_B = partial((prev_idx if op == Subtraction else next_idx), poly = B)

    closed_A_loops = list((piece, loc, kind) for piece, loc, kind in A_pieces if next_in_A(piece[-1]) == piece[0])
    closed_B_loops = list((piece, loc, kind) for piece, loc, kind in B_pieces if next_in_B(piece[-1]) == piece[0])


    def _append_loop_union(loop, A_piece, A_loc, B_piece, B_loc):
        A_in_B_out = A_loc == Inside and B_loc == Outside

        #### FOR DEBUG #### ####
        B_in_A_out = B_loc == Inside and A_loc == Outside
        if not (A_in_B_out or B_in_A_out):
            raise RuntimeError("Boolean operation failure")
        #### #### #### #### ####

        if A_in_B_out:
            loop.extend((B.vertices[idx] for idx in B_piece))
            return B_contacts.index(next_in_B(B_piece[-1]))
        else:
            loop.extend((A.vertices[idx] for idx in A_piece))
            return A_contacts.index(next_in_A(A_piece[-1]))


    def _append_loop_subtraction(loop, A_piece, A_loc, B_piece, B_loc):
        A_out_B_out = A_loc == Outside and B_loc == Outside

        #### FOR DEBUG #### ####
        A_in_B_in = B_loc == Inside and A_loc == Inside
        if not (A_in_B_in or A_out_B_out):
            raise RuntimeError("Boolean operation failure")
        #### #### #### #### ####

        if A_out_B_out:
            loop.extend((A.vertices[idx] for idx in A_piece))
            return A_contacts.index(next_in_A(A_piece[-1]))
        else:
            loop.extend((B.vertices[idx] for idx in B_piece))
            return B_contacts.index(next_in_B(B_piece[-1]))


    loops = []
    consumed = [False] * len(A_contacts)
    append_loop = _append_loop_union if op == Union else _append_loop_subtraction

    for idx, _ in enumerate(A_contacts):
        if consumed[idx]: continue

        loop = []
        while not consumed[idx]:
            consumed[idx] = True

            A_piece, A_loc = next((p, loc) for p, loc, _ in A_pieces if p[0] == A_contacts[idx])
            B_piece, B_loc = next((p, loc) for p, loc, _ in B_pieces if p[0] == B_contacts[idx])
            idx = append_loop(loop, A_piece, A_loc, B_piece, B_loc)

        loop_kind = Outline if Geom2.poly_signed_area(loop) > 0 else Hole
        loops.append( (loop, loop_kind) )



    skip_B_loop = Inside if op == Union else Outside
    map_B_kinds = (lambda kind: kind) if op == Union else (lambda kind: Outline if kind == Hole else Hole)

    for A_loop, A_loc, A_kind in closed_A_loops:
        if A_loc == Inside:
            continue
        loops.append( (list(A.vertices[idx] for idx in A_loop), A_kind) )

    for B_loop, B_loc, B_kind in closed_B_loops:
        if B_loc == skip_B_loop:
            continue
        loops.append( (list(B.vertices[idx] for idx in B_loop), map_B_kinds(B_kind)) )


    new_polys = []
    new_holes = []

    # Create new polygon for each outline loop
    for verts, kind in loops:
        if kind == Outline:
            new_polys.append(Polygon2d(verts))
        else:
            new_holes.append(verts)

    # Add new holes to new polygons
    for verts in new_holes:
        which_poly = next((poly for poly in new_polys if poly.point_inside(verts[0])), None)
        if which_poly is not None: which_poly.add_hole(verts)

    return new_polys



class ComparableSegment:
    def __init__(self, a, b, payload):
        self._a = a
        self._b = b
        self.calc_area = get_area_calculator(self._a, self._b)
        self.payload = payload

    def __lt__(self, other):
        area_a = other.calc_area(self._a)
        area_b = other.calc_area(self._b)

        if area_a < 0:
            return area_b < 0 or self.calc_area(other._a) > 0
        else:
            return area_b < 0 and self.calc_area(other._a) > 0


def _has_intersection(X, Y, A, B, area_A, area_B):
    if area_B > area_A:
        if area_A == 0 or (area_A < 0 and area_B > 0):
            # intersection is at A or between A and B
            area_Y = signed_area(A, B, Y)
            return area_Y == 0 or (area_Y < 0 and signed_area(A, B, X) > 0)

    elif area_B < area_A:
        if area_B == 0 or (area_B < 0 and area_A > 0):
            # intersection is at B or between A and B
            area_X = signed_area(A, B, X)
            return area_X == 0 or (area_X < 0 and signed_area(A, B, Y) > 0)

    return False


def _intersect_segment_with_polygon(segment, polygon):
    intersections = []
    seg0, seg1 = segment
    calc_area = get_area_calculator(seg0, seg1)

    for loop in polygon.graph.loops:
        vertex_ids = polygon.graph.loop_iterator(loop)
        vertex_A = next(vertex_ids)
        A = polygon.vertices[vertex_A]
        area_A = calc_area(A)

        vertex_first = vertex_A
        area_first = area_A

        for vertex_B in vertex_ids:
            B = polygon.vertices[vertex_B]
            area_B = calc_area(B)

            if _has_intersection(seg0, seg1, A, B, area_A, area_B):
                if area_B > area_A:
                    intersections.append(ComparableSegment(A, B, vertex_A))
                else:
                    intersections.append(ComparableSegment(B, A, vertex_A))
            vertex_A = vertex_B
            A = B
            area_A = area_B

        vertex_B = vertex_first
        B = polygon.vertices[vertex_B]
        area_B = area_first
        if _has_intersection(seg0, seg1, A, B, area_A, area_B):
            if area_B > area_A:
                intersections.append(ComparableSegment(A, B, vertex_A))
            else:
                intersections.append(ComparableSegment(B, A, vertex_A))

    intersections.sort()
    return tuple(isect.payload for isect in intersections)


def _add_intersections_to_polys(A, B):

    def get_segment_crds(seg, verts):
        return tuple(verts[idx] for idx in seg)

    def seglike_to_raylike(seglike):
        return (seglike[0], seglike[1] - seglike[0])

    def get_segments(poly):
        def loop_segments(loop):
            return ((idx, poly.graph.next[idx]) for idx in poly.graph.loop_iterator(loop))
        return chain(*(loop_segments(loop) for loop in poly.graph.loops))

    # List of vertices on edge intersections.
    intersection_param_pairs = []

    # These are maps that for each edge maintain a list of vertices to add to that edge.
    # These maps actually store only indices of those vertices. The vertices themselves are
    # in the 'intersection_param_pairs' list as pairs of parameters from0 to 1.
    A_new_vert_lists = defaultdict(list)
    B_new_vert_lists = defaultdict(list)

    # Find all intersections of A and B borders.
    for A_edge in get_segments(A):
        intersections = _intersect_segment_with_polygon(
            get_segment_crds(A_edge, A.vertices),
            B,
        )
        for intersection_idx in intersections:
            B_edge = (intersection_idx, B.graph.next[intersection_idx])
            seg_A = seglike_to_raylike(get_segment_crds(A_edge, A.vertices))
            seg_B = seglike_to_raylike(get_segment_crds(B_edge, B.vertices))
            a, b, _ = Geom2.lines_intersect(seg_A, seg_B)
            pair_index = len(intersection_param_pairs)
            intersection_param_pairs.append((a, b))

            A_new_vert_lists[A_edge].append(pair_index)
            B_new_vert_lists[B_edge].append(pair_index)

    A_inserted_ids = {}
    for (A_edge, pair_ids) in A_new_vert_lists.items():
        params = list(intersection_param_pairs[idx][0] for idx in pair_ids)
        inserted_ids = A.add_vertices_to_border(A_edge, params)
        A_inserted_ids.update(dict(zip(pair_ids, inserted_ids)))


    B_inserted_ids = {}
    for (B_edge, pair_ids) in B_new_vert_lists.items():
        params = list(intersection_param_pairs[idx][1] for idx in pair_ids)
        inserted_ids = B.add_vertices_to_border(B_edge, params)
        B_inserted_ids.update(dict(zip(pair_ids, inserted_ids)))


    ids_in_A, ids_in_B = [], []
    for pair_idx in range(len(intersection_param_pairs)):
        ids_in_A.append(A_inserted_ids[pair_idx])
        ids_in_B.append(B_inserted_ids[pair_idx])

    return ids_in_A, ids_in_B


def bool_subtract(A, B, canvas=None):
    '''
    Performs boolean subtraction of B from A. Returns a list of new Polygon2d instances
    or an empty list.
    '''
    return _bool_impl(A, B, Subtraction, canvas)


def bool_add(A, B, canvas=None):
    '''
    Performs boolean subtraction of B from A. Returns a list of new Polygon2d instances
    or an empty list.
    '''
    return _bool_impl(A, B, Union, canvas)
