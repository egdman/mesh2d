from itertools import chain, cycle, repeat
from functools import partial
from collections import defaultdict
from copy import deepcopy

try:
    from itertools import izip as zip
except ImportError:
    pass

from .mesh2d import Polygon2d
from .vector2 import Geom2
from .utils import debug_draw_bool

class Union       : pass
class Subtraction : pass
class Intersection: pass

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



def _bool_do(A, B, op, canvas=None):
    # Copy the original polygons because they will be changed by this algorithm.
    A = deepcopy(A)
    B = deepcopy(B)

    A_intersection_ids, B_intersection_ids, idx_map = _add_intersections_to_polys(A, B)

    A_pieces = list(split_poly_boundaries(A, A_intersection_ids, B, False))
    B_pieces = list(split_poly_boundaries(B, B_intersection_ids, A, op == Subtraction))

    def next_idx(idx, poly):
        return poly.graph.next[idx]

    def prev_idx(idx, poly):
        return poly.graph.prev[idx]

    next_in_A = partial(next_idx, poly = A)
    next_in_B = partial((prev_idx if op == Subtraction else next_idx), poly = B)

    closed_A_loops = list((piece, loc, kind) for piece, loc, kind in A_pieces if next_in_A(piece[-1]) == piece[0])
    closed_B_loops = list((piece, loc, kind) for piece, loc, kind in B_pieces if next_in_B(piece[-1]) == piece[0])

    if op == Union:
        # for each idx in intersection ids we have exactly 1 A-piece and 1 B-piece that start from it
        # we need to choose 0, 1, or 2 pieces for each intersection

        if len(idx_map):
            B_contacts, A_contacts = zip(*idx_map.items())
        else:
            B_contacts, A_contacts = [], []

        consumed = [False] * len(A_contacts)

        loops = []
        for idx, _ in enumerate(A_contacts):
            if consumed[idx]: continue

            loop = []
            while not consumed[idx]:
                consumed[idx] = True

                A_contact = A_contacts[idx]
                B_contact = B_contacts[idx]
                A_piece, A_loc, A_kind = next((p, loc, kind) for p, loc, kind in A_pieces if p[0] == A_contact)
                B_piece, B_loc, B_kind = next((p, loc, kind) for p, loc, kind in B_pieces if p[0] == B_contact)

                A_in_B_out = A_loc == Inside and B_loc == Outside

                #### FOR DEBUG #### ####
                # pieces cannot be both inside or both outside
                B_in_A_out = B_loc == Inside and A_loc == Outside
                if A_in_B_out == B_in_A_out:
                    raise RuntimeError("Boolean operation failure")
                #### #### #### #### ####

                if A_in_B_out:
                    loop.extend((B.vertices[idx] for idx in B_piece))
                    idx = B_contacts.index(next_in_B(B_piece[-1]))
                else:
                    loop.extend((A.vertices[idx] for idx in A_piece))
                    idx = A_contacts.index(next_in_A(A_piece[-1]))

            loop_kind = Outline if Geom2.poly_signed_area(loop) > 0 else Hole
            loops.append( (loop, loop_kind) )


        for A_loop, A_loc, A_kind in closed_A_loops:
            if A_loc == Inside:
                continue
            loops.append( (list(A.vertices[idx] for idx in A_loop), A_kind) )

        for B_loop, B_loc, B_kind in closed_B_loops:
            if B_loc == Inside:
                continue
            loops.append( (list(B.vertices[idx] for idx in B_loop), B_kind) )


    elif op == Subtraction:
        if len(idx_map):
            B_contacts, A_contacts = zip(*idx_map.items())
        else:
            B_contacts, A_contacts = [], []

        consumed = [False] * len(A_contacts)

        loops = []
        for idx, _ in enumerate(A_contacts):
            if consumed[idx]: continue

            loop = []
            while not consumed[idx]:
                consumed[idx] = True

                A_contact = A_contacts[idx]
                B_contact = B_contacts[idx]
                A_piece, A_loc, A_kind = next((p, loc, kind) for p, loc, kind in A_pieces if p[0] == A_contact)
                B_piece, B_loc, B_kind = next((p, loc, kind) for p, loc, kind in B_pieces if p[0] == B_contact)

                A_out_B_out = A_loc == Outside and B_loc == Outside

                #### FOR DEBUG #### ####
                # pieces cannot be both inside or both outside
                A_in_B_in = B_loc == Inside and A_loc == Inside
                if A_out_B_out == A_in_B_in:
                    raise RuntimeError("Boolean operation failure")
                #### #### #### #### ####

                if A_out_B_out:
                    loop.extend((A.vertices[idx] for idx in A_piece))
                    idx = A_contacts.index(next_in_A(A_piece[-1]))
                else:
                    loop.extend((B.vertices[idx] for idx in B_piece))
                    idx = B_contacts.index(next_in_B(B_piece[-1]))

            loop_kind = Outline if Geom2.poly_signed_area(loop) > 0 else Hole
            loops.append( (loop, loop_kind) )

        for A_loop, A_loc, A_kind in closed_A_loops:
            if A_loc == Inside:
                continue
            loops.append( (list(A.vertices[idx] for idx in A_loop), A_kind) )

        for B_loop, B_loc, B_kind in closed_B_loops:
            if B_loc == Outside:
                continue
            loops.append( (list(B.vertices[idx] for idx in B_loop), Outline if B_kind == Hole else Hole) )


    new_polys = []
    new_holes = []

    # Create new polygon for each outline loop
    for verts, kind in loops:
        if kind == Outline:
            new_polys.append(Polygon2d(verts))
        else:
            new_holes.append(verts)

    # Try to add new holes to new polygons
    for verts in new_holes:
        which_poly = next((poly for poly in new_polys if poly.point_inside(verts[0])), None)
        if which_poly is not None: which_poly.add_hole(verts)

    return new_polys



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
        for B_edge in get_segments(B):
            seg_A = seglike_to_raylike(get_segment_crds(A_edge, A.vertices))
            seg_B = seglike_to_raylike(get_segment_crds(B_edge, B.vertices))

            a, b, _ = Geom2.lines_intersect(seg_A, seg_B)
            if 0 <= a and a < 1 and 0 <= b and b < 1:
                pair_index = len(intersection_param_pairs)
                intersection_param_pairs.append((a, b))

                A_new_vert_lists[A_edge].append(pair_index)
                B_new_vert_lists[B_edge].append(pair_index)

    # print("{} intersection points".format(len(intersection_param_pairs)))
    # This list will hold indices into A's vertex buffer of all intersection verts that will be added to A.
    A_new_ids = []

    # Same for B.
    B_new_ids = []

    # This is a mapping of {index-in-B : index-in-A} for all intersection verts.
    idx_map = {}

    A_inserted_ids = {}
    for (A_edge, pair_ids) in A_new_vert_lists.iteritems():
        params = list(intersection_param_pairs[idx][0] for idx in pair_ids)
        inserted_ids = A.add_vertices_to_border(A_edge, params)
        A_inserted_ids.update(dict(zip(pair_ids, inserted_ids)))
        A_new_ids.extend(inserted_ids)


    B_inserted_ids = {}
    for (B_edge, pair_ids) in B_new_vert_lists.iteritems():
        params = list(intersection_param_pairs[idx][1] for idx in pair_ids)
        inserted_ids = B.add_vertices_to_border(B_edge, params)
        B_inserted_ids.update(dict(zip(pair_ids, inserted_ids)))
        B_new_ids.extend(inserted_ids)


    for pair_idx in range(len(intersection_param_pairs)):
        inserted_into_A = A_inserted_ids[pair_idx]
        inserted_into_B = B_inserted_ids[pair_idx]
        idx_map[inserted_into_B] = inserted_into_A

    return A_new_ids, B_new_ids, idx_map


def bool_subtract(A, B, canvas=None):
    '''
    Performs boolean subtraction of B from A. Returns a list of new Polygon2d instances
    or an empty list.
    '''
    return _bool_do(A, B, Subtraction, canvas)


def bool_add(A, B, canvas=None):
    '''
    Performs boolean subtraction of B from A. Returns a list of new Polygon2d instances
    or an empty list.
    '''
    return _bool_do(A, B, Union, canvas)