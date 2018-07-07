from itertools import chain, tee, cycle, repeat
from collections import deque, defaultdict
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


def _cut_to_pieces(array, cut_items):
    pieces = [[]]
    for el in array:
        pieces[-1].append(el)
        if el in cut_items:
            pieces.append([el])

    if len(pieces) == 1:
        pieces[0].append(pieces[0][0])
    else:
        pieces[0] = pieces[-1] + pieces[0]
        del pieces[-1]

    return pieces



def _bool_cut_loop(loop, this_poly, intersect_ids, other_poly):
        loop_pieces = _cut_to_pieces(loop, intersect_ids)

        # find first piece that contains a non-intersection vertex
        first_piece, start_num = next(((piece, num) for num, piece \
            in enumerate(loop_pieces) if len(piece) > 2))

        # put this piece at the start of list of pieces
        loop_pieces = deque(loop_pieces)
        loop_pieces.rotate(-start_num)
        loop_pieces = list(loop_pieces)

        inside = other_poly.point_inside(this_poly.vertices[first_piece[1]])
        if inside:
            pieces_inside = loop_pieces[0::2]
            pieces_outside = loop_pieces[1::2]
        else:
            pieces_inside = loop_pieces[1::2]
            pieces_outside = loop_pieces[0::2]

        return pieces_outside, pieces_inside



def _get_pieces_outside_inside(this_poly, intersect_ids, other_poly):
    '''
    Returns 2 lists:
    1) all pieces of the border of this_poly outside other_poly, and
    2) all pieces of the border of this_poly inside other_poly.
    '''
    this_poly_outline = list(this_poly.graph.loop_iterator(this_poly.graph.loops[0]))
    outside, inside = _bool_cut_loop(this_poly_outline, this_poly, intersect_ids, other_poly)

    this_poly_holes = list(list(this_poly.graph.loop_iterator(h)) for h in this_poly.graph.loops[1:])
    for hole in this_poly_holes:
        h_o, h_i = _bool_cut_loop(hole, this_poly, intersect_ids, other_poly)
        outside.extend(h_o)
        inside.extend(h_i)
    return outside, inside


#### DO NOT APPEND START AT END ########
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
#######################################




# def find_polygon_pieces(this_poly, intersect_ids):
#     outline = this_poly.graph.loop_iterator(this_poly.graph.loops[0])
#     holes = (this_poly.graph.loop_iterator(hole) for hole in this_poly.graph.loops[1:])

#     for outline_piece in cut_loop_into_pieces(outline, intersect_ids):
#         yield (outline_piece, Outline)

#     for hole in holes:
#         for hole_piece in cut_loop_into_pieces(hole, intersect_ids):
#             yield (hole_piece, Hole)


# def _bool_cut_loop(loop, this_poly, intersect_ids, other_poly):
#     loop_pieces = _cut_to_pieces(loop, intersect_ids)

#     # find first piece that contains a non-intersection vertex
#     first_piece, start_num = next(((piece, num) for num, piece \
#         in enumerate(loop_pieces) if len(piece) > 2))

#     # put this piece at the start of list of pieces
#     loop_pieces = deque(loop_pieces)
#     loop_pieces.rotate(-start_num)
#     loop_pieces = list(loop_pieces)

#     inside = other_poly.point_inside(this_poly.vertices[first_piece[1]])
#     if inside:
#         pieces_inside = loop_pieces[0::2]
#         pieces_outside = loop_pieces[1::2]
#     else:
#         pieces_inside = loop_pieces[1::2]
#         pieces_outside = loop_pieces[0::2]

#     return pieces_outside, pieces_inside


def split_poly_boundaries(this_poly, intersect_ids, other_poly):
    '''
    returns 2 lists:
    pieces of this_poly inside other_poly, and
    pieces of this_poly outside other_poly
    '''
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

    for piece, location in _cut_loop(outline):
        yield piece, location, Outline

    for hole in holes:
        for piece, location in _cut_loop(hole):
            yield piece, location, Hole









def _bool_do(A, B, op, canvas=None):
    '''
    op values:
    -1 (subtract B from A)
     0 (intersect A and B)
     1 (add B to A)
    '''

    # # Copy the original polygons because they will be changed by this algorithm.
    A = deepcopy(A)
    B = deepcopy(B)

    A_intersection_ids, B_intersection_ids, idx_map = _add_intersections_to_polys(A, B)

    A_pieces = list(split_poly_boundaries(A, A_intersection_ids, B))
    B_pieces = list(split_poly_boundaries(B, B_intersection_ids, A))

    # for each idx in intersection ids we have exactly 1 A-piece and 1 B-piece that start from it
    # we need to choose one of 2 pieces

    chosen_pieces = []
    for B_idx, A_idx in idx_map.items():
        A_piece = next(p for p in A_pieces if p[0] == A_idx)
        B_piece = next(p for p in B_pieces if p[0] == B_idx)

        A_piece, loc_rel_to_B, _ = A_piece
        B_piece, loc_rel_to_A, _ = B_piece

        if op == Subtraction:
            if loc_rel_to_B == Outside:
                chosen_pieces.append(A_piece)
            elif loc_rel_to_A == Inside:
                chosen_pieces.append(B_piece)

        if op == Union:
            if loc_rel_to_B == Outside:
                chosen_pieces.append(A_piece)
            elif loc_rel_to_A == Outside:
                chosen_pieces.append(B_piece)

        else:
            raise RuntimeError("Unknown boolean operation")


    if op == Subtraction:
        # find pieces of A that are outside B
        A_border_pieces, _ = _get_pieces_outside_inside(A, A_intersection_ids, B)

        # find pieces of B that are inside A
        _, B_border_pieces = _get_pieces_outside_inside(B, B_intersection_ids, A)

    # elif op == Intersection:
    #     # find pieces of A that are inside B
    #     _, A_border_pieces = _get_pieces_outside_inside(A, A_intersection_ids, B)

    #     # find pieces of B that are inside A
    #     _, B_border_pieces = _get_pieces_outside_inside(B, B_intersection_ids, A)

    elif op == Union:
        # find pieces of A that are outside B
        A_border_pieces, _ = _get_pieces_outside_inside(A, A_intersection_ids, B)

        # find pieces of B that are outside A
        B_border_pieces, _ = _get_pieces_outside_inside(B, B_intersection_ids, A)



    # debug draw
    if canvas:
        for idx in canvas.find_all(): canvas.delete(idx)
        debug_draw_bool(A, B, A_border_pieces, B_border_pieces, idx_map, canvas)


    # divide into closed and open loops
    A_closed = list(p for p in A_border_pieces if p[0] == p[-1])
    B_closed = list(p for p in B_border_pieces if p[0] == p[-1])

    A_open = list(p for p in A_border_pieces if p[0] != p[-1])
    B_open = list(p for p in B_border_pieces if p[0] != p[-1])

    A_closed.extend(_concat_border_pieces(A, B, A_open, B_open, idx_map, op == Subtraction))


    new_polys = []
    new_holes = []

    # A_closed contains outlines and holes.
    # Create new polygon for each loop that is CCW
    # CW loops represent holes
    for loop in A_closed:
        verts = list(A.vertices[idx] for idx in loop[:-1])
        # if CCW
        if Geom2.poly_signed_area(verts) > 0:
            new_polys.append(Polygon2d(verts))
        else:
            new_holes.append(verts)

    # B_closed contains outlines and holes.
    # Create new polygon for each loop that is CCW (CW if subtracting)
    # CW (CCW if subtracting) loops represent holes
    flip = -1 if op == Subtraction else 1
    for loop in B_closed:
        verts = list(B.vertices[idx] for idx in loop[:-1])
        # if CW
        if flip * Geom2.poly_signed_area(verts) > 0:
            new_polys.append(Polygon2d(verts))
        else:
            new_holes.append(verts)

    # Try to add new holes to new polygons. Ignore holes that are outside all polys.
    for verts in new_holes:
        which_poly = next((poly for poly in new_polys if poly.point_inside(verts[0])), None)
        if which_poly is not None: which_poly.add_hole(verts)

    return new_polys














def get_segment_crds(seg, verts):
    return tuple(verts[idx] for idx in seg)


def seglike_to_raylike(seglike):
    return (seglike[0], seglike[1] - seglike[0])


def get_segments(poly):
    def loop_segments(loop):
        return ((idx, poly.graph.next[idx]) for idx in poly.graph.loop_iterator(loop))
    return chain(*(loop_segments(loop) for loop in poly.graph.loops))


def _add_intersections_to_polys(A, B):
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



def _concat_border_pieces(A, B, A_open, B_open, idx_map, tail_to_tail):
    closed_A_pieces = []

    attach_point = -1 if tail_to_tail else 0

    def nonzero_pieces(pieces):
        return ((i, piece) for i, piece in enumerate(pieces) if len(piece) > 0)

    while True:

        # print("A:")
        # print (A_open)
        # print("B:")
        # print (list(list(idx_map.get(idx, str(idx)) for idx in loop) for loop in B_open))
        # print("")

        A_pos, A_piece = next(nonzero_pieces(A_open), (None, None))
        if A_pos is None: break

        new_loop = A_piece[:]

        del A_open[A_pos]

        while True:

            # stop when outline becomes closed
            if new_loop[0] == new_loop[-1]: break

            # find B's piece to attach:
            B_pos, B_piece = next(((i, piece) for i, piece in nonzero_pieces(B_open) \
                if new_loop[-1] == idx_map[piece[attach_point]]), (None, None))
            if B_pos is None: break

            B_piece_iter = reversed(B_piece[1:-1]) if tail_to_tail else iter(B_piece[1:-1])

            for B_idx in B_piece_iter:
                B_vrt = B.vertices[B_idx]
                new_loop.append(len(A.vertices))
                A.vertices.append(B_vrt)
            new_loop.append(idx_map[B_piece[0]] if tail_to_tail else idx_map[B_piece[-1]])

            del B_open[B_pos]


            # find A's piece to attach
            A_pos, next_A_piece = next(((i, piece) for i, piece in nonzero_pieces(A_open) \
                if new_loop[-1] == piece[0]), (None, None))
            if A_pos is None: break

            new_loop.extend(next_A_piece[1:])
            del A_open[A_pos]

        if new_loop[0] != new_loop[-1]:
            raise RuntimeError("Boolean operation failed")

        closed_A_pieces.append(new_loop)

    return closed_A_pieces



def _bool_do(A, B, op, canvas=None):
    '''
    op values:
    -1 (subtract B from A)
     0 (intersect A and B)
     1 (add B to A)
    '''

    # # Copy the original polygons because they will be changed by this algorithm.
    A = deepcopy(A)
    B = deepcopy(B)

    A_intersection_ids, B_intersection_ids, idx_map = _add_intersections_to_polys(A, B)

    if op == Subtraction: # subtract
        # find pieces of A that are outside B
        A_border_pieces, _ = _get_pieces_outside_inside(A, A_intersection_ids, B)

        # find pieces of B that are inside A
        _, B_border_pieces = _get_pieces_outside_inside(B, B_intersection_ids, A)

    elif op == Intersection: # intersect
        # find pieces of A that are inside B
        _, A_border_pieces = _get_pieces_outside_inside(A, A_intersection_ids, B)

        # find pieces of B that are inside A
        _, B_border_pieces = _get_pieces_outside_inside(B, B_intersection_ids, A)

    elif op == Union: # add
        # find pieces of A that are outside B
        A_border_pieces, _ = _get_pieces_outside_inside(A, A_intersection_ids, B)

        # find pieces of B that are outside A
        B_border_pieces, _ = _get_pieces_outside_inside(B, B_intersection_ids, A)
    else:
        raise RuntimeError("Unknown boolean operation")


    # debug draw
    if canvas:
        for idx in canvas.find_all(): canvas.delete(idx)
        debug_draw_bool(A, B, A_border_pieces, B_border_pieces, idx_map, canvas)


    # divide into closed and open loops
    A_closed = list(p for p in A_border_pieces if p[0] == p[-1])
    B_closed = list(p for p in B_border_pieces if p[0] == p[-1])

    A_open = list(p for p in A_border_pieces if p[0] != p[-1])
    B_open = list(p for p in B_border_pieces if p[0] != p[-1])

    A_closed.extend(_concat_border_pieces(A, B, A_open, B_open, idx_map, op == Subtraction))


    new_polys = []
    new_holes = []

    # A_closed contains outlines and holes.
    # Create new polygon for each loop that is CCW
    # CW loops represent holes
    for loop in A_closed:
        verts = list(A.vertices[idx] for idx in loop[:-1])
        # if CCW
        if Geom2.poly_signed_area(verts) > 0:
            new_polys.append(Polygon2d(verts))
        else:
            new_holes.append(verts)

    # B_closed contains outlines and holes.
    # Create new polygon for each loop that is CCW (CW if subtracting)
    # CW (CCW if subtracting) loops represent holes
    flip = -1 if op == Subtraction else 1
    for loop in B_closed:
        verts = list(B.vertices[idx] for idx in loop[:-1])
        # if CW
        if flip * Geom2.poly_signed_area(verts) > 0:
            new_polys.append(Polygon2d(verts))
        else:
            new_holes.append(verts)

    # Try to add new holes to new polygons. Ignore holes that are outside all polys.
    for verts in new_holes:
        which_poly = next((poly for poly in new_polys if poly.point_inside(verts[0])), None)
        if which_poly is not None: which_poly.add_hole(verts)

    return new_polys



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