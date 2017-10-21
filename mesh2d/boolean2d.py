from itertools import izip, chain
from collections import deque

from .mesh2d import Polygon2d
from .vector2 import vec
from .utils import debug_draw_bool

class Union       : pass
class Subtraction : pass
class Intersection: pass

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
    1) all pieces of the border oth this_poly outside other_poly, and
    2) all pieces of the border oth this_poly inside other_poly.
    '''
    outside, inside = _bool_cut_loop(this_poly.outline, this_poly, intersect_ids, other_poly)

    for hole in this_poly.holes:
        h_o, h_i = _bool_cut_loop(hole, this_poly, intersect_ids, other_poly)
        outside.extend(h_o)
        inside.extend(h_i)
    return outside, inside        



def _add_intersections_to_polys(A, B):
    # List of vertices on edge intersections.
    intersection_verts = []

    # These are maps that for each edge maintain a list of vertices to add to that edge.
    # These maps actually store only indices of those vertices. The vertices themselves are
    # in the 'intersection_verts' list.
    A_new_vert_lists = {}
    B_new_vert_lists = {}

    A_edges = Polygon2d.get_segments(chain([A.outline], A.holes))
    B_edges = Polygon2d.get_segments(chain([B.outline], B.holes))

    # Find all intersections of A and B borders.
    for A_edge in A_edges:
        for B_edge in B_edges:
            seg_x = vec.where_segments_cross_inclusive(
                A.vertices[A_edge[0]],
                A.vertices[A_edge[1]],
                B.vertices[B_edge[0]],
                B.vertices[B_edge[1]])
            if seg_x is not None:
                if A_edge not in A_new_vert_lists:
                    A_new_vert_lists[A_edge] = []

                if B_edge not in B_new_vert_lists:
                    B_new_vert_lists[B_edge] = []

                inters_index = len(intersection_verts)
                intersection_verts.append(seg_x)

                A_new_vert_lists[A_edge].append(inters_index)
                B_new_vert_lists[B_edge].append(inters_index)


    # This list will hold indices into A's vertex buffer of all intersection verts that will be added to A.
    A_new_ids = []

    # Same for B.
    B_new_ids = []

    # This is a mapping of {index-in-B : index-in-A} for all intersection verts.
    idx_map = {}


    A_inserted_ids = {}
    for (A_edge, x_ids) in A_new_vert_lists.iteritems():
        x_verts = list(intersection_verts[idx] for idx in x_ids)
        inserted_ids = A.add_vertices_to_border(x_verts, A_edge)

        # A_inserted_ids.update({x_idx: ins_idx for (x_idx, ins_idx) in izip(x_ids, inserted_ids)})
        A_inserted_ids.update(dict(izip(x_ids, inserted_ids)))
        A_new_ids.extend(inserted_ids)


    B_inserted_ids = {}
    for (B_edge, x_ids) in B_new_vert_lists.iteritems():
        x_verts = list(intersection_verts[idx] for idx in x_ids)
        inserted_ids = B.add_vertices_to_border(x_verts, B_edge)

        # B_inserted_ids.update({x_idx: ins_idx for (x_idx, ins_idx) in izip(x_ids, inserted_ids)})
        B_inserted_ids.update(dict(izip(x_ids, inserted_ids)))
        B_new_ids.extend(inserted_ids)


    for x_idx in range(len(intersection_verts)):
        inserted_into_A = A_inserted_ids[x_idx]
        inserted_into_B = B_inserted_ids[x_idx]
        idx_map[inserted_into_B] = inserted_into_A

    return A_new_ids, B_new_ids, idx_map



def _concat_border_pieces(A, B, A_open, B_open, idx_map, tail_to_tail):
    closed_A_pieces = []

    attach_point = -1 if tail_to_tail else 0

    def nonzero_pieces(pieces):
        return ((i, piece) for i, piece in enumerate(pieces) if len(piece) > 0)

    while True:

        print("A:")
        print (A_open)
        print("B:")
        print (list(list(idx_map.get(idx, str(idx)) for idx in loop) for loop in B_open))
        print("")

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

    # Copy the original polygons because they will be changed by this algorithm.
    A = A.copy()
    B = B.copy()

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

    tail_to_tail = True if op == Subtraction else False
    A_closed.extend(_concat_border_pieces(A, B, A_open, B_open, idx_map, tail_to_tail))


    new_polys = []
    new_holes = []

    # A_closed contains outlines and holes.
    # Create new polygon for each loop that is CCW
    # CW loops represent holes
    for loop in A_closed:
        verts = list(A.vertices[idx] for idx in loop[:-1])
        # if CCW
        if vec.poly_signed_area(verts) > 0:
            new_polys.append(Polygon2d(verts, range(len(verts))))
        else:
            new_holes.append(verts)

    # B_closed contains outlines and holes.
    # Create new polygon for each loop that is CCW (CW if subtracting)
    # CW (CCW if subtracting) loops represent holes
    flip = -1 if op == Subtraction else 1
    for loop in B_closed:
        verts = list(B.vertices[idx] for idx in loop[:-1])
        # if CW
        if flip * vec.poly_signed_area(verts) > 0:
            new_polys.append(Polygon2d(verts, range(len(verts))))
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