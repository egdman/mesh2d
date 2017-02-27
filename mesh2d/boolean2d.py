from itertools import izip, chain
from collections import deque

from .mesh2d import Polygon2d
from .vector2 import Vector2
from .utils import debug_draw_bool


def _cut_to_pieces(array, cut_items):
    if len(cut_items) == 0: cut_items = [array[0]]
    pieces = [[]]
    for el in array:
        pieces[-1].append(el)
        if el in cut_items:
            pieces.append([el])

    pieces[0] = pieces[-1] + pieces[0]
    del pieces[-1]
    return pieces



def _bool_cut_outline(this_poly, intersect_ids, other_poly):
        '''
        Returns 2 lists:
        1) all border pieces of this_poly outside other_poly, and
        2) all border pieces of this_poly inside other_poly.
        '''
        my_outline_pieces = _cut_to_pieces(this_poly.outline, intersect_ids)

        # find first piece that contains a non-intersection vertex
        first_piece, start_num = next(((piece, num) for num, piece \
            in enumerate(my_outline_pieces) if len(piece) > 2), None)

        # put this piece at the start of list of pieces
        my_outline_pieces = deque(my_outline_pieces)
        my_outline_pieces.rotate(-start_num)
        my_outline_pieces = list(my_outline_pieces)

        inside = other_poly.point_inside(this_poly.vertices[first_piece[1]])
        if inside:
            pieces_inside = my_outline_pieces[0::2]
            pieces_outside = my_outline_pieces[1::2]
        else:
            pieces_inside = my_outline_pieces[1::2]
            pieces_outside = my_outline_pieces[0::2]

        return pieces_outside, pieces_inside



def bool_subtract(A, B, canvas=None):
    '''
    Performs boolean subtraction of B from A. Returns a list of new Polygon2d instances
    or an empty list.
    '''

    # Copy the original polygons because they will be changed by this algorithm.
    A = A.copy()
    B = B.copy()

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
            seg_x = Vector2.where_segments_cross_inclusive(
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


    # # TODO check if entire B is inside A
    # if len(intersection_verts) == 0:
    #     # if A.point_inside(B.vertices[B.outline[0]]):
    #     #     A.add_hole()
    #     return []


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

        A_inserted_ids.update({x_idx: ins_idx for (x_idx, ins_idx) in izip(x_ids, inserted_ids)})
        A_new_ids.extend(inserted_ids)


    B_inserted_ids = {}
    for (B_edge, x_ids) in B_new_vert_lists.iteritems():
        x_verts = list(intersection_verts[idx] for idx in x_ids)
        inserted_ids = B.add_vertices_to_border(x_verts, B_edge)

        B_inserted_ids.update({x_idx: ins_idx for (x_idx, ins_idx) in izip(x_ids, inserted_ids)})
        B_new_ids.extend(inserted_ids)


    for x_idx in range(len(intersection_verts)):
        inserted_into_A = A_inserted_ids[x_idx]
        inserted_into_B = B_inserted_ids[x_idx]
        idx_map[inserted_into_B] = inserted_into_A



    # find pieces of A that are outside B
    A_pieces_outside, _ = _bool_cut_outline(A, A_new_ids, B)

    # find pieces of B that are inside A
    _, B_pieces_inside = _bool_cut_outline(B, B_new_ids, A)

    print("A outside B: {}".format(A_pieces_outside))
    print("B inside A: {}".format(B_pieces_inside))
    # debug draw
    if canvas:
        for idx in canvas.find_all(): canvas.delete(idx)
        debug_draw_bool(A, B, A_pieces_outside, B_pieces_inside, idx_map, canvas)


    new_outlines = []

    print("A pieces:")
    for p in A_pieces_outside: print(p)
    print("----------")
    print("B pieces:")
    for p in B_pieces_inside: print(p)


    while True:
        A_found = next(((pos, piece) for pos, piece in enumerate(A_pieces_outside) if len(piece) > 0), None)
        if A_found is None: break

        A_pos, A_piece = A_found
        new_outline = A_piece[:]

        del A_pieces_outside[A_pos]

        while True:


            # stop when outline becomes closed
            if new_outline[0] == new_outline[-1]: break


            # find B's piece to attach:
            B_found = next(((pos, piece) for pos, piece in enumerate(B_pieces_inside) \
                if len(piece) > 0 and new_outline[-1] == idx_map[piece[-1]]), None)

            if B_found is None: break

            B_pos, B_piece = B_found

            for B_idx in reversed(B_piece[1:-1]):
                B_vrt = B.vertices[B_idx]
                new_outline.append(len(A.vertices))
                A.vertices.append(B_vrt)
            new_outline.append(idx_map[B_piece[0]])

            del B_pieces_inside[B_pos]


            # find A's piece to attach
            A_found = next(((pos, piece) for pos, piece in enumerate(A_pieces_outside) \
                if len(piece) > 0 and new_outline[-1] == piece[0]), None)
            if A_found is None: break

            A_pos, next_A_piece = A_found

            new_outline.extend(next_A_piece[1:])
            del A_pieces_outside[A_pos]



        if new_outline[0] != new_outline[-1]:
            raise RuntimeError("Boolean operation failed")

        new_outlines.append(new_outline)


    # create new polys
    polys = []
    for new_outline in new_outlines:
        verts = list(A.vertices[idx] for idx in new_outline[:-1])
        print("creating new poly with {} verts".format(len(verts)))
        polys.append(Polygon2d(verts, range(len(verts))))


    # if we still have pieces of B inside A, they are holes that we must add to new polygons
    # B_pieces_inside = list(piece for piece in B_pieces_inside if len(piece) > 0)
    print("B pieces remaining: {}".format(B_pieces_inside))

    for B_piece in B_pieces_inside:
        verts = list(B.vertices[idx] for idx in B_piece[:-1])

        # find which poly this hole must be added to
        which_poly = next((poly for poly in polys if poly.point_inside(verts[0])), None)
        if which_poly is None:
            raise RuntimeError("Boolean operation failed (could not add hole)")
        which_poly.add_hole(verts)
        del B_piece[:]

    return polys
