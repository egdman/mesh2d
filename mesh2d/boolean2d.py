from itertools import izip, chain
from collections import deque

from .mesh2d import Polygon2d
from .vector2 import Vector2
from .utils import debug_draw_bool


def _cut_to_pieces(array, cut_items):
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

        inside = other_poly.point_inside(this_poly.vertices[first_piece[1]])
        if inside: my_outline_pieces.rotate(-1)

        my_outline_pieces = list(my_outline_pieces)
        # pick every other piece because first one is outside
        pieces_outside = my_outline_pieces[::2]
        pieces_inside = my_outline_pieces[1::2]
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


    # TODO check if entire B is inside A
    if len(intersection_verts) == 0:
        return []


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
        A_piece = next((piece for piece in A_pieces_outside if len(piece) > 0), None)
        if A_piece is None: break

        new_outline = A_piece[:]
        del A_piece[:]
        print("creating new outline")
        while True:

            # find B's piece to attach:
            B_piece = next((piece for piece in B_pieces_inside if len(piece) > 0 \
                and (new_outline[-1] == idx_map[piece[0]] \
                or new_outline[-1] == idx_map[piece[-1]])), None)
            if B_piece is None: break

            if new_outline[-1] == idx_map[B_piece[-1]]:
                print("reversing")
                B_piece = list(reversed(B_piece))

            for B_idx in B_piece[1:-1]:
                B_vrt = B.vertices[B_idx]
                new_outline.append(len(A.vertices))
                A.vertices.append(B_vrt)
            new_outline.append(idx_map[B_piece[-1]])
            del B_piece[:]
            print(new_outline)

            # find A's piece to attach
            next_A_piece = next((piece for piece in A_pieces_outside if len(piece) > 0 \
                and new_outline[-1] == piece[0]), None)
            if next_A_piece is None: break

            new_outline.extend(next_A_piece[1:])
            del next_A_piece[:]
            print(new_outline)


        if new_outline[0] == new_outline[-1]:
            new_outline = new_outline[:-1]

        new_outlines.append(new_outline)

    # create new polys

    polys = []
    for new_outline in new_outlines:
        verts = list(A.vertices[idx] for idx in new_outline)
        print("creating new poly with {} verts".format(len(verts)))
        polys.append(Polygon2d(verts, range(len(verts))))

    return polys
