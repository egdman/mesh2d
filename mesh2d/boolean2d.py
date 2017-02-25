from .mesh2d import Polygon2d
from .vector2 import Vector2
from itertools import chain



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
        my_outline_pieces = _cut_to_pieces(this_poly.outline, intersect_ids)

        # find first piece that contains a non-intersection vertex
        first_piece, start_num = next(((piece, num) for num, piece \
            in enumerate(my_outline_pieces) if len(piece) > 2))

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




def bool_subtract(A, B):
    '''
    Performs boolean subtraction of B from A.
    Returns a new Polygon2d instance even if it's identical to A.
    '''

    A_edges = Polygon2d.get_segments(chain([A.outline], A.holes))
    B_edges = Polygon2d.get_segments(chain([B.outline], B.holes))


    # find all intersections of borders
    inters = []
    for A_edge in A_edges:
        for B_edge in B_edges:
            seg_x = Vector2.where_segments_cross_inclusive(
                A.vertices[A_edge[0]],
                A.vertices[A_edge[1]],
                B.vertices[B_edge[0]],
                B.vertices[B_edge[1]])
            if seg_x is not None:
                inters.append((seg_x, A_edge, B_edge))

    # # find all other's vertices inside self
    # other_ids = chain(other.outline, *other.holes)
    # other_verts = (other.vertices[idx] for idx in other_ids)
    # verts_inside = list(vrt for vrt in other_verts if self.point_indise(vrt))

    # add intersections to A's border and remember their indices
    A_new_ids = []
    B_new_ids = []
    idx_map = {}
    for inter in inters:
        A_new_idx = A.add_vertex_to_border(inter[0], inter[1])
        B_new_idx = B.add_vertex_to_border(inter[0], inter[2])
        idx_map[A_new_idx] = B_new_idx
        A_new_ids.append(A_new_idx)
        B_new_ids.append(B_new_idx)


    # find pieces of A that are outside B
    A_pieces_outside, _ = _bool_cut_outline(A, A_new_ids, B)

    # find pieces of B that are inside A
    _, B_pieces_inside = _bool_cut_outline(B, B_new_ids, A)
    
    new_outlines = []

    while True:
        A_piece = next((piece for piece in A_pieces_outside if len(piece) > 0), None)
        if A_piece = None: break

        new_outline = A_piece[:]
        del A_piece[:]

        while True:

            # find B's piece to attach:
            B_piece = next((piece for piece in B_pieces_inside if len(piece) > 0 \
                and (idx_map[new_outline[-1]] == piece[0] \
                or idx_map[new_outline[-1]] == piece[-1])), None)
            if B_piece is None: break

            if idx_map[new_outline[-1]] == B_piece[-1]:
                B_piece = list(reversed(B_piece))

            for B_idx in B_piece[1:-1]:
                B_vrt = B.vertices[B_idx]
                A.add_vertex_to_loop(B_vrt, (new_outline[-1], new_outline[0]), new_outline)
            new_outline.append(B[-1])
            del B_piece[:]


            # find A's piece to attach
            next_A_piece = next((piece for piece in A_pieces_outside if len(piece) > 0 \
                and new_outline[-1] == piece[0]), None)
            if next_A_piece is None: break

            new_outline.extend(next_A_piece[:-1])
            del next_A_piece[:]

        if new_outline[0] == new_outline[-1]:
            new_outline = new_outline[:-1]

        new_outlines.append(new_outline)

    # create new polys

    polys = []
    for new_outline in new_outlines:
        verts = list(A.vertices[idx] for idx in new_outline)
        polys.append(Polygon2d(verts, new_outline))

    return polys