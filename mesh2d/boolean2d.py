from itertools import cycle, groupby
from functools import partial
from collections import defaultdict
from operator import itemgetter

try:
    from itertools import izip as zip
except ImportError:
    pass

from .polygon import Polygon2d
from .vector2 import Geom2
from .vector2 import vec
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



def _bool_impl(A, B, op, db_visitor=None):
    _find_all_intersections(A, B)


    A, B, A_contacts, B_contacts = _add_intersections_to_polys(A, B)

    if db_visitor:
        for idx, vert in enumerate(A.vertices):
            db_visitor.add_text(vert, str(idx), color="gold")

        for idx, vert in enumerate(B.vertices):
            db_visitor.add_text(vert + vec(15, 0), str(idx), color="green")

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
        area_a = self.calc_area(other._a)
        area_b = self.calc_area(other._b)

        if area_a == 0:
            return area_b < 0

        elif area_b == 0:
            return area_a < 0

        elif area_a < 0:
            return area_b < 0 or other.calc_area(self._a) > 0
        else:
            return area_b < 0 and other.calc_area(self._a) > 0


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
    # list of polygon segments that intersect the given segment,
    # each wrapped in a comparable object for sorting
    sections = []

    seg0, seg1 = segment
    calc_area = get_area_calculator(seg0, seg1)

    # this value is only meaningful if segment actually intersects polygon
    seg0_inside = False

    def _add_intersection(A, B, area_diff, A_idx):
        if area_diff > 0:
            section = ComparableSegment(A, B, A_idx)
        else:
            section = ComparableSegment(B, A, A_idx)

        nonlocal seg0_inside
        if len(sections) == 0:
            sections.append(section)
            seg0_inside = area_diff > 0
        elif section < sections[0]:
            sections.append(sections[0])
            sections[0] = section
            seg0_inside = area_diff > 0
        else:
            sections.append(section)

    traversal_idx = 0
    for loop in polygon.graph.loops:
        vertex_ids = polygon.graph.loop_iterator(loop)
        vertex_A = next(vertex_ids)
        A = polygon.vertices[vertex_A]
        area_A = calc_area(A)

        vertex_first = vertex_A
        area_first = area_A

        for vertex_B in vertex_ids:
            assert vertex_A == traversal_idx, f"{vertex_A} != {traversal_idx}" # TODO: this is not guaranteed in general
            B = polygon.vertices[vertex_B]
            area_B = calc_area(B)

            if _has_intersection(seg0, seg1, A, B, area_A, area_B):
                _add_intersection(A, B, area_B - area_A, vertex_A)

            vertex_A = vertex_B
            A = B
            area_A = area_B
            traversal_idx += 1

        assert vertex_A == traversal_idx, f"{vertex_A} != {traversal_idx}" # TODO: this is not guaranteed in general
        vertex_B = vertex_first
        B = polygon.vertices[vertex_B]
        area_B = area_first
        if _has_intersection(seg0, seg1, A, B, area_A, area_B):
            _add_intersection(A, B, area_B - area_A, vertex_A)
        traversal_idx += 1

    sections.sort()
    return tuple(section.payload for section in sections), seg0_inside


def calc_intersection_point(s1, r1, s2, r2):
    r2r1 = vec.cross2(r2, r1)
    r2s1 = vec.cross2(r2, s1)
    r2s2 = vec.cross2(r2, s2)
    return (r2s2 - r2s1) / r2r1


def _add_intersections_to_polys(A, B):
    sect_data = defaultdict(list)
    A_new = None
    loop_offset = 0
    for loop in A.graph.loops:
        A_new_verts = []

        for A_idx in A.graph.loop_iterator(loop):
            A_p0 = A.vertices[A_idx]
            A_p1 = A.vertices[A.graph.next[A_idx]]
            A_diff = A_p1 - A_p0
            A_new_verts.append(A_p0)

            intersections, _ = _intersect_segment_with_polygon((A_p0, A_p1), B)

            for B_idx in intersections:
                B_p0 = B.vertices[B_idx]
                B_diff = B.vertices[B.graph.next[B_idx]] - B_p0

                sect_param = calc_intersection_point(A_p0, A_diff, B_p0, B_diff)
                sect_data[B_idx].append((A_p0, A_p1, loop_offset + len(A_new_verts)))
                A_new_verts.append(A_p0 + sect_param * A_diff)

        loop_offset += len(A_new_verts)
        if A_new is None:
            A_new = Polygon2d(A_new_verts)
        else:
            A_new.add_hole(A_new_verts)


    ids_in_A, ids_in_B = [], []
    B_new = None
    loop_offset = 0
    for loop in B.graph.loops:
        B_new_verts = []

        for B_idx in B.graph.loop_iterator(loop):
            # we need to occlusion-sort A edges that intersect this B edge
            B_p0 = B.vertices[B_idx]
            B_new_verts.append(B_p0)

            if len(this_edge_intersections := sect_data[B_idx]) == 0:
                continue

            intersections = []
            B_calc_area = get_area_calculator(B_p0, B.vertices[B.graph.next[B_idx]])
            for A_p0, A_p1, idx_in_A in this_edge_intersections:
                if B_calc_area(A_p1) > B_calc_area(A_p0):
                    intersections.append(ComparableSegment(A_p0, A_p1, idx_in_A))
                else:
                    intersections.append(ComparableSegment(A_p1, A_p0, idx_in_A))

            intersections.sort()
            for isect in intersections:
                idx_in_A = isect.payload
                ids_in_A.append(idx_in_A)
                ids_in_B.append(loop_offset + len(B_new_verts))
                B_new_verts.append(A_new.vertices[idx_in_A])

        loop_offset += len(B_new_verts)
        if B_new is None:
            B_new = Polygon2d(B_new_verts)
        else:
            B_new.add_hole(B_new_verts)

    return A_new, B_new, ids_in_A, ids_in_B


def _find_all_intersections(A, B):
    # intersections sorted in traversal order of A
    A_sections = []

    # intersections that will later be sorted in traversal order of B
    B_sections = []
    for loop in A.graph.loops:
        for A_idx in A.graph.loop_iterator(loop):
            A_p0 = A.vertices[A_idx]
            A_p1 = A.vertices[A.graph.next[A_idx]]

            sections, p0_inside_B = _intersect_segment_with_polygon((A_p0, A_p1), B)

            for B_idx in sections:
                B_p0 = B.vertices[B_idx]
                B_p1 = B.vertices[B.graph.next[B_idx]]
                section = A_idx, B_idx, p0_inside_B

                if p0_inside_B:
                    occlusion_key = ComparableSegment(B_p0, B_p1, (*section, len(A_sections)))
                else:
                    occlusion_key = ComparableSegment(B_p1, B_p0, (*section, len(A_sections)))

                B_traversal_idx = B_idx # TODO: this is not guaranteed in general

                A_sections.append(section)
                # we'll sort lexicographically, first by traversal index and then by occlusion
                B_sections.append((B_traversal_idx, occlusion_key))
                p0_inside_B = not p0_inside_B

    B_sections.sort()
    B_sections = list(key.payload for _, key in B_sections)

    # set reference indexes to A_sections
    for idx, section in enumerate(B_sections):
        idx_into_A_sections = section[-1]
        A_sections[idx_into_A_sections] = *A_sections[idx_into_A_sections], idx

    print("A_sections:")
    for section in A_sections:
        print(section)

    print("B_sections:")
    for section in B_sections:
        print(section)


    # section_idx_A = 0
    # A_idx = A.graph.loops[0]
    # for A_loop in A.graph.loops:
    #     A_idx = A_loop

    #     while True:
    #         section = A_sections[section_idx_A]

    #         print(f"A: {A_idx}")
    #         while A_idx != section[0]:
    #             A_idx = A.graph.next[A_idx]
    #             if A_idx == A_loop:
    #                 # reached the end of the loop
    #                 section = None
    #                 break
    #             print(f"A: {A_idx}")

    #         if not section:
    #             break

    #         _, B_idx, is_inside, section_idx_B = section
    #         section_idx_B += 1
    #         section_idx_B %= len(B_sections)
    #         section = B_sections[section_idx_B]

    #         B_loop = B_idx
    #         print(f"B: {B_idx}")
    #         while B_idx != section[1]:
    #             B_idx = B.graph.next[B_idx]
    #             if B_idx == B_loop:
    #                 # reached the end of the loop
    #                 section = None
    #                 break
    #             print(f"B: {B_idx}")

    #         if not section:
    #             break

    #         A_idx, _, is_inside, section_idx_A = section
    #         section_idx_A += 1
    #         section_idx_A %= len(A_sections)


def bool_subtract(A, B, db_visitor=None):
    '''
    Performs boolean subtraction of B from A. Returns a list of new Polygon2d instances
    or an empty list.
    '''
    return _bool_impl(A, B, Subtraction, db_visitor)


def bool_add(A, B, db_visitor=None):
    '''
    Performs boolean subtraction of B from A. Returns a list of new Polygon2d instances
    or an empty list.
    '''
    return _bool_impl(A, B, Union, db_visitor)
