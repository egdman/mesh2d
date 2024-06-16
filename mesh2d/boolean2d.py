from itertools import cycle
from functools import partial
from collections import defaultdict
from operator import itemgetter

from .polygon import Polygon2d, Loops
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
    if op == Union:
        A_sections, A_order, B_sections, B_order, enclosures = _find_all_intersections(A, B)
        return _calc_polygon_union(A, B, A_sections, A_order, B_sections, B_order, enclosures, db_visitor)

    # if db_visitor:
        # for idx, vert in enumerate(A.vertices):
        #     db_visitor.add_text(vert, str(idx), color="gold")

        # for idx, vert in enumerate(B.vertices):
        #     db_visitor.add_text(vert + vec(15, 0), str(idx), color="green")

    A, B, A_contacts, B_contacts = _add_intersections_to_polys(A, B)

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
    def __init__(self, a, b):
        self._a = a
        self._b = b
        self.calc_area = get_area_calculator(self._a, self._b)

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
    '''
    returns 2 booleans: intersects_XY_segment, intersects_XY_ray
    '''
    if area_B > area_A:
        if area_A == 0 or (area_A < 0 and area_B > 0):
            # intersection is at A or between A and B
            area_Y = signed_area(A, B, Y)
            if area_Y < 0:
                has_section = signed_area(A, B, X) > 0
                return has_section, has_section
            else:
                return area_Y == 0, True

    elif area_B < area_A:
        if area_B == 0 or (area_B < 0 and area_A > 0):
            # intersection is at B or between A and B
            area_X = signed_area(A, B, X)
            if area_X < 0:
                return signed_area(A, B, Y) > 0, True
            else:
                has_section = area_X == 0
                return has_section, has_section

    return False, False


class AmbiguousEnclosure:
    pass

class NoEnclosure:
    pass

def _intersect_segment_with_polygon(segment, polygon):
    # list of polygon segments that intersect the given segment,
    # each wrapped in a comparable object for sorting
    sections = []

    seg0, seg1 = segment
    calc_area = get_area_calculator(seg0, seg1)

    # polygon loop that encloses seg0
    enclosure = NoEnclosure

    def _add_intersection(A, B, area_A, area_B, *section_data):
        if area_B > area_A:
            occlusion_key = ComparableSegment(A, B)
        else:
            occlusion_key = ComparableSegment(B, A)

        # calculate intersection coordinates
        area_A = abs(area_A)
        param = area_A / (area_A + abs(area_B))
        section_data = *section_data, A + param * (B - A)
        sections.append((occlusion_key, section_data))

    traversal_idx = 0
    for loop in polygon.graph.loops:
        sects_ray_odd_times = False

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

            sects_segment, sects_ray = _has_intersection(seg0, seg1, A, B, area_A, area_B)
            if sects_segment:
                _add_intersection(A, B, area_A, area_B, loop, vertex_A)
            if sects_ray:
                sects_ray_odd_times = not sects_ray_odd_times

            vertex_A = vertex_B
            A = B
            area_A = area_B
            traversal_idx += 1

        assert vertex_A == traversal_idx, f"{vertex_A} != {traversal_idx}" # TODO: this is not guaranteed in general
        vertex_B = vertex_first
        B = polygon.vertices[vertex_B]
        area_B = area_first

        sects_segment, sects_ray = _has_intersection(seg0, seg1, A, B, area_A, area_B)
        if sects_segment:
            _add_intersection(A, B, area_A, area_B, loop, vertex_A)
        if sects_ray:
            sects_ray_odd_times = not sects_ray_odd_times

        if sects_ray_odd_times:
            enclosure = loop

        traversal_idx += 1

    sections.sort(key=itemgetter(0))
    return tuple(section for _, section in sections), enclosure


def _find_point_enclosure(point, polygon):
    '''find the loop of polygon that encloses point, or return NoEnclosure'''
    def _has_intersection(A, B):
        if B[1] < A[1]:
            if A[1] == point[1] or (A[1] > point[1] and B[1] < point[1]):
                return signed_area(A, B, point) > 0

        elif B[1] > A[1]:
            if B[1] == point[1] or (B[1] > point[1] and A[1] < point[1]):
                return signed_area(A, B, point) <= 0

        return False

    enclosure = NoEnclosure
    outer_loop = polygon.graph.loops[0]

    for loop in polygon.graph.loops:
        point_in_loop = False

        vertex_ids = polygon.graph.loop_iterator(loop)
        vertex_first = next(vertex_ids)

        A = polygon.vertices[vertex_first]
        for vertex_B in vertex_ids:
            B = polygon.vertices[vertex_B]
            if _has_intersection(A, B):
                point_in_loop = not point_in_loop
            A = B

        B = polygon.vertices[vertex_first]
        if _has_intersection(A, B):
            point_in_loop = not point_in_loop

        if point_in_loop:
            if loop != outer_loop:
                return loop
            enclosure = loop
        elif loop == outer_loop:
            return NoEnclosure

    return enclosure


def _add_intersections_to_polys(A, B):
    sect_data = defaultdict(list)
    A_new = None
    loop_offset = 0
    for loop in A.graph.loops:
        A_new_verts = []

        for A_idx in A.graph.loop_iterator(loop):
            A_p0 = A.vertices[A_idx]
            A_p1 = A.vertices[A.graph.next[A_idx]]
            A_new_verts.append(A_p0)

            intersections, _ = _intersect_segment_with_polygon((A_p0, A_p1), B)

            for _, B_idx, sect_coords in intersections:
                B_p0 = B.vertices[B_idx]
                B_diff = B.vertices[B.graph.next[B_idx]] - B_p0

                sect_data[B_idx].append((A_p0, A_p1, loop_offset + len(A_new_verts)))
                A_new_verts.append(sect_coords)

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
                    intersections.append((ComparableSegment(A_p0, A_p1), idx_in_A))
                else:
                    intersections.append((ComparableSegment(A_p1, A_p0), idx_in_A))

            intersections.sort(key=itemgetter(0))
            for _, idx_in_A in intersections:
                ids_in_A.append(idx_in_A)
                ids_in_B.append(loop_offset + len(B_new_verts))
                B_new_verts.append(A_new.vertices[idx_in_A])

        loop_offset += len(B_new_verts)
        if B_new is None:
            B_new = Polygon2d(B_new_verts)
        else:
            B_new.add_hole(B_new_verts)

    return A_new, B_new, ids_in_A, ids_in_B


def _loop_iter(poly, loop):
    '''make an iterator over the loop's vertex coordinates'''
    return (poly.vertices[idx] for idx in poly.graph.loop_iterator(loop))


def _make_traversal_map(graph):
    trav_map = {}
    for loop_idx, loop in enumerate(graph.loops):
        for traversal_idx, vert_idx in enumerate(graph.loop_iterator(loop)):
            trav_map[vert_idx] = (loop_idx, traversal_idx)
    return trav_map


def _iterate_section(polygon, section_coords, start_idx, last_idx):
    yield section_coords
    vert_idx = start_idx
    while True:
        vert_idx = polygon.graph.next[vert_idx]
        yield polygon.vertices[vert_idx]
        if vert_idx == last_idx:
            break


def _find_all_intersections(A, B):
    # intersections sorted in traversal order of A
    A_sections = []
    A_order = Loops()

    # intersections that will later be sorted in traversal order of B
    B_sections = []

    enclosures = []

    B_trav_map = _make_traversal_map(B.graph)

    sections_count = 0
    for A_loop in A.graph.loops:
        A_loop_has_section = False

        for A_idx in A.graph.loop_iterator(A_loop):
            A_p0 = A.vertices[A_idx]
            A_p1 = A.vertices[A.graph.next[A_idx]]

            sections, enclosure = _intersect_segment_with_polygon((A_p0, A_p1), B)
            if not sections:
                continue

            if A_loop_has_section:
                tail_idx, tail_sect_coords, tail_exiting = A_sections[-1]
                A_sections[-1] = _iterate_section(A, tail_sect_coords, tail_idx, A_idx), tail_exiting
            else:
                A_loop_has_section = True
                A_idx_first = A_idx

            # is the 1st intersection on segment exiting or entering B?
            exiting = enclosure == B.graph.loops[0]
            for _, _, sect_coords in sections[:-1]:
                A_sections.append((sect_coords,), exiting)
                exiting = not exiting
            _, _, sect_coords = sections[-1]
            A_sections.append((A_idx, sect_coords, exiting))


            exiting = enclosure == B.graph.loops[0]
            for B_loop, B_idx, sect_coords in sections:
                if exiting:
                    occlusion_key = ComparableSegment(A_p1, A_p0)
                else:
                    occlusion_key = ComparableSegment(A_p0, A_p1)

                B_traversal_idx = B_trav_map[B_idx]

                # we'll sort lexicographically, first by traversal index and then by occlusion
                sorting_key = *B_traversal_idx, occlusion_key
                B_section = B_idx, sect_coords, exiting
                B_sections.append((sorting_key, B_section))
                exiting = not exiting

        if A_loop_has_section:
            assert len(A_sections) > sections_count
            A_order.add_loop(len(A_sections) - sections_count)
            sections_count = len(A_sections)

            tail_idx, tail_sect_coords, tail_exiting = A_sections[-1]
            A_sections[-1] = _iterate_section(A, tail_sect_coords, tail_idx, A_idx_first), tail_exiting
            enclosures.append(AmbiguousEnclosure)
        else:
            enclosures.append(enclosure)

    B_order = Loops()
    B_sections_out = []
    if len(B_sections) > 0:
        B_sections.sort(key=itemgetter(0))
        B_sections = iter(B_sections)

        (curr_loop_idx, curr_seg_idx, _), section = next(B_sections)
        B_sections_out = [section]
        B_idx_first, _, _ = section

        loop_length = 1
        for (loop_idx, seg_idx, _), section in B_sections:
            if loop_idx == curr_loop_idx:
                # loop continuation
                loop_length += 1
                if seg_idx == curr_seg_idx:
                    # segment continuation
                    _, sect_coords, exiting = B_sections_out[-1]
                    B_sections_out[-1] = (sect_coords,), exiting
                else:
                    # segment start
                    B_idx, _, _ = section
                    tail_idx, tail_sect_coords, tail_exiting = B_sections_out[-1]
                    B_sections_out[-1] = _iterate_section(B, tail_sect_coords, tail_idx, B_idx), tail_exiting
                    curr_seg_idx = seg_idx
            else:
                # loop start
                B_order.add_loop(loop_length)
                loop_length = 1
                curr_loop_idx = loop_idx
                curr_seg_idx = seg_idx
                tail_idx, tail_sect_coords, tail_exiting = B_sections_out[-1]
                B_sections_out[-1] = _iterate_section(B, tail_sect_coords, tail_idx, B_idx_first), tail_exiting
                B_idx_first, _, _ = section

            B_sections_out.append(section)

        tail_idx, tail_sect_coords, tail_exiting = B_sections_out[-1]
        B_sections_out[-1] = _iterate_section(B, tail_sect_coords, tail_idx, B_idx_first), tail_exiting

        print("A_sections:")
        for section in A_sections:
            print(section)

        print("B_sections:")
        for section in A_sections:
            print(section)

    return A_sections, A_order, B_sections_out, B_order, enclosures


def _calc_polygon_union(A, B, A_sections, A_order, B_sections, B_order, enclosures, db_visitor=None):
    # if len(sections) > 0:
    #     raise RuntimeError("Union of intersecting polygons not yet supported")

    assert len(A.graph.loops) == len(enclosures), f"{enclosures=}"

    new_graph = Loops()
    new_verts = []
    vertex_count = 0

    for first_sect_idx in A_order.loops:
        if sections[first_sect_idx] is None:
            continue

        sect_idx = first_sect_idx
        idx_A, idx_B, sp, exiting_B = sections[sect_idx]
        sections[sect_idx] = None
        first_idx_A = idx_A
        first_idx_B = idx_B
        while True:
            new_verts.append(sp)
            if db_visitor:
                db_visitor.add_text(sp, str(len(new_verts)-1), color="green")

            if exiting_B:
                # follow A
                vert_idx = idx_A
                sect_idx = A_order.next[sect_idx]
                print(f"follow A until {sect_idx=}")

                end_of_loop = sections[sect_idx] is None
                if end_of_loop:
                    print("closing the loop")
                    stay_on_segment = vert_idx == first_idx_A and sect_idx != first_sect_idx
                    if not stay_on_segment:
                        while True:
                            vert_idx = A.graph.next[vert_idx]
                            new_verts.append(A.vertices[vert_idx])
                            if db_visitor:
                                db_visitor.add_text(new_verts[-1], str(len(new_verts)-1), color="purple")
                            if vert_idx == first_idx_A:
                                break
                    break

                idx_A, idx_B, sp, exiting_B = sections[sect_idx]
                sections[sect_idx] = None
                stay_on_segment = vert_idx == idx_A and sect_idx != first_sect_idx

                if not stay_on_segment:
                    while True:
                        vert_idx = A.graph.next[vert_idx]
                        new_verts.append(A.vertices[vert_idx])
                        if db_visitor:
                            db_visitor.add_text(new_verts[-1], str(len(new_verts)-1), color="blue")
                        if vert_idx == idx_A:
                            break

            else:
                # follow B
                vert_idx = idx_B
                sect_idx = B_order.next[sect_idx]
                print(f"follow B until {sect_idx=}")

                end_of_loop = sections[sect_idx] is None
                if end_of_loop:
                    print("closing the loop")
                    stay_on_segment = vert_idx == first_idx_B and sect_idx != first_sect_idx
                    if not stay_on_segment:
                        while True:
                            vert_idx = B.graph.next[vert_idx]
                            new_verts.append(B.vertices[vert_idx])
                            if db_visitor:
                                db_visitor.add_text(new_verts[-1], str(len(new_verts)-1), color="purple")
                            if vert_idx == first_idx_B:
                                break
                    break

                idx_A, idx_B, sp, exiting_B = sections[sect_idx]
                sections[sect_idx] = None
                stay_on_segment = vert_idx == idx_B and sect_idx != first_sect_idx

                if not stay_on_segment:
                    while True:
                        vert_idx = B.graph.next[vert_idx]
                        new_verts.append(B.vertices[vert_idx])
                        if db_visitor:
                            db_visitor.add_text(new_verts[-1], str(len(new_verts)-1), color="red")
                        if vert_idx == idx_B:
                            break

        new_graph.add_loop(len(new_verts) - vertex_count)
        vertex_count = len(new_verts)

    def _add_holes(enclosures_AB):
        for hole_A, enclosure_B in enclosures_AB:
            if enclosure_B == NoEnclosure:
                yield _loop_iter(A, hole_A)

            elif enclosure_B not in (B.graph.loops[0], AmbiguousEnclosure):
                yield _loop_iter(A, hole_A)

        for hole_B in B.graph.loops[1:]:
            B_hole_vertex = B.graph.loops[0]
            enclosure_A = _find_point_enclosure(B.vertices[B_hole_vertex], A)
            if enclosure_A == NoEnclosure:
                yield _loop_iter(B, hole_B)
            elif enclosure_A not in (A.graph.loops[0], AmbiguousEnclosure):
                yield _loop_iter(B, hole_B)

    new_loops = []
    enclosures = iter(zip(A.graph.loops, enclosures))

    _, A_enclosed_in = next(enclosures)
    if A_enclosed_in == NoEnclosure:
        B_outer_vertex = B.graph.loops[0]
        B_enclosed_in = _find_point_enclosure(B.vertices[B_outer_vertex], A)
        if B_enclosed_in == NoEnclosure:
            # B is fully outside A
            assert len(sections) == 0
            return A, B

        elif B_enclosed_in == A.graph.loops[0]:
            new_loops = [_loop_iter(A, A.graph.loops[0])] + new_loops
            new_loops.extend(_add_holes(enclosures))

        else:
            # B is fully enclosed in a hole
            assert len(sections) == 0
            return A, B

    elif A_enclosed_in == AmbiguousEnclosure:
        # outer loops of A and B intersect, need to process non-intersecting holes
        new_loops.extend(_add_holes(enclosures))

    elif A_enclosed_in == B.graph.loops[0]:
        # A is enclosed in the outer loop of B, but not in any hole
        new_loops = [_loop_iter(B, B.graph.loops[0])] + new_loops
        new_loops.extend(_add_holes(enclosures))

    else:
        # A is fully enclosed in a hole
        assert len(sections) == 0
        return A, B

    for loop in new_loops:
        new_verts.extend(loop)
        new_graph.add_loop(len(new_verts) - vertex_count)
        vertex_count = len(new_verts)
    return (Polygon2d(new_verts, new_graph),)


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
