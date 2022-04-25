from itertools import chain, cycle, repeat
from functools import partial
from collections import defaultdict
from copy import deepcopy

try:
    from itertools import izip as zip
except ImportError:
    pass

from .polygon import Polygon2d
from .vector2 import Geom2, vec

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







from operator import itemgetter, lt as op_less, gt as op_more

class Ray:
    def __init__(self, tip, target):
        self.tip = tip
        self.target = target
        self.stride = target - tip
        if abs(self.stride[0]) > abs(self.stride[1]):
            self.main_component = 0
        else:
            self.main_component = 1

        if self.stride[self.main_component] > 0:
            self.less = op_less
        else:
            self.less = op_more

    # def __eq__(self, right):
    #     return (self.tip, self.target) == (right.tip, right.target)

    # def __ne__(self, right):
    #     return (self.tip, self.target) != (right.tip, right.target)


    def intersect_main_comp(self, A, B, AxB, area_diff):
        GxT = vec.cross2(self.target, self.tip)

        c0 = self.main_component
        i = (AxB / area_diff) * self.stride[c0] + (GxT / area_diff) * (B[c0] - A[c0])

        lower, upper = A[c0], B[c0]
        if self.less(upper, lower):
            lower, upper = upper, lower

        if self.less(i, lower):
            return lower
        if self.less(upper, i):
            return upper
        return i


def get_area_calculator(tip, target):
    stride = target - tip

    if tip.comps <= target.comps:
        def _calc_area(A):
            if target.comps <= A.comps:
                return vec.cross2(tip - A, target - A)
            else:
                return vec.cross2(target - A, stride)
    else:
        def _calc_area(A):
            if tip.comps <= A.comps:
                return vec.cross2(tip - A, target - A)
            else:
                return vec.cross2(tip - A, stride)
    return _calc_area


def trace_ray(ray, edges):
    calc_area = get_area_calculator(ray.tip, ray.target)

    # xl, xu = sorted((ray.tip[0], ray.target[0]))
    # yl, yu = sorted((ray.tip[1], ray.target[1]))

    for edge, seg in edges:
        B, A = seg

        # if A[0] < xl and B[0] < xl:
        #     continue
        # if A[0] >= xu and B[0] >= xu:
        #     continue
        # if A[1] < yl and B[1] < yl:
        #     continue
        # if A[1] >= yu and B[1] >= yu:
        #     continue

        orient_wrt_tip = Geom2.signed_area(A, B, ray.tip)
        orient_wrt_target = Geom2.signed_area(A, B, ray.target)

        if orient_wrt_tip == 0:
            if orient_wrt_target < 0:
                A, B = B, A # now orient_wrt_target > 0
            elif orient_wrt_target == 0:
                continue

        elif orient_wrt_tip > 0:
            if orient_wrt_target >= 0:
                continue
            else:
                A, B = B, A # now orient_wrt_target > 0

        # orient_wrt_tip < 0
        elif orient_wrt_target <= 0:
            continue
        # now orient_wrt_target > 0

        area_A = calc_area(A)
        if area_A < 0:
            continue

        area_B = calc_area(B)
        if area_A == 0:
            if area_B < 0:
                intersection_mc = A[ray.main_component]
            else:
                continue

        elif area_B < 0: # area_A > 0
            intersection_mc = ray.intersect_main_comp(A, B, vec.cross2(A, B), area_B - area_A)
        else: # area_B >= 0
            continue

        param_ray = (intersection_mc - ray.tip[ray.main_component]) / ray.stride[ray.main_component]
        param_ray = min(max(0., param_ray), 1.)

        s0, s1 = seg
        seg_stride = s1 - s0
        if abs(seg_stride[0]) > abs(seg_stride[1]):
            seg_main_component = 0
        else:
            seg_main_component = 1

        if seg_main_component == ray.main_component:
            param_seg = (intersection_mc - s0[seg_main_component]) / seg_stride[seg_main_component]
        else:
            c1 = 1 - ray.main_component
            intersection_sc = param_ray * ray.target[c1] + (1. - param_ray) * ray.tip[c1]
            param_seg = (intersection_sc - s0[seg_main_component]) / seg_stride[seg_main_component]

        param_seg = min(max(0., param_seg), 1.)
        yield edge, seg, param_ray, param_seg




def _add_intersections_to_polys(A, B):

    def get_segment_crds(seg, verts):
        return tuple(verts[idx] for idx in seg)

    def seglike_to_raylike(seglike):
        return (seglike[0], seglike[1] - seglike[0])

    def iter_segments(poly):
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


    # # find all A's edges that intersect B's border
    # def _A_edges():
    #     for A_loop in A.graph.loops:
    #         loop_it = A.graph.loop_iterator(A_loop)
    #         v0 = next(loop_it)
    #         v_first = v0

    #         inside_B = B.point_inside(A.vertices[v0])
    #         first_inside_B = inside_B

    #         for v1 in loop_it:
    #             inside_B_ = B.point_inside(A.vertices[v1])
    #             if inside_B_ != inside_B:
    #                 yield v0, v1

    #             inside_B = inside_B_
    #             v0 = v1

    #         if inside_B != first_inside_B:
    #             yield v0, v_first



    # Find all intersections of A and B borders.
    for A_edge in iter_segments(A):
        A_seg = get_segment_crds(A_edge, A.vertices)
        B_segs = ((B_edge, get_segment_crds(B_edge, B.vertices)) for B_edge in iter_segments(B))
        
        for B_edge, B_seg, a, b in trace_ray(Ray(*A_seg), B_segs):
            # print(f"intersection_mc={intersection_mc}")
        # for B_edge in iter_segments(B):

            # a, b, _ = Geom2.lines_intersect(seglike_to_raylike(A_seg), seglike_to_raylike(B_seg))

            if a<=0 or a>=1 or b<=0 or b>=1:
                print(f"!!!!! a={a} b={b}")

            # if 0 <= a and a <= 1 and 0 <= b and b <= 1:
            pair_index = len(intersection_param_pairs)
            intersection_param_pairs.append((a, b))

            A_new_vert_lists[A_edge].append(pair_index)
            B_new_vert_lists[B_edge].append(pair_index)
            # else:
            #     raise RuntimeError(f"this is wrong!! {a} and {b}")


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
