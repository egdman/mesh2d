import math

from collections import defaultdict
from operator import itemgetter, lt as op_less, gt as op_more
from itertools import chain
from copy import deepcopy
from time import clock
from contextlib import contextmanager
from random import randint, seed as randseed

try:
    from itertools import izip as zip
except ImportError:
    pass

from .vector2 import vec, Geom2

r2d = 180./math.pi

def _printf(s, *args):
    print(s.format(*args))

class debug:
    def __init__(self, cond):
        cond = not not cond
        if cond:
            self.__call__ = _printf
        else:
            self.__call__ = lambda *a, **kw: None
        self.cond = cond

    def __nonzero__(self):
        return self.cond

def rand_color(s):
    randseed(s)
    return "#{:02x}{:02x}{:02x}".format(randint(0, 255), randint(0, 255), randint(0, 255))

class CallTime:
    def __init__(self):
        self.accum = 0.
        self.n = 0

    def add(self, time):
        self.n += 1
        self.accum += time

    def average(self):
        return self.accum / self.n

timing = defaultdict(CallTime)

@contextmanager
def timed_exec(name):
    s = clock()
    yield
    timing[name].add(clock() - s)

@contextmanager
def counted_exec(name):
    timing[name].add(0.)
    yield


def signed_area(a, b, c):
    '''
    signed_area must return the same value when you rotate arguments (a, b, c -> b, c, a).
    If you flip order of arguments (a, b, c -> b, a, c), it must return the same value but with flipped sign
    '''
    if a.comps <= b.comps:
        if b.comps <= c.comps:
            # a, b, c
            return vec.cross2(a - c, b - c)
        else:
            # c, a, b
            return vec.cross2(c - b, a - b)
    else:
        if a.comps <= c.comps:
            # b, a, c
            return vec.cross2(c - b, a - c)
        else:
            # c, b, a
            return vec.cross2(c - a, a - b)


def _second_area(v0, v1, v2, diff):
    if v0.comps <= v1.comps:
        return vec.cross2(v0 - v1, v2 - v1)
    else:
        return vec.cross2(v0 - v1, diff)

def two_signed_areas(a, b, c, d):
    if a.comps <= b.comps:
        if b.comps <= c.comps:
            # a, b, c
            u = a - c
            return vec.cross2(u, b - c), lambda: _second_area(c, d, a, u)

        elif a.comps <= c.comps:
            # a, c, b
            return vec.cross2(c - b, a - b), lambda: _second_area(c, d, a, a - c)

        else:
            # c, a, b
            return vec.cross2(c - b, a - b), lambda: -_second_area(a, d, c, c - a)

    else:
        if a.comps <= c.comps:
            # b, a, c
            u = a - c
            return vec.cross2(c - b, u), lambda: _second_area(c, d, a, u)
        else:
            # c, b, a
            u = c - a
            return vec.cross2(b - a, u), lambda: -_second_area(a, d, c, u)


class Ray:
    def __init__(self, tip, target):
        self.tip = tip
        self.target = target
        self.GxT = vec.cross2(target, tip)
        self.stride = target - tip
        if abs(self.stride[0]) > abs(self.stride[1]):
            self.main_component = 0
        else:
            self.main_component = 1

        if self.stride[self.main_component] > 0:
            self.pick_least, self.less = min, op_less
        else:
            self.pick_least, self.less = max, op_more

    def __eq__(self, right):
        return (self.tip, self.target) == (right.tip, right.target)

    def __ne__(self, right):
        return (self.tip, self.target) != (right.tip, right.target)

    def intersect_main_comp(self, A, B, AxB, area_diff):
        c0 = self.main_component
        i = (AxB / area_diff) * self.stride[c0] + (self.GxT / area_diff) * (B[c0] - A[c0])

        lower, upper = A[c0], B[c0]
        if self.less(upper, lower):
            lower, upper = upper, lower

        if self.less(i, lower):
            return lower
        if self.less(upper, i):
            return upper
        return i

    def intersect_full(self, main_comp):
        c0 = self.main_component
        param = (main_comp - self.tip[c0]) / self.stride[c0]
        i = [0., 0.]
        c1 = 1 - c0
        i[c0] = main_comp
        i[c1] = param * self.target[c1] + (1. - param) * self.tip[c1]
        return vec(*i)


def make_sector(verts, topo, spike_eid):#, accum_angles, threshold):
    # TODO: support increased opening angles
    A, B, C = topo.get_triangle(spike_eid, topo.next_edge(spike_eid))
    return Ray(verts[C], verts[B]), Ray(verts[A], verts[B]) 

    # # increase opening angle of the sector due to threshold
    # extern_angle = calc_external_angle(unit_edges, spike_eid, topo.next_edge(spike_eid))
    # extra_opening = .5 * (extern_angle - accum_angles[spike_eid] + threshold)

    # cosine = math.cos(extra_opening)
    # sine = math.sin(extra_opening)

    # v1x, v1y = unit_edges[topo.next_edge(spike_eid)].comps
    # v2x, v2y = unit_edges[spike_eid].comps

    # v1x, v1y = (-cosine*v1x - sine*v1y, sine*v1x - cosine*v1y)
    # v2x, v2y = (cosine*v2x - sine*v2y, sine*v2x + cosine*v2y)
    # return vec(v1x, v1y), vec(v2x, v2y)


def trace_ray(verts, topo, ray, edges, db):
    lower = ray.tip[ray.main_component]
    upper = ray.target[ray.main_component]

    intersections = []
    for eid in edges:
        db("    EDGE {}", topo.debug_repr(eid))

        v0, v1 = topo.edge_verts(eid)
        B, A = verts[v0], verts[v1]
        if ray.target in (A, B):
            continue

        orientation = signed_area(A, B, ray.target)
        db("    orientation w.r.t G = {:.18f}", orientation)
        if orientation >= 0:
            continue

        area_A, area_B = two_signed_areas(ray.target, A, ray.tip, B)
        area_B = area_B()
        # at this point we know that area_A < area_B

        db("        a.y = {}, b.y = {}", area_A, area_B)

        if area_A > 0:
            continue

        elif area_A == 0:
            if area_B > 0:
                intersection = A[ray.main_component]
                if ray.less(lower, intersection) and ray.less(intersection, upper):
                    pass
                else:
                    continue
            else:
                continue

        else: # area_A < 0:
            if area_B > 0:
                intersection = ray.intersect_main_comp(A, B, vec.cross2(A, B), area_B - area_A)
     
            elif area_B == 0:
                intersection = B[ray.main_component]
                if ray.less(lower, intersection) and ray.less(intersection, upper):
                    pass
                else:
                    continue
            else:
                continue

        intersections.append((eid, intersection))

    if len(intersections) == 0:
        return None

    db("    INTERSECTIONS: {}", list((eid, ray.target[ray.main_component] - p) for eid, p in intersections))
    occluder, _ = ray.pick_least(intersections, key=itemgetter(1))
    return occluder


def calc_external_angle(verts, topo, eid, next_eid, triangle_area):
    if triangle_area == 0:
        return 0.

    A, B, C = topo.get_triangle(eid, next_eid)
    angle = math.acos(Geom2.cos_angle(verts[B] - verts[A], verts[C] - verts[B]))
    if triangle_area > 0:
        return angle
    else:
        return -angle


def calc_accum_angles(verts, topo, areas, threshold):
    accum_angles = [None] * topo.num_edges()
    for eid0 in topo.iterate_all_loops():
        loop_iter = iter(topo.iterate_loop_edges(eid0))
        tail = []
        for eid in loop_iter:
            tail.append(eid)
            if abs(calc_external_angle(verts, topo, eid, topo.next_edge(eid), areas[eid])) > threshold:
                break

        accum_angle = 0.
        for eid in chain(loop_iter, tail):
            accum_angle = max(accum_angle + calc_external_angle(verts, topo, eid, topo.next_edge(eid), areas[eid]), 0.) # accumulate only positive values
            accum_angles[eid] = accum_angle
            if accum_angle > threshold:
                accum_angle = 0.
    return accum_angles


def calc_accum_angle(verts, topo, prev_accum, eid, next_eid, triangle_area, threshold):
    this_external = calc_external_angle(verts, topo, eid, next_eid, triangle_area)
    if prev_accum > threshold:
        return max(this_external, 0.)
    return max(prev_accum + this_external, 0.)


def delete_redundant_portals(verts, topo, areas, accum_angles, threshold):
    deleted = [False] * topo.num_edges()

    def recalc_angles(eid):
        # get next counter-clockwise edge from eid skipping previously deleted edges
        eid_ccw = topo.prev_edge(topo.opposite(eid))
        while deleted[eid_ccw]:
            eid_ccw = topo.prev_edge(topo.opposite(eid_ccw))

        next_eid = topo.next_edge(eid)
        while deleted[next_eid]:
            next_eid = topo.next_edge(topo.opposite(next_eid))

        eid_ccw_prev = topo.prev_edge(eid_ccw)
        while deleted[eid_ccw_prev]:
            eid_ccw_prev = topo.prev_edge(topo.opposite(eid_ccw_prev))

        eid_area = calc_triangle_area(verts, topo, eid_ccw, next_eid)
        accum_angle = calc_accum_angle(verts, topo, accum_angles[eid_ccw_prev], eid_ccw, next_eid, eid_area, threshold)
        if accum_angle > threshold:
            # cannot delete eid
            return False, []

        new_accum_angles = [(eid_ccw, eid_area, accum_angle)]
        while True:
            eid = topo.next_edge(eid)
            while deleted[eid]:
                eid = topo.next_edge(topo.opposite(eid))

            next_eid = topo.next_edge(eid)
            while deleted[next_eid]:
                next_eid = topo.next_edge(topo.opposite(next_eid))

            eid_area = calc_triangle_area(verts, topo, eid, next_eid)
            accum_angle = calc_accum_angle(verts, topo, accum_angle, eid, next_eid, eid_area, threshold)
            if accum_angle > threshold:
                # cannot delete eid
                return False, []

            if accum_angle == accum_angles[eid]:
                # we can delete eid
                return True, new_accum_angles

            new_accum_angles.append((eid, eid_area, accum_angle))


    for eid in topo.iterate_all_internal_edges():
        if not topo.is_portal(eid): continue
        if deleted[eid]: continue

        redundant, new_accum = recalc_angles(eid)
        if not redundant: continue

        eid_ = topo.opposite(eid)
        redundant, new_accum_ = recalc_angles(eid_)
        if not redundant: continue

        deleted[eid] = deleted[eid_] = True
        accum_angles[eid] = accum_angles[eid_] = None
        areas[eid] = areas[eid_] = None

        for mod_eid, mod_triangle_area, mod_accum_angle in chain(new_accum, new_accum_):
            areas[mod_eid] = mod_triangle_area
            accum_angles[mod_eid] = mod_accum_angle

    for eid, d in enumerate(deleted):
        if d and topo.edges[eid] is not None:
            topo.remove_edge(eid)


def update_angles_after_connecting(verts, topo, areas, accum_angles, new_eid, threshold):
    areas.extend((None, None))
    accum_angles.extend((None, None))
    db=debug(False)#new_eid==25

    def _impl(new_eid):
        prev_eid = topo.prev_edge(new_eid)

        areas[prev_eid] = calc_triangle_area(verts, topo, prev_eid, new_eid)
        accum_angles[prev_eid] = calc_accum_angle(verts, topo,
            accum_angles[topo.prev_edge(prev_eid)], prev_eid, new_eid, areas[prev_eid], threshold)
        db("=== {} AA={}", topo.debug_repr(prev_eid), r2d*accum_angles[prev_eid])

        areas[new_eid] = calc_triangle_area(verts, topo, new_eid, topo.next_edge(new_eid))
        accum_angles[new_eid] = calc_accum_angle(verts, topo,
            accum_angles[prev_eid], new_eid, topo.next_edge(new_eid), areas[new_eid], threshold)
        db("=== {} AA={}", topo.debug_repr(new_eid), r2d*accum_angles[new_eid])

        accum_angle = accum_angles[new_eid]
        for eid in topo.iterate_loop_edges(topo.next_edge(new_eid)):
            if topo.next_edge(eid) == topo.opposite(new_eid): break

            accum_angle = calc_accum_angle(verts, topo, accum_angle, eid, topo.next_edge(eid), areas[eid], threshold)
            old_accum_angle = accum_angles[eid]

            accum_angles[eid] = accum_angle
            db("=== {} AA={}", topo.debug_repr(eid), r2d*accum_angles[eid])

            if accum_angle == old_accum_angle:
                break # no further change needed

            elif accum_angle > threshold:
                if old_accum_angle > threshold:
                    break # no further change needed
                else:
                    accum_angle = 0.

    _impl(new_eid)
    _impl(topo.opposite(new_eid))


def clamp_by_orientation(tip, L, R, point):
    orient_L = signed_area(tip, L, point)
    if orient_L <= 0:
        return L
    orient_R = signed_area(tip, R, point)
    if orient_R >= 0:
        return R
    return point


def select_connectable_endpoint(topo, vertex_is_connectable, eid):
    '''
    Gotta make sure that a portal endpoint that is selected for connection is actually connectable.
    This function cycles through the inbound eid's to find a connectable eid that belongs to the same room.
    If such eid cannot be found, it means that this endpoint has to have only 1 inbound eid belonging to
    the same room (the endpoint is a 'simple' vertex with no inbound 'antennas'),
    therefore we have to select it and hope that it will be occluded by another edge.
    '''
    eid_ = eid
    room_id = topo.room_id(eid)
    while topo.room_id(eid) != room_id or (not vertex_is_connectable[eid]):
        eid = topo.opposite(topo.next_edge(eid))
        if eid == eid_: break
    return eid


def find_connection_point_for_spike(verts, topo, areas, spike_eid, db_visitor):
    # see if vertex B is visible from tip
    def B_is_visible(area_ABC, area_BAT, area_CBT):
        if area_CBT > 0:
            return area_BAT >= 0 or area_ABC > 0
        else:
            return area_BAT > 0 and area_ABC > 0

    db = debug(spike_eid==-999 and db_visitor)

    tip = verts[topo.target(spike_eid)]
    clipped_verts = [(None, None)] * topo.num_edges()
    vertex_is_connectable = [False] * topo.num_edges()
    visible_edges = []
    skip = (topo.prev_edge(spike_eid), topo.next_edge(spike_eid))

    room = topo.rooms[topo.room_id(spike_eid)]
    for loop_eid in [room.outline] + room.holes:
        loop = iter(topo.iterate_loop_edges(loop_eid))
        eid = next(loop)
        v0, v1, v2 = topo.get_triangle(eid, topo.next_edge(eid))
        A, B, C = verts[v0], verts[v1], verts[v2]
        area_BAT = signed_area(B, A, tip)
        db("    {} area_BAT = {}", topo.debug_repr(eid), area_BAT)
        if area_BAT > 0:
            visible_edges.append(eid)

        area_CBT = signed_area(C, B, tip)
        vertex_is_connectable[eid] = eid not in skip and B_is_visible(areas[eid], area_BAT, area_CBT)

        for eid in loop:
            A, B = B, C
            C = verts[topo.target(topo.next_edge(eid))]
            area_BAT = area_CBT

            db("    {} area_BAT = {}", topo.debug_repr(eid), area_BAT)
            if area_BAT > 0:
                visible_edges.append(eid)

            area_CBT = signed_area(C, B, tip)
            vertex_is_connectable[eid] = eid not in skip and B_is_visible(areas[eid], area_BAT, area_CBT)

    db("    NUM VISIBLE EDGES IS {}", len(visible_edges))

    ray0, ray1 = make_sector(verts, topo, spike_eid)

    while True:
        db("-"*80)

        closest_edge_point, closest_edge_distSq, closest_eid = None, None, None
        closest_vertex_eid, closest_vertex_distSq = None, None

        for eid in visible_edges:
            db("    EDGE {}", topo.debug_repr(eid))
            # we must reverse the segment to match the orientation of the sector rays
            v0, v1 = topo.edge_verts(eid)
            B, A = verts[v0], verts[v1]

            areas0_A, calc_area0_B = two_signed_areas(ray0.target, A, ray0.tip, B)
            if areas0_A >= 0:
                areas1_A, calc_area1_B = two_signed_areas(ray1.target, A, ray1.tip, B)
                if areas1_A < 0:
                    clip_A = A
                    areas1_B = calc_area1_B()

                    if areas1_B <= 0:
                        clip_B = B
                    else:
                        AxB = vec.cross2(A, B)
                        clip_B = ray1.intersect_full(ray1.intersect_main_comp(A, B, AxB, areas1_B - areas1_A))
                        clip_B = clamp_by_orientation(tip, clip_A, B, clip_B)
                elif areas1_A == 0:
                    areas1_B = calc_area1_B()
                    if areas1_B > 0:
                        clip_A = A
                        clip_B = A
                    else:
                        clip_A, clip_B = None, None
                else:
                    clip_A, clip_B = None, None
            else: # areas0_A < 0
                areas0_B = calc_area0_B()
                if areas0_B > 0:
                    AxB = vec.cross2(A, B)
                    clip_A = ray0.intersect_full(ray0.intersect_main_comp(A, B, AxB, areas0_B - areas0_A))
                    clip_A = clamp_by_orientation(tip, A, B, clip_A)
                    areas1_B = signed_area(ray1.target, B, ray1.tip)
                    if areas1_B <= 0:
                        clip_B = B
                    else:
                        AxB = vec.cross2(A, B)
                        areas1_A = signed_area(ray1.target, A, ray1.tip)
                        clip_B = ray1.intersect_full(ray1.intersect_main_comp(A, B, AxB, areas1_B - areas1_A))
                        clip_B = clamp_by_orientation(tip, clip_A, B, clip_B)

                elif areas0_B == 0:
                    clip_A = B
                    clip_B = B
                else:
                    clip_A, clip_B = None, None

            db("      CLIPPED {}, {}", clip_B, clip_A)
            clipped_verts[eid] = clip_B, clip_A
            if clip_A is None:
                continue

            points = []

            distSq_B = (clip_B - tip).normSq()
            if clip_B == B:
                if vertex_is_connectable[topo.prev_edge(eid)]:
                    points.append((B, distSq_B))
                    if closest_vertex_distSq is None or distSq_B <= closest_vertex_distSq:
                        closest_vertex_distSq = distSq_B
                        closest_vertex_eid = topo.prev_edge(eid)

            else:
                points.append((clip_B, distSq_B))

            distSq_A = (clip_A - tip).normSq()
            if clip_A == A:
                if vertex_is_connectable[eid]:
                    points.append((A, distSq_A))
                    if closest_vertex_distSq is None or distSq_A <= closest_vertex_distSq:
                        closest_vertex_distSq = distSq_A
                        closest_vertex_eid = eid

            else:
                points.append((clip_A, distSq_A))

            if clip_B != clip_A:
                diff = clip_A - clip_B
                ortho_param = Geom2.project_to_line(tip, (clip_B, diff))
                if 0 < ortho_param and ortho_param < 1:
                    ortho_point = clip_B + ortho_param*diff
                    ortho_point = clamp_by_orientation(tip, clip_A, clip_B, ortho_point)
                    if ortho_point not in (clip_A, clip_B):
                        points.append((ortho_point, (ortho_point - tip).normSq()))

            db("        -- {} POINTS", len(points))
            if len(points) == 0:
                continue

            edge_point, edge_distSq = min(points, key=itemgetter(1))
            db("        edge_distSq = {}", edge_distSq)
            if closest_edge_distSq is None or edge_distSq <= closest_edge_distSq:
                closest_edge_distSq = edge_distSq
                closest_edge_point = edge_point
                closest_eid = eid


        if closest_eid is None:
            if db_visitor:
                db_visitor.add_polygon((ray0.tip, ray0.target),color="gold")
                db_visitor.add_polygon((ray1.tip, ray1.target),color="gold")
                db_visitor.add_text(tip, str(topo.target(spike_eid)), color="gold")

            raise RuntimeError("Could not find a single edge for spike {} -> {}".format(
                    topo.debug_repr(spike_eid), topo.debug_repr(topo.next_edge(spike_eid))))

        # if the closest vertex belongs to the closest edge,
        # we can connect to the vertex instead of the edge, even though it's more distant
        if closest_vertex_eid is not None and \
            closest_eid in (closest_vertex_eid, topo.next_edge(closest_vertex_eid)):
            target_eid = closest_vertex_eid
            target_coords = verts[topo.target(target_eid)]

            db("    connecting to vertex instead of edge, closest_eid={}, closest_vertex_eid={}",
                topo.debug_repr(closest_eid), topo.debug_repr(closest_vertex_eid))

        else:
            # we're not allowed to create T-shaped portals
            if topo.is_portal(closest_eid):
                db("    PICKED A PORTAL {}", topo.debug_repr(closest_eid))

                if closest_eid == topo.prev_edge(spike_eid):
                    target_eid = topo.prev_edge(closest_eid)

                elif topo.prev_edge(closest_eid) == topo.next_edge(spike_eid):
                    target_eid = closest_eid

                else:
                    clip0, clip1 = clipped_verts[closest_eid]
                    v0, v1 = topo.edge_verts(closest_eid)
                    r_inside = verts[v0] == clip0
                    l_inside = verts[v1] == clip1

                    # only one endpoint is inside sector
                    if r_inside != l_inside:
                        db("ONE ENDPOINT INSIDE")
                        if r_inside:
                            target_eid = topo.prev_edge(closest_eid)
                        else:
                            target_eid = closest_eid

                    # portal is fully inside sector, select closest endpoint
                    elif r_inside:
                        db("FULLY INSIDE")
                        if (clip0 - tip).normSq() < (clip1 - tip).normSq():
                            target_eid = topo.prev_edge(closest_eid)
                        else:
                            target_eid = closest_eid

                    # only the middle section passes through sector
                    else:
                        db("ONLY MIDDLE SECTION INSIDE")
                        target_eid = topo.prev_edge(closest_eid)

                target_eid = select_connectable_endpoint(topo, vertex_is_connectable, target_eid)
                target_coords = verts[topo.target(target_eid)]

            else:
                target_eid = closest_eid
                target_coords = closest_edge_point


        db("    SELECTED TARGET {}", topo.debug_repr(target_eid))
        db("CHECK OCCLUSION {} TO {}", topo.debug_repr(spike_eid), target_coords)

        occluder = trace_ray(verts, topo, Ray(tip, target_coords), visible_edges, db)
        if occluder is None or occluder in (target_eid, topo.next_edge(target_eid)):
            db("    NOT OCCLUDED")
            return target_eid, target_coords

        else:
            db("    OCCLUDED BY {}", topo.debug_repr(occluder))
            clip0, clip1 = clipped_verts[occluder]
            oc0, oc1 = topo.edge_verts(occluder)

            ray0_ = ray0
            ray1_ = ray1
            # if occluder is fully outside the sector
            if clip0 is None:
                ray0 = Ray(tip, verts[oc1])
                ray1 = Ray(tip, verts[oc0])
                db("    OCCLUDER IS OUTSIDE SECTOR")
            else:
                ray0 = Ray(tip, clip1)
                ray1 = Ray(tip, clip0)
                db("    OCCLUDER IS  INSIDE SECTOR")

            if ray0 == ray0_ and ray1 == ray1_:
                raise RuntimeError("infinite inner loop")

            visible_edges_ = visible_edges[:]
            visible_edges = []
            for eid in visible_edges_:
                if eid == occluder:
                    visible_edges.append(eid)
                else:
                    v0, v1 = topo.edge_verts(eid)
                    if signed_area(verts[v0], ray0.target, ray1.target) >= 0 and \
                       signed_area(verts[v1], ray0.target, ray1.target) >= 0:
                        visible_edges.append(eid)


def calc_triangle_area(verts, topo, eid, next_eid):
    A, B, C = topo.get_triangle(eid, next_eid)
    return signed_area(verts[A], verts[B], verts[C])


def calc_areas(verts, topo):
    areas = [None] * topo.num_edges()
    for eid in topo.iterate_all_internal_edges():
        areas[eid] = calc_triangle_area(verts, topo, eid, topo.next_edge(eid))
    return areas


def convex_subdiv(verts, topo, threshold, db_visitor=None):
    """
    This function uses algorithm from R. Oliva and N. Pelechano - 
    Automatic Generation of Suboptimal NavMeshes
    """

    if db_visitor:
        for eid in topo.iterate_all_internal_edges():
            vid = topo.target(eid)
            db_visitor.add_text(verts[vid], str(vid), color="#93f68b")

    areas = calc_areas(verts, topo)
    accum_angles = calc_accum_angles(verts, topo, areas, threshold)
    # for eid in topo.iterate_all_internal_edges():
    #     print("{}, AA = {}".format(topo.debug_repr(eid), accum_angles[eid]*r2d))

    def connect_to_new_vertex(start_eid, eid, point):
        new_eid = topo.insert_vertex(len(verts), eid)

        if db_visitor:
            db_visitor.add_text(point, str(topo.target(new_eid)), color="cyan")

        assert topo.next_edge(new_eid) == eid
        assert topo.prev_edge(eid) == new_eid
        assert topo.room_id(new_eid) is not None
        assert topo.room_id(eid) is not None

        # add angles for new_eid
        areas.extend((None, None))
        accum_angles.extend((None, None))

        prev_accum = accum_angles[topo.prev_edge(new_eid)]
        if prev_accum > threshold:
            accum_angles[new_eid] = 0.
        else:
            accum_angles[new_eid] = max(prev_accum, 0.)

        assert len(verts) == topo.target(new_eid), "something went wrong here"

        verts.append(point)
        prev_eid = topo.prev_edge(new_eid)
        next_eid = topo.next_edge(eid)
        areas[prev_eid] = calc_triangle_area(verts, topo, prev_eid, new_eid)
        areas[eid] = calc_triangle_area(verts, topo, eid, next_eid)
        areas[new_eid] = None # will be calculated in the function connect_to_target

        # If the new point lies on the straight line of the edge,
        #  the new accumulated angle will be very close to the previous value,
        #  but it must be recalculated anyway.
        accum_angles[prev_eid] = calc_accum_angle(verts, topo,
            accum_angles[topo.prev_edge(prev_eid)], prev_eid, new_eid, areas[prev_eid], threshold)

        return connect_to_target(start_eid, new_eid)

    def connect_to_target(start_eid, target_eid):
        if target_eid in (topo.prev_edge(start_eid), topo.next_edge(start_eid)):
            raise RuntimeError("tried to connect {} to {}".format(topo.debug_repr(start_eid), topo.debug_repr(target_eid)))

        # _printf("    connecting {} to {}", topo.debug_repr(start_eid), topo.debug_repr(target_eid))

        if db_visitor:
            db_visitor.add_polygon((verts[topo.target(start_eid)], verts[topo.target(target_eid)]), color="cyan")

        new_eid = topo.connect(start_eid, target_eid, verts)
        update_angles_after_connecting(verts, topo, areas, accum_angles, new_eid, threshold)
        return new_eid, topo.opposite(new_eid)


    # sorting spikes by external angle makes it faster!
    all_edges = list(sorted(tuple(topo.iterate_all_internal_edges()),
        key=lambda eid: -accum_angles[eid]))

    print("NUM EDGES IS {}".format(len(all_edges)))

    for spike_eid in all_edges:
        if accum_angles[spike_eid] <= threshold: continue # not a spike

        # if db_visitor:
        #     vid = topo.target(spike_eid)
        #     db_visitor.add_text(verts[vid], str(vid), color="#93f68b")

        # _printf("SPIKE {} ==> {}, AA={}", topo.debug_repr(spike_eid), topo.debug_repr(topo.next_edge(spike_eid)), r2d*accum_angles[spike_eid])

        target_eid, target_coords = find_connection_point_for_spike(verts, topo, areas, spike_eid, db_visitor)
        if target_coords == verts[topo.target(target_eid)]:
            all_edges.extend(connect_to_target(spike_eid, target_eid))
        else:
            all_edges.extend(connect_to_new_vertex(spike_eid, target_eid, target_coords))

        if accum_angles[spike_eid] > threshold:
            all_edges.append(spike_eid)

    delete_redundant_portals(verts, topo, areas, accum_angles, threshold)


r'''
        |
       /|
      //|
     ///|
    ////|
   /////|
        |
        |
        |
        |
        |
        |/////
        |   /
        |  /
        | /
        |/
        |
'''

class EdgeStruct(object):
    def __init__(self, prev, next, target, room_id):
        self.prev = prev
        self.next = next
        self.target = target
        self.room_id = room_id

    def __repr__(self):
        return "p:{}, n:{}, t:{}, r:{}".format(self.prev, self.next, self.target, self.room_id)


class Room():
    def __init__(self, outline, holes):
        self.outline = outline
        self.holes = list(holes)

    def debug_repr(self, topo):
        outl = topo.debug_repr(self.outline)
        holes = ", ".join((topo.debug_repr(h) for h in self.holes))
        return "outline: {}, holes: {}".format(outl, holes)



class Topology(object):

    def debug_repr(self, eid):
        vid0 = self.edges[self.opposite(eid)].target
        vid1 = self.edges[eid].target
        return "e{}, {}->{} @r{}".format(eid, vid0, vid1, self.edges[eid].room_id)

    def debug_verbose(self, eid):
        e = self.edges[eid]
        return "edge {}, prv {}, nxt {}".format(self.debug_repr(eid), self.debug_repr(e.prev), self.debug_repr(e.next))

    @staticmethod
    def opposite(eid):
        return eid - 1 if eid & 1 else eid + 1

    def next_edge(self, eid):
        return self.edges[eid].next

    def prev_edge(self, eid):
        return self.edges[eid].prev

    def num_edges(self):
        return len(self.edges)

    def target(self, eid):
        '''
        Get vid of the target vertex for the given half-edge.
        '''
        return self.edges[eid].target

    def edge_verts(self, eid):
        return self.edges[self.opposite(eid)].target, self.edges[eid].target

    def room_id(self, eid):
        return self.edges[eid].room_id

    def is_portal(self, eid):
        return self.edges[eid].room_id is not None and \
               self.edges[self.opposite(eid)].room_id is not None

    def iterate_loop_edges(self, eid0):
        yield eid0
        eid = self.edges[eid0].next
        while eid != eid0:
            yield eid
            eid = self.edges[eid].next

    def iterate_all_loops(self):
        for room in self.rooms:
            if room is not None:
                yield room.outline
                for hole in room.holes:
                    yield hole

    def iterate_all_internal_edges(self):
        return chain(*(self.iterate_loop_edges(eid) for eid in self.iterate_all_loops()))

    def iterate_room_edges(self, eid):
        room = self.rooms[self.edges[eid].room_id]
        edge_ids = [room.outline] + room.holes
        return chain(*(self.iterate_loop_edges(eid) for eid in edge_ids))

    def insert_vertex(self, new_vid, edge_to_split):
        '''
        Inserts a new vertex into the given edge eid_given.
        Returns the new edge inserted before eid_given,
        such that the target of the new edge is the new vertex.
        '''
        eid_given = edge_to_split
        eid_oppos = self.opposite(edge_to_split)

        if self.edges[eid_oppos].room_id is None:
            before_given = len(self.edges) + 1
            after_oppos = len(self.edges)
        else:
            before_given = len(self.edges)
            after_oppos = len(self.edges) + 1

        edge_before_given = EdgeStruct(
            prev=self.edges[eid_given].prev,
            next=eid_given,
            target=new_vid,
            room_id=self.edges[eid_given].room_id)

        edge_after_oppos = EdgeStruct(
            prev=eid_oppos,
            next=self.edges[eid_oppos].next,
            target=self.edges[eid_oppos].target,
            room_id=self.edges[eid_oppos].room_id)

        self.edges.extend((None, None))
        self.edges[before_given] = edge_before_given
        self.edges[after_oppos] = edge_after_oppos

        self.edges[eid_given].prev = before_given
        self.edges[eid_oppos].next = after_oppos
        self.edges[eid_oppos].target = new_vid
        self.edges[edge_before_given.prev].next = before_given
        self.edges[edge_after_oppos.next].prev = after_oppos
        return before_given


    def same_loop(self, eid0, eid1):
        for eid in self.iterate_loop_edges(eid0):
            if eid == eid1:
                return True
        return False

    def inside_loop(self, query_eid, loop_eid, verts):
        query_pt = verts[self.edges[query_eid].target]
        vert_ids = (self.edges[eid].target for eid in self.iterate_loop_edges(loop_eid))
        polyline = (verts[vid] - query_pt for vid in vert_ids)
        return Geom2.is_origin_inside_polyline(polyline)

    def loop_is_ccw(self, loop_eid, verts):
        vert_ids = (self.edges[eid].target for eid in self.iterate_loop_edges(loop_eid))
        points = (verts[vid] for vid in vert_ids)
        return Geom2.poly_signed_area(points) > 0

    def get_outline(self, eid):
        return self.rooms[self.edges[eid].room_id].outline

    def get_triangle(self, eid, next_eid):
        op_eid = self.opposite(eid)
        return self.edges[op_eid].target, self.edges[eid].target, self.edges[next_eid].target

    def connect(self, source_eid, target_eid, verts):
        def _change_topology():
            vid0 = self.target(source_eid)
            vid1 = self.target(target_eid)

            enxt0 = self.edges[source_eid].next
            enxt1 = self.edges[target_eid].next

            new_eid = len(self.edges)
            new_eid_oppo = new_eid + 1
            self.edges.append(EdgeStruct(prev=source_eid, next=enxt1, target=vid1, room_id=None))
            self.edges.append(EdgeStruct(prev=target_eid, next=enxt0, target=vid0, room_id=None))

            self.edges[source_eid].next = new_eid
            self.edges[enxt0].prev = new_eid_oppo
            self.edges[target_eid].next = new_eid_oppo
            self.edges[enxt1].prev = new_eid
            return new_eid, new_eid_oppo

        assert self.room_id(source_eid) is not None
        assert self.room_id(target_eid) is not None
        assert self.room_id(source_eid) == self.room_id(target_eid)

        old_room_id = self.edges[source_eid].room_id

        # splitting one loop into two, therefore creating a new room
        if self.same_loop(source_eid, target_eid):

            # splitting the outline
            if self.same_loop(source_eid, self.get_outline(source_eid)):
                unsorted_holes = self.rooms[old_room_id].holes
                old_outline, new_outline = _change_topology()
                old_loops = [old_outline]
                new_loops = [new_outline]


            # splitting one of holes
            else:
                # find out which hole we are splitting and remove it from the room
                unsorted_holes = [hole_eid for hole_eid in self.rooms[old_room_id].holes if not self.same_loop(hole_eid, source_eid)]
                modified_hole, new_outline = _change_topology()
                if not self.loop_is_ccw(modified_hole, verts):
                    modified_hole, new_outline = new_outline, modified_hole

                old_loops = [self.rooms[old_room_id].outline, modified_hole]
                new_loops = [new_outline]

            # sort holes; TODO: this can be done faster
            for hole_eid in unsorted_holes:
                if self.inside_loop(hole_eid, new_loops[0], verts):
                    new_loops.append(hole_eid)
                else:
                    old_loops.append(hole_eid)

            for loop in old_loops:
                for eid in self.iterate_loop_edges(loop):
                    self.edges[eid].room_id = old_room_id

            new_room_id = len(self.rooms)
            for loop in new_loops:
                for eid in self.iterate_loop_edges(loop):
                    self.edges[eid].room_id = new_room_id

            self.rooms[old_room_id] = Room(outline=old_loops[0], holes=old_loops[1:])
            self.rooms.append(Room(outline=new_loops[0], holes=new_loops[1:]))
            new_eid = new_loops[0]

        # joining two loops into one, therefore no new room is created
        else:
            remaining_holes = []
            for hole_eid in self.rooms[old_room_id].holes:
                if not self.same_loop(hole_eid, source_eid) and not self.same_loop(hole_eid, target_eid):
                    remaining_holes.append(hole_eid)

            e0, e1 = _change_topology()
            outline = self.rooms[old_room_id].outline
            if not self.same_loop(outline, e0):
                remaining_holes.append(e0)

            self.edges[e0].room_id = self.edges[e1].room_id = old_room_id
            self.rooms[old_room_id] = Room(outline=outline, holes=remaining_holes)
            new_eid = e0

        if self.target(new_eid) != self.target(target_eid):
            return self.opposite(new_eid)
        return new_eid
 

    def remove_edge(self, eid):
        eid_ = self.opposite(eid)
        room0, room1 = self.edges[eid].room_id, self.edges[eid_].room_id

        # keep the room which is None. If both aren't, keep room0
        if room1 is None:
            eid, eid_ = eid_, eid
            room0, room1 = room1, room0
        else:
            assert len(self.rooms[room1].holes) == 0

        for eid0 in self.iterate_room_edges(eid_):
            self.edges[eid0].room_id = room0

        eid1 = self.next_edge(eid)
        eid2 = self.prev_edge(eid_)
        eid3 = self.next_edge(eid_)
        eid4 = self.prev_edge(eid)

        if room0 is not None:
            room = self.rooms[room0]
            assert len(room.holes) == 0

            if room.outline == eid:
                room.outline = eid1

        if room1 is not None:
            self.rooms[room1] = None

        self.edges[eid1].prev = eid2
        self.edges[eid2].next = eid1
        self.edges[eid3].prev = eid4
        self.edges[eid4].next = eid3  

        self.edges[eid] = self.edges[eid_] = None



    @staticmethod
    def of_a_polygon(poly):
        t = Topology()
        t.edges = []
        loops = []

        for loop_start in poly.graph.loops:
            loop_verts = poly.graph.loop_iterator(loop_start)
            first_eid = len(t.edges)
            loops.append(first_eid + 1)

            src0 = next(loop_verts)
            tgt0 = poly.graph.next[src0]
            t.edges.append(EdgeStruct(target=tgt0, room_id=None, prev=None, next=None))
            t.edges.append(EdgeStruct(target=src0, room_id=0, prev=None, next=None))

            for src in loop_verts:
                tgt = poly.graph.next[src]

                t.edges[-2].next = len(t.edges)
                t.edges[-1].prev = len(t.edges) + 1
                t.edges.append(EdgeStruct(target=tgt, room_id=None, prev=len(t.edges)-2, next=None))
                t.edges.append(EdgeStruct(target=src, room_id=0, prev=None, next=len(t.edges)-2))

            t.edges[-2].next = first_eid
            t.edges[-1].prev = first_eid + 1
            t.edges[first_eid].prev = len(t.edges) - 2
            t.edges[first_eid + 1].next = len(t.edges) - 1

        t.rooms = [Room(outline=loops[0], holes=loops[1:])]
        return t


def inside_bbox(bbox, query_point):
    p0, p1 = bbox
    a = all(comp >= 0 for comp in (query_point - p0).comps)
    b = all(comp >= 0 for comp in (p1 - query_point).comps)
    return a and b


class Mesh2d(object):
    def __init__(self, poly, convex_relax_thresh = 0.0, db_visitor=None):
        self.vertices = deepcopy(poly.vertices)

        # degrees to radian
        convex_relax_thresh = math.pi * convex_relax_thresh / 180.

        topo = Topology.of_a_polygon(poly)

        with timed_exec("convex_subdiv"):
            convex_subdiv(self.vertices, topo, convex_relax_thresh, db_visitor)


        self.portals = list()
        for eid in topo.iterate_all_internal_edges():
            if topo.is_portal(eid):
                vid0, vid1 = topo.edge_verts(eid)
                if vid0 < vid1:
                    self.portals.append((vid0, vid1))

        # make rooms (a room is just a list of vertex ids that form a nearly convex polygon)
        self.rooms = []
        self.bboxes = []
        self.adjacent_rooms = defaultdict(list)
        room_id_map = {}

        for room_id, room in enumerate(topo.rooms):
            if room is None: continue

            vert_ids = tuple(topo.target(eid) for eid in topo.iterate_room_edges(room.outline))
            self.rooms.append(vert_ids)
            self.bboxes.append(vec.aabb(self.vertices[vid] for vid in vert_ids))
            room_id_map[room_id] = len(self.rooms) - 1

        for room_id, room in enumerate(topo.rooms):
            if room is None: continue

            room_id = room_id_map[room_id] 
            for eid in topo.iterate_room_edges(room.outline):
                adj_room_id = topo.room_id(topo.opposite(eid))
                if adj_room_id is None: continue

                adj_room_id = room_id_map[adj_room_id]

                # store the portal endpoints along with the adjacent room index
                portal = tuple(sorted(topo.edge_verts(eid)))
                self.adjacent_rooms[room_id].append((adj_room_id, portal))


        self.outline = list(poly.graph.loop_iterator(poly.graph.loops[0]))
        self.holes = list(list(poly.graph.loop_iterator(h)) for h in poly.graph.loops[1:])

        for name, t in timing.items():
            print("{}, {} times, total time {}".format(name, t.n, t.accum*1000))
        timing.clear()
        print("NUM OF PORTALS IS {}".format(len(self.portals)))


    def get_room_id(self, point):
        for room_id, bbox in enumerate(self.bboxes):
            if not inside_bbox(bbox, point):
                continue

            room = self.rooms[room_id]
            verts = (self.vertices[vid] - point for vid in room)
            if Geom2.is_origin_inside_polyline(verts):
                return room_id
        return None
