import math

from collections import defaultdict
from operator import itemgetter
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


def get_180_sector(verts, topo, spike_eid, extern_angles):
    # increase opening angle of the sector due to threshold
    extra_opening = .5 * extern_angles[spike_eid]
    cosine = math.cos(extra_opening)
    sine = math.sin(extra_opening)

    vid0, vid1 = topo.edge_verts(spike_eid)
    x, y = (verts[vid1] - verts[vid0]).normalized().comps
    x, y = (cosine*x - sine*y, sine*x + cosine*y)
    return vec(-x, -y), verts[vid1], vec(x, y)


def get_sector(verts, topo, spike_eid, extern_angles, accum_angles, threshold):
    '''
    Returns 2 vectors that define the sector rays, and the coordinates of the sector tip
    The vectors have unit lengths
    The vector pair is right-handed
    '''
    angle  = .5 * (extern_angles[spike_eid] - accum_angles[spike_eid] + threshold)
    angle_ = .5 * (extern_angles[spike_eid] + accum_angles[spike_eid] - threshold)

    sine = math.sin(angle)
    cosine = math.cos(angle)
    sine_ = math.sin(angle_)
    cosine_ = math.cos(angle_)

    vid1 = topo.target(spike_eid)
    vid2 = topo.target(topo.next_edge(spike_eid))
    x, y = (verts[vid1] - verts[vid2]).normalized().comps

    x1 = x*cosine + y*sine
    y1 = -x*sine + y*cosine
    x2 = -x*cosine_ - y*sine_
    y2 = x*sine_ - y*cosine_
    return vec(x1, y1), verts[vid1], vec(x2, y2)


'''
slower version of get_sector with 2 normalizations
'''
# def get_sector(verts, topo, spike_eid, extern_angles, accum_angles, threshold):
#     '''
#     Returns 2 vectors that define the sector rays, and the coordinates of the sector tip
#     The vectors have unit lengths
#     The vector pair is right-handed
#     '''
#     vid0, vid1 = topo.edge_verts(spike_eid)
#     vid2 = topo.target(topo.next_edge(spike_eid))
#     tip = verts[vid1]
#     vec1 = (tip - verts[vid2]).normalized()
#     vec2 = (tip - verts[vid0]).normalized()

#     # increase opening angle of the sector due to threshold
#     extra_opening = .5 * (extern_angles[spike_eid] - accum_angles[spike_eid] + threshold)

#     # rotate sector vectors to increase the opening angle
#     cosine = math.cos(extra_opening)
#     sine = math.sin(extra_opening)

#     v1x, v1y = vec1.comps
#     v1x, v1y = (cosine*v1x + sine*v1y, -sine*v1x + cosine*v1y)
#     v2x, v2y = vec2.comps
#     v2x, v2y = (cosine*v2x - sine*v2y, sine*v2x + cosine*v2y)
#     return vec(v1x, v1y), tip, vec(v2x, v2y)


def segment_sector_clip(segment, sector, sector_normalized=True):
    '''
    compute segment-sector overlap
    original order of endpoints is not guaranteed
    '''
    def align(pt, axis, flip_y):
        c, s = axis.comps
        x, y = pt.comps
        x, y = c*x + s*y, -s*x + c*y
        if flip_y: return x, -y
        return x, y

    seg0, seg1 = segment

    def intersect(tip, axis, flip_y):
        a = align(seg0 - tip, axis, flip_y)
        b = align(seg1 - tip, axis, flip_y)
        
        if a[1] < 0:
            if b[1] > 0:
                free_point = 1
                cross = a[0] * b[1] - a[1] * b[0]
                if cross > 0:
                    i_x = cross / (b[1] - a[1])
                    lower_x, upper_x = sorted((a[0], b[0]))
                    i_x = min(max(i_x, lower_x), upper_x)
                    if not sector_normalized: i_x /= axis.normSq()
                    i_point = tip + i_x * axis

                else:
                    i_point = None

            else: # b[1] <= 0
                return None, None

        elif a[1] == 0:
            if b[1] > 0:
                free_point = 1
                if a[0] > 0:
                    i_point = seg0
                else:
                    i_point = None

            else: # b[1] <= 0
                return None, None

        else: # a[1] > 0
            free_point = 0
            if b[1] < 0:
                cross = a[0] * b[1] - a[1] * b[0]
                if cross < 0:
                    i_x = cross / (b[1] - a[1])
                    lower_x, upper_x = sorted((a[0], b[0]))
                    i_x = min(max(i_x, lower_x), upper_x)
                    if not sector_normalized: i_x /= axis.normSq()
                    i_point = tip + i_x * axis

                else:
                    i_point = None

            else: #b[1] >= 0
                i_point = None

        return i_point, free_point

    dir1, tip, dir2 = sector
    inters1, free1 = intersect(tip, dir1, False)
    if inters1 is None and free1 is None:
        return None, None

    inters2, free2 = intersect(tip, dir2, True)
    if inters2 is None and free2 is None:
        return None, None

    if not inters1 and not inters2:
        if free1 == free2:
            return seg0, seg1 # do not clip
        else:
            return None, None

    elif inters1 and inters2:
        a, b = inters1, inters2 # clip both sides

    elif inters1:
        a, b = inters1, segment[free1] # clip one side

    else:
        a, b = segment[free2], inters2 # clip one side

    if a == b: # this can happen
        return None, None
    return a, b


def segment_closest_point_inside_sector(segment, sector):
    seg0, seg1 = segment_sector_clip(segment, sector)
    if seg1 is None:
        return None, None

    # find point on line closest to tip
    _, tip, _ = sector
    proj_param = Geom2.project_to_line(tip, (seg0, seg1 - seg0))
    if 0 < proj_param and proj_param < 1:
        points = (seg0, seg0 + proj_param*(seg1 - seg0), seg1)
    else:
        points = (seg0, seg1)

    distances = ((point, (tip - point).normSq()) for point in points)
    return min(distances, key=itemgetter(1))


def find_closest_edge_for_spike(verts, topo, sector, spike_eid, db_visitor):
    _, tip_vertex, _ = sector
    far_cutoff_plane = None
    closest_point, closest_distSq, closest_edge = None, None, None
    skip_edges = (spike_eid, topo.next_edge(spike_eid))
    eid_before, eid_after = (topo.prev_edge(spike_eid), topo.next_edge(skip_edges[1]))

    for eid in topo.iterate_room_edges(spike_eid):
        with timed_exec("inner routine"):
            if eid in skip_edges:
                continue

            seg0, seg1 = topo.edge_verts(eid)
            seg0, seg1 = verts[seg0], verts[seg1]
            if vec.cross2((seg0 - tip_vertex), (seg1 - tip_vertex)) >= 0:
                continue

            seg0x, seg1x = seg0.append(1), seg1.append(1)
            if far_cutoff_plane and seg0x.dot(far_cutoff_plane) > 0 and seg1x.dot(far_cutoff_plane) > 0:
                continue

            point, distSq = segment_closest_point_inside_sector((seg0, seg1), sector)
            if point is None: continue
            if eid == eid_before and point == verts[topo.target(eid_before)]: continue
            if eid == eid_after and point == verts[topo.target(skip_edges[1])]: continue

            if closest_distSq is None or distSq <= closest_distSq:
                closest_distSq = distSq
                closest_point = point
                closest_edge = eid

                # make cutoff line
                n = closest_point - tip_vertex
                far_cutoff_plane = n.append(-n.dot(closest_point))

    return closest_edge, closest_point


def check_occlusion(verts, topo, from_eid, to_eid, db_visitor):
    def align(pt, axis):
        c, s = axis.comps
        x, y = pt.comps
        x, y = c*x + s*y, -s*x + c*y
        return vec(x, y)

    vid0 = topo.target(from_eid)
    vid1 = topo.target(to_eid)
    from_point, to_point = verts[vid0], verts[vid1]

    closest_edge, x_point = None, None
    los = to_point - from_point
    closest_param = los.norm()
    los /= closest_param

    skip_edges = (from_eid, topo.next_edge(from_eid))
    eid_before, eid_after = topo.prev_edge(from_eid), topo.next_edge(skip_edges[1])

    for eid in topo.iterate_room_edges(from_eid):
        if eid in skip_edges: continue

        seg_vid0, seg_vid1 = topo.edge_verts(eid)
        if vid1 in (seg_vid0, seg_vid1):
            continue

        seg0, seg1 = verts[seg_vid0], verts[seg_vid1]
        if vec.cross2((seg0 - from_point), (seg1 - from_point)) >= 0:
            continue

        a = align(seg0-from_point, los)
        b = align(seg1-from_point, los)
        inters = None
        if a[1] >= 0:
            if b[1] < 0:
                cross = a[0] * b[1] - a[1] * b[0]
                if cross < 0:
                    inters = cross / (b[1] - a[1])

        elif b[1] >= 0:
            cross = a[0] * b[1] - a[1] * b[0]
            if cross > 0:
                inters = cross / (b[1] - a[1])

        if inters is None: continue

        if inters <= closest_param:
            point = from_point + inters * los
            if eid == eid_before and point == verts[topo.target(eid_before)]: continue
            if eid == eid_after and point == verts[topo.target(skip_edges[1])]: continue

            x_point = point
            closest_param = inters
            closest_edge = eid

    return closest_edge, x_point


def calc_external_angle(verts, topo, eid):
    vid0, vid1 = topo.edge_verts(eid)
    vid2 = topo.target(topo.next_edge(eid))

    prev_v = verts[vid0]
    this_v = verts[vid1]
    next_v = verts[vid2]
    dir0 = this_v - prev_v
    dir1 = next_v - this_v

    if vec.cross2(dir0, dir1) > 0:
        return math.acos(Geom2.cos_angle(dir0, dir1))
    else:
        return -math.acos(Geom2.cos_angle(dir0, dir1))


def calc_accum_angles(external_angles, topo, threshold):
    accum_angles = [None] * len(external_angles)
    for eid0 in topo.iterate_all_loops():
        loop_iter = iter(topo.iterate_loop_edges(eid0))
        tail = []
        for eid in loop_iter:
            tail.append(eid)
            if abs(external_angles[eid]) > threshold:
                break

        accum_angle = 0.
        for eid in chain(loop_iter, tail):
            accum_angle = max(accum_angle + external_angles[eid], 0.) # accumulate only positive values
            accum_angles[eid] = accum_angle
            if accum_angle > threshold:
                accum_angle = 0.
    return accum_angles


def calc_accum_angle(prev_accum, this_external, threshold):
    if prev_accum > threshold:
        return max(this_external, 0.)
    return max(prev_accum + this_external, 0.)


def delete_redundant_portals(topo, external_angles, accum_angles, threshold):
    deleted = [False] * topo.num_edges()

    def recalc_angles(eid):
        # get next counter-clockwise edge from eid skipping previously deleted edges
        eid_ccw = topo.prev_edge(topo.opposite(eid))
        while deleted[eid_ccw]:
            eid_ccw = topo.prev_edge(topo.opposite(eid_ccw))

        new_external_angle = math.pi + external_angles[eid] + external_angles[eid_ccw]
        eid_ccw_prev = topo.prev_edge(eid_ccw)
        while deleted[eid_ccw_prev]:
            eid_ccw_prev = topo.prev_edge(topo.opposite(eid_ccw_prev))

        accum_angle = calc_accum_angle(accum_angles[eid_ccw_prev], new_external_angle, threshold)
        if accum_angle > threshold:
            # cannot delete eid
            return False, eid_ccw, new_external_angle, []

        new_accum_angles = [(eid_ccw, accum_angle)]
        while True:
            eid = topo.next_edge(eid)
            while deleted[eid]:
                eid = topo.next_edge(topo.opposite(eid))

            accum_angle = calc_accum_angle(accum_angle, external_angles[eid], threshold)
            if accum_angle > threshold:
                # cannot delete eid
                return False, eid_ccw, new_external_angle, []

            if accum_angle == accum_angles[eid]:
                # we can delete eid
                return True, eid_ccw, new_external_angle, new_accum_angles

            new_accum_angles.append((eid, accum_angle))


    for eid in topo.iterate_all_internal_edges():
        if not topo.is_portal(eid): continue
        if deleted[eid]: continue

        can_delete, eid_ccw, new_extern, new_accum = recalc_angles(eid)
        if not can_delete: continue

        eid_ = topo.opposite(eid)
        can_delete, eid_ccw_, new_extern_, new_accum_ = recalc_angles(eid_)
        if not can_delete: continue

        deleted[eid] = deleted[eid_] = True
        external_angles[eid] = accum_angles[eid] = None
        external_angles[eid_] = accum_angles[eid_] = None

        external_angles[eid_ccw] = new_extern
        external_angles[eid_ccw_] = new_extern_
        for mod_eid, mod_accum_angle in chain(new_accum, new_accum_):
            accum_angles[mod_eid] = mod_accum_angle

    for eid, d in enumerate(deleted):
        if d and topo.edges[eid] is not None:
            topo.remove_edge(eid)


def update_angles_after_connecting(verts, topo, extern_angles, accum_angles, new_eid, threshold):
    extern_angles.extend((None, None))
    accum_angles.extend((None, None))

    def _impl(new_eid):
        prev_eid = topo.prev_edge(new_eid)

        extern_angles[prev_eid] = calc_external_angle(verts, topo, prev_eid)
        accum_angles[prev_eid] = calc_accum_angle(
            accum_angles[topo.prev_edge(prev_eid)], extern_angles[prev_eid], threshold)

        extern_angles[new_eid] = calc_external_angle(verts, topo, new_eid)
        accum_angles[new_eid] = calc_accum_angle(accum_angles[prev_eid], extern_angles[new_eid], threshold)

        accum_angle = accum_angles[new_eid]
        for eid in topo.iterate_loop_edges(topo.next_edge(new_eid)):
            if eid == topo.opposite(new_eid): break

            accum_angle = calc_accum_angle(accum_angle, extern_angles[eid], threshold)
            old_accum_angle = accum_angles[eid]
            accum_angles[eid] = accum_angle

            if accum_angle == old_accum_angle:
                break # no further change needed

            elif accum_angle > threshold:
                if old_accum_angle > threshold:
                    break # no further change needed
                else:
                    accum_angle = 0.

    _impl(new_eid)
    _impl(topo.opposite(new_eid))


def convex_subdiv(verts, topo, threshold, db_visitor=None):
    """
    This function uses algorithm from R. Oliva and N. Pelechano - 
    Automatic Generation of Suboptimal NavMeshes
    """
    extern_angles = [None] * topo.num_edges()
    for eid in topo.iterate_all_internal_edges():
        extern_angles[eid] = calc_external_angle(verts, topo, eid)

    accum_angles = calc_accum_angles(extern_angles, topo, threshold)

    def connect_to_target(start_eid, edge):
        new_eid = topo.connect(start_eid, edge, verts)
        update_angles_after_connecting(verts, topo, extern_angles, accum_angles, new_eid, threshold)


    def connect_to_exterior_wall(start_eid, edge, point):
        v0, v1 = topo.edge_verts(edge)
        v0, v1 = verts[v0], verts[v1]
        if point == v0:
            connect_to_target(start_eid, topo.prev_edge(edge))
        elif point == v1:
            connect_to_target(start_eid, edge)
        else:
            new_eid = topo.insert_vertex(len(verts), edge)

            # if db_visitor:
            #     db_visitor.add_text(point, str(topo.target(new_eid)), color="cyan")

            assert topo.next_edge(new_eid) == edge
            assert topo.prev_edge(edge) == new_eid
            assert topo.room_id(new_eid) is not None
            assert topo.room_id(edge) is not None

            # add angles for new_eid
            extern_angles.extend((None, None))
            accum_angles.extend((None, None))
            extern_angles[new_eid] = 0.
            accum_angles[new_eid] = calc_accum_angle(accum_angles[topo.prev_edge(new_eid)], 0., threshold)

            assert len(verts) == topo.target(new_eid), "something went wrong here"

            verts.append(point)
            conn_eid = topo.connect(start_eid, new_eid, verts)
            update_angles_after_connecting(verts, topo, extern_angles, accum_angles, conn_eid, threshold)


    def connect_to_closest_endpoint_if_los_is_free(from_eid, target_eid):
        vid0, vid1 = topo.edge_verts(target_eid)
        start_vert = verts[topo.target(from_eid)]
        if (verts[vid0] - start_vert).normSq() < (verts[vid1] - start_vert).normSq():
            if topo.prev_edge(target_eid) == topo.next_edge(from_eid):
                return
            return connect_to_target_if_los_is_free(from_eid,  topo.prev_edge(target_eid))
        else:
            if target_eid == topo.prev_edge(from_eid):
                return
            return connect_to_target_if_los_is_free(from_eid, target_eid)


    def connect_to_target_if_los_is_free(start_eid, portal_eid):
        with timed_exec("LOS"):
            occluding_eid, point = check_occlusion(verts, topo, start_eid, portal_eid, db_visitor)
        if occluding_eid is None or occluding_eid == portal_eid:
            return connect_to_target(start_eid, portal_eid)

        if topo.is_portal(occluding_eid):
            return connect_to_closest_endpoint_if_los_is_free(start_eid, occluding_eid)
        else:
            return connect_to_exterior_wall(start_eid, occluding_eid, point)


    # if db_visitor:
    #     for eid in tuple(topo.iterate_all_internal_edges()):
    #         vid = topo.target(eid)
    #         db_visitor.add_text(verts[vid], str(vid), color="#93f68b")

    # sorting spikes by external angle makes it faster!
    all_edges = list(sorted(tuple(topo.iterate_all_internal_edges()),
        key=lambda eid: -accum_angles[eid]))

    for spike_eid in all_edges:
        if accum_angles[spike_eid] <= threshold: continue # not a spike

        # if db_visitor:
        #     vid = topo.target(spike_eid)
        #     db_visitor.add_text(verts[vid], str(vid), color="#93f68b")

        # special case for almost 180 degree sectors
        if accum_angles[spike_eid] <= threshold + 1e-6:
            sector = get_180_sector(verts, topo, spike_eid, extern_angles)
        else:
            sector = get_sector(verts, topo, spike_eid, extern_angles, accum_angles, threshold)

        # if db_visitor:
        #     d1, t, d2 = sector
        #     db_visitor.add_polygon((t+20*d1, t), color="green")
        #     db_visitor.add_polygon((t+20*d2, t), color="red")

        closest_edge, closest_point = find_closest_edge_for_spike(verts, topo, sector, spike_eid, db_visitor)
        if closest_edge is None:
            if db_visitor:
                v0, v1 = topo.edge_verts(spike_eid)
                v2 = topo.target(topo.next_edge(spike_eid))
                d1, t, d2 = sector
                db_visitor.add_polygon((t+200*d1, t, t+200*d2, t), color="gold")
                db_visitor.add_text(verts[v0], str(v0), color="gold")
                db_visitor.add_text(verts[v1], str(v1), color="gold")
                db_visitor.add_text(verts[v2], str(v2), color="gold")
            nxe = topo.next_edge(spike_eid)
            raise RuntimeError("Could not find a single edge for spike {} -> {}".format(
                topo.debug_repr(spike_eid), topo.debug_repr(nxe)))

        # if closest edge is a portal
        if topo.is_portal(closest_edge):
            _, tip, _ = sector

            v0, v1 = topo.edge_verts(closest_edge)
            v0, v1 = verts[v0], verts[v1]
            if closest_point == v0:
                connect_to_target(spike_eid, topo.prev_edge(closest_edge))
            elif closest_point == v1:
                connect_to_target(spike_eid, closest_edge)

            # If closest point is not one of the endpoints,
            # we still create the portal(s) to one or two endpoints:
            else:
                def pt_inside_sector(sector, pt):
                    dir1, tip, dir2 = sector
                    rel_pt = pt - tip
                    return vec.cross2(dir1, rel_pt) > 0 and vec.cross2(dir2, rel_pt) < 0

                start_inside = pt_inside_sector(sector, v0)
                end_inside = pt_inside_sector(sector, v1)

                # if none of the portal endpoints is inside sector, create 2 portals to both ends:
                if not start_inside and not end_inside:
                    connect_to_target_if_los_is_free(spike_eid, topo.prev_edge(closest_edge)),
                    connect_to_target_if_los_is_free(spike_eid, closest_edge)

                elif start_inside:
                    connect_to_target_if_los_is_free(spike_eid, topo.prev_edge(closest_edge))

                else:
                    connect_to_target_if_los_is_free(spike_eid, closest_edge)

                # if the closest portal was occluded by other portals,
                # there is no guarantee that the spike was removed 
                if accum_angles[spike_eid] > threshold:
                    all_edges.append(spike_eid)

        else:
            connect_to_exterior_wall(spike_eid, closest_edge, closest_point)

    delete_redundant_portals(topo, extern_angles, accum_angles, threshold)


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
        return "e{}, {}->{} @ room {}".format(eid, vid0, vid1, self.edges[eid].room_id)

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

    def connect(self, e0, e1, verts):
        def _change_topology():
            vid0 = self.target(e0)
            vid1 = self.target(e1)

            enxt0 = self.edges[e0].next
            enxt1 = self.edges[e1].next

            new_eid = len(self.edges)
            new_eid_oppo = new_eid + 1
            self.edges.append(EdgeStruct(prev=e0, next=enxt1, target=vid1, room_id=None))
            self.edges.append(EdgeStruct(prev=e1, next=enxt0, target=vid0, room_id=None))

            self.edges[e0].next = new_eid
            self.edges[enxt0].prev = new_eid_oppo
            self.edges[e1].next = new_eid_oppo
            self.edges[enxt1].prev = new_eid
            return new_eid, new_eid_oppo

        assert self.room_id(e0) is not None
        assert self.room_id(e1) is not None
        assert self.room_id(e0) == self.room_id(e1)

        old_room_id = self.edges[e0].room_id

        # splitting one loop into two, therefore creating a new room
        if self.same_loop(e0, e1):

            # splitting the outline
            if self.same_loop(e0, self.get_outline(e0)):
                unsorted_holes = self.rooms[old_room_id].holes
                old_outline, new_outline = _change_topology()
                old_loops = [old_outline]
                new_loops = [new_outline]


            # splitting one of holes
            else:
                # find out which hole we are splitting and remove it from the room
                unsorted_holes = [hole_eid for hole_eid in self.rooms[old_room_id].holes if not self.same_loop(hole_eid, e0)]
                modified_hole, new_outline = _change_topology()
                if not self.loop_is_ccw(modified_hole, verts):
                    modified_hole, new_outline = new_outline, modified_hole

                old_loops = [self.rooms[old_room_id].outline, modified_hole]
                new_loops = [new_outline]

            # sort holes
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
            return new_loops[0]

        # joining two loops into one, therefore no new room is created
        else:
            remaining_holes = []
            for hole_eid in self.rooms[old_room_id].holes:
                if not self.same_loop(hole_eid, e0) and not self.same_loop(hole_eid, e1):
                    remaining_holes.append(hole_eid)

            e0, e1 = _change_topology()
            outline = self.rooms[old_room_id].outline
            if not self.same_loop(outline, e0):
                remaining_holes.append(e0)

            self.edges[e0].room_id = self.edges[e1].room_id = old_room_id
            self.rooms[old_room_id] = Room(outline=outline, holes=remaining_holes)
            return e0
 

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
            print("{}, {} times, average time {}".format(name, t.n, t.average()*1000))
        timing.clear()


    def get_room_id(self, point):
        for room_id, bbox in enumerate(self.bboxes):
            if not inside_bbox(bbox, point):
                continue

            room = self.rooms[room_id]
            verts = (self.vertices[vid] - point for vid in room)
            if Geom2.is_origin_inside_polyline(verts):
                return room_id
        return None
