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


def get_sector(prev_v, spike_v, next_v, threshold):
    '''
    Returns 2 vectors that define the sector rays.
    The vectors have unit lengths
    The vector pair is right-hand
    '''
    vec1 = (spike_v - prev_v).normalized()
    vec2 = (spike_v - next_v).normalized()

    cosine = Geom2.cos_angle(vec1, vec2)
    sine = Geom2.sin_angle(vec1, vec2)

    sector_angle = 2 * math.pi - math.acos(cosine) if sine < 0 else math.acos(cosine)
    sector_angle_plus = sector_angle + threshold
    
    # limit sector opening to 180 degrees:
    sector_angle_plus = min(sector_angle_plus, math.pi)

    clearance = .5 * (sector_angle_plus - sector_angle)

    cosine = math.cos(clearance)
    sine = math.sin(clearance)

    # rotate sector vectors to increase angle for clearance
    return (
        Geom2.mul_mtx_2x2((cosine, sine, -sine, cosine), vec1),
        Geom2.mul_mtx_2x2((cosine, -sine, sine, cosine), vec2)
    )



def _segment_closest_point_inside_sector(segment, sector):
    dir1, tip, dir2 = sector

    def pt_inside(pt):
        rel_pt = pt - tip
        return vec.cross2(dir1, rel_pt) > 0 and vec.cross2(dir2, rel_pt) < 0

    # make ray-like line
    line = (segment[0], segment[1] - segment[0])
    (linepara1, raypara1, _) = Geom2.lines_intersect(line, (tip, dir1))
    (linepara2, raypara2, _) = Geom2.lines_intersect(line, (tip, dir2))

    # check whether line containing segment passes through sector:
    line_through_sector = raypara1 > 0 or raypara2 > 0

    if not line_through_sector:
        return None, None, None

    # now we know that line goes through sector
    # find intersections between segment and sector rays
    intersect_params = []
    if raypara1 > 0 and 0 < linepara1 and linepara1 < 1:
        intersect_params.append(linepara1)

    if raypara2 > 0 and 0 < linepara2 and linepara2 < 1:
        intersect_params.append(linepara2)

    # one intersection => one endpoint inside, other outside
    if len(intersect_params) == 1:
        if pt_inside(segment[0]):
            candid_params = [0, intersect_params[0]]
        else:
            candid_params = [intersect_params[0], 1]
    # two intersections => middle section passes through sector
    elif len(intersect_params) == 2:
        candid_params = list(sorted(intersect_params))

    # no intersections => entirely inside or outside
    else:
        if pt_inside(segment[0]):
            candid_params = [0, 1]
        else:
            return None, None, None

    # find point on line closest to tip
    projpara = Geom2.project_to_line(tip, line)

    if candid_params[0] < projpara and projpara < candid_params[1]:
        candid_params.append(projpara)

    def para_to_pt(para):
        return line[0] + (para * line[1])

    distances = ((tip - para_to_pt(para)).normSq() for para in candid_params)

    p, d = min(zip(candid_params, distances), key = itemgetter(1))
    return p, para_to_pt(p), d


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
        return math.acos(Geom2.cos_angle(dir0, dir1)) * (-1.)


def find_spikes(topo, extern_angles, threshold):
    spikes = []
    accum_angle = 0.

    for eid in topo.iterate_all_internal_edges():
        accum_angle = max(0., accum_angle + extern_angles[eid])
        if accum_angle > threshold:
            spikes.append(topo.target(eid))
            accum_angle = 0.
    return spikes


def calc_accumulated_extern_angle(topo, extern_angles, eid):
    angle = extern_angles[eid]
    accumulated = max(0, angle)

    eid0 = eid
    eid = topo.prev_edge(eid)
    while angle > 0. and eid != eid0:
        angle = extern_angles[eid]
        accumulated += max(0, angle)
        eid = topo.prev_edge(eid)
    return accumulated


def find_closest_edge_for_spike(verts, topo, extern_angles, vertex_idx, threshold):
    inbound_edges = topo.edges_around_vertex(vertex_idx)
    border_edge = next(inbound_edges) # skip the external edge

    for eid0 in inbound_edges:

        extern_angle = calc_accumulated_extern_angle(topo, extern_angles, eid0)
        if extern_angle <= threshold: continue # not a spike

        vid0, vid1 = topo.edge_verts(eid0)
        vid2 = topo.target(topo.next_edge(eid0))
        tip_vertex = verts[vid1]

        right_dir, left_dir = get_sector(verts[vid2], tip_vertex, verts[vid0], threshold)
        sector = (right_dir, tip_vertex, left_dir)

        far_cutoff_plane = None
        closest_para, closest_distSq, closest_edge = None, None, None

        for eid in topo.iterate_room_edges(eid0):
            with timed_exec("inner routine"):
                seg_i1, seg_i2 = topo.edge_verts(eid)

                if vertex_idx in (seg_i1, seg_i2):
                    continue

                seg0, seg1 = verts[seg_i1], verts[seg_i2]
                seg0x, seg1x = seg0.append(1), seg1.append(1)

                if far_cutoff_plane and seg0x.dot(far_cutoff_plane) > 0 and seg1x.dot(far_cutoff_plane) > 0:
                    continue

                para, point, distSq = _segment_closest_point_inside_sector((seg0, seg1), sector)

                if para is None: continue
                if closest_distSq is None or distSq < closest_distSq:
                    closest_distSq = distSq
                    closest_para = para
                    closest_edge = eid

                    # make cutoff line
                    n = point - tip_vertex
                    far_cutoff_plane = n.append(-n.dot(point))

        if closest_edge is None:
            raise RuntimeError("Could not find a single edge inside the sector ({}, {}, {})".format(vid0, vid1, vid2))

        return ((eid0, sector, closest_edge, closest_para),) # there can be at most 1 spike at a vertex

    return ()


def choose_closer_side_of_edge(topo, verts, seen_from_point, eid, param):
    if topo.room_id(eid) == topo.room_id(topo.opposite(eid)):
        vid0, vid1 = topo.edge_verts(eid)
        v0, v1 = verts[vid0], verts[vid1]

        if vec.cross2(v0 - seen_from_point, v1 - seen_from_point) > 0:
            return topo.opposite(eid), 1 - param

    return eid, param


def create_portals(verts, topo, threshold, db_visitor=None):
    """
    This function uses algorithm from R. Oliva and N. Pelechano - 
    Automatic Generation of Suboptimal NavMeshes
    This is work in progress
    The endpoints of portals can be new vertices,
    but they are guaranteed to lie on the polygon boundary (not inside the polygon)
    """

    extern_angles = [None] * topo.num_edges()
    for eid in topo.iterate_all_internal_edges():
        extern_angles[eid] = calc_external_angle(verts, topo, eid)


    def check_line_of_sight(from_eid, to_eid):
        vid0 = topo.target(from_eid)
        vid1 = topo.target(to_eid)
        from_point, to_point = verts[vid0], verts[vid1]
        n = to_point - from_point
        near_cutoff = n.append(-n.dot(from_point))
        far_cutoff = (-n).append(n.dot(to_point))

        closest_param = 1.
        closest_edge, closest_edge_param = None, None
        for eid in topo.iterate_room_edges(from_eid):
            seg_vid0, seg_vid1 = topo.edge_verts(eid)

            if vid0 in (seg_vid0, seg_vid1) or vid1 in (seg_vid0, seg_vid1):
                continue

            seg0, seg1 = verts[seg_vid0], verts[seg_vid1]
            seg0x, seg1x = seg0.append(1), seg1.append(1)

            if \
                (seg0x.dot(near_cutoff) < 0 and seg1x.dot(near_cutoff) < 0) or \
                (seg0x.dot( far_cutoff) < 0 and seg1x.dot( far_cutoff) < 0):
                continue

            coef0, coef1, dist = Geom2.lines_intersect((from_point, n), (seg0, seg1 - seg0))

            if math.isnan(coef0) and dist < 1e-20:
                d0 = (seg0 - from_point).normSq()
                d1 = (seg1 - from_point).normSq()
                coef0 = math.sqrt(min(d0, d1)) / n.norm()

                if coef0 < closest_param:
                    closest_param = coef0
                    closest_edge = eid
                    if d0 < d1:
                        closest_edge_param = 0
                        x_point = seg0
                    else:
                        closest_edge_param = 1
                        x_point = seg1
                    far_cutoff = (-n).append(n.dot(x_point))


            elif 0 < coef1 and coef1 < 1 and 0 < coef0 and coef0 < closest_param:
                closest_param = coef0
                closest_edge = eid
                closest_edge_param = coef1
                x_point = from_point + coef0 * n
                far_cutoff = (-n).append(n.dot(x_point))

        # choose correct side of edge
        if closest_edge is not None:
            closest_edge, closest_edge_param = choose_closer_side_of_edge(topo, verts, from_point, closest_edge, closest_edge_param)

        return closest_edge, closest_edge_param


    def update_angles(eid):
        eid_ = topo.opposite(eid)
        extern_angles.extend((None, None))
        extern_angles[eid] = calc_external_angle(verts, topo, eid)
        extern_angles[eid_] = calc_external_angle(verts, topo, eid_)
        eprev = topo.prev_edge(eid)
        eprev_ = topo.prev_edge(eid_)
        extern_angles[eprev] = calc_external_angle(verts, topo, eprev)
        extern_angles[eprev_] = calc_external_angle(verts, topo, eprev_)

    def connect_to_startpoint(start_eid, edge):
        update_angles(topo.connect(start_eid, topo.prev_edge(edge), verts))


    def connect_to_endpoint(start_eid, edge):
        update_angles(topo.connect(start_eid, edge, verts))


    def connect_to_exterior_wall(start_eid, edge, param):
        seg0, seg1 = topo.edge_verts(edge)

        if param == 0:
            connect_to_startpoint(start_eid, edge)
        elif param == 1:
            connect_to_endpoint(start_eid, edge)
        else:
            v0, v1 = verts[seg0], verts[seg1]
            new_vert = v0 + param * (v1 - v0)
            new_eid = topo.insert_vertex(edge)

            # update external angles
            extern_angles.extend((None, None))
            extern_angles[topo.opposite(new_eid)] = extern_angles[edge]
            extern_angles[edge] = 0.

            new_vid = topo.target(new_eid)
            assert len(verts) == new_vid, "something went wrong here"

            verts.append(new_vert)

            new_eid_internal = topo.prev_edge(topo.opposite(new_eid))
            update_angles(topo.connect(start_eid, new_eid_internal, verts))


    def connect_to_closest_portal_endpoint(from_eid, target_eid):
        vid0, vid1 = topo.edge_verts(target_eid)
        start_vert = verts[topo.target(from_eid)]
        if (verts[vid0] - start_vert).normSq() < (verts[vid1] - start_vert).normSq():
            return portal_to_portal_startpoint_if_los_is_free(from_eid, target_eid)
        else:
            return portal_to_portal_endpoint_if_los_is_free(from_eid, target_eid)


    def portal_to_portal_startpoint_if_los_is_free(start_eid, portal_eid):
        with timed_exec("LOS"):
            closest_eid, param = check_line_of_sight(start_eid, topo.prev_edge(portal_eid))
        if closest_eid is None or closest_eid == portal_eid:
            return connect_to_startpoint(start_eid, portal_eid)

        if topo.is_portal(closest_eid):
            return connect_to_closest_portal_endpoint(start_eid, closest_eid)
        else:
            return connect_to_exterior_wall(start_eid, closest_eid, param)


    def portal_to_portal_endpoint_if_los_is_free(start_eid, portal_eid):
        with timed_exec("LOS"):
            closest_eid, param = check_line_of_sight(start_eid, portal_eid)
        if closest_eid is None or closest_eid == portal_eid:
            return connect_to_endpoint(start_eid, portal_eid)

        if topo.is_portal(closest_eid):
            return connect_to_closest_portal_endpoint(start_eid, closest_eid)
        else:
            return connect_to_exterior_wall(start_eid, closest_eid, param)


    if db_visitor:
        for eid in topo.iterate_all_internal_edges():
            vid = topo.target(eid)
            db_visitor.add_text(verts[vid], str(vid), color="#93f68b")


    spikes = find_spikes(topo, extern_angles, threshold)

    for spike_idx in spikes:
        closest_edge_info = find_closest_edge_for_spike(verts, topo, extern_angles, spike_idx, threshold)

        if not closest_edge_info: continue
        closest_edge_info, = closest_edge_info
        eid, sector, closest_edge, closest_param = closest_edge_info

        # if closest edge is a portal
        if topo.is_portal(closest_edge):
            _, tip, _ = sector

            # choose the correct side of the portal
            closest_edge, closest_param = choose_closer_side_of_edge(topo, verts, tip, closest_edge, closest_param)

            if closest_param == 0:
                connect_to_startpoint(eid, closest_edge)
            elif closest_param == 1:
                connect_to_endpoint(eid, closest_edge)

            # If closest point is not one of the endpoints,
            # we still create the portal(s) to one or two endpoints:
            else:
                def pt_inside_sector(sector, pt):
                    dir1, tip, dir2 = sector
                    rel_pt = pt - tip
                    return vec.cross2(dir1, rel_pt) > 0 and vec.cross2(dir2, rel_pt) < 0

                other_start_pt, other_end_pt = topo.edge_verts(closest_edge)
                other_start_pt, other_end_pt = verts[other_start_pt], verts[other_end_pt]

                start_inside = pt_inside_sector(sector, other_start_pt)
                end_inside = pt_inside_sector(sector, other_end_pt)

                # if none of the portal endpoints is inside sector, create 2 portals to both ends:
                if not start_inside and not end_inside:
                        portal_to_portal_startpoint_if_los_is_free(eid, closest_edge),
                        portal_to_portal_endpoint_if_los_is_free(eid, closest_edge)

                elif start_inside:
                    portal_to_portal_startpoint_if_los_is_free(eid, closest_edge)

                else:
                    portal_to_portal_endpoint_if_los_is_free(eid, closest_edge)

        # if no portals are in the way, attach to an edge/vertex on the exterior wall
        else:
            connect_to_exterior_wall(eid, closest_edge, closest_param)

    return extern_angles


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
        return "{}->{} @ room {}".format(vid0, vid1, self.edges[eid].room_id)

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

    def inbound(self, vid):
        '''
        Get eid of an inbound half-edge for the given vertex.
        The returned half-edge is always from a border loop.
        '''
        return self.inbound_edges[vid]

    def room_id(self, eid):
        return self.edges[eid].room_id

    def is_portal(self, eid):
        return self.edges[eid].room_id is not None and \
               self.edges[self.opposite(eid)].room_id is not None

    def _iterate_loop_edges(self, eid0):
        yield eid0
        eid = self.edges[eid0].next
        while eid != eid0:
            yield eid
            eid = self.edges[eid].next

    def iterate_all_internal_edges(self):
        edge_ids = list(chain(*([room.outline] + room.holes for room in self.rooms if room is not None)))
        return chain(*(self._iterate_loop_edges(eid) for eid in edge_ids))

    def iterate_room_edges(self, eid):
        room = self.rooms[self.edges[eid].room_id]
        edge_ids = [room.outline] + room.holes
        return chain(*(self._iterate_loop_edges(eid) for eid in edge_ids))

    def edges_around_vertex(self, vid):
        eid0 = self.inbound_edges[vid]
        yield eid0
        eid = self.edges[self.opposite(eid0)].prev
        while eid != eid0:
            yield eid
            eid = self.edges[self.opposite(eid)].prev


    def insert_vertex(self, eid):
        eid_ = self.opposite(eid)
        if self.edges[eid].room_id is not None:
            eid, eid_ = eid_, eid

        new_vid = len(self.inbound_edges)

        eprv = self.edges[eid].prev
        enxt_ = self.edges[eid_].next

        eplus  = EdgeStruct(prev=eprv, next=eid, target=new_vid, room_id=None)
        eplus_ = EdgeStruct(prev=eid_, next=enxt_, target=self.edges[eid_].target, room_id=self.edges[eid_].room_id)

        idxof_eplus = len(self.edges)
        idxof_eplus_ = len(self.edges) + 1
        self.edges.append(eplus)
        self.edges.append(eplus_)

        self.edges[eprv].next = idxof_eplus
        self.edges[enxt_].prev = idxof_eplus_
        self.edges[eid].prev = idxof_eplus
        self.edges[eid_].next = idxof_eplus_
        self.edges[eid_].target = new_vid

        self.inbound_edges.append(idxof_eplus)

        assert self.room_id(eid) is None
        assert self.room_id(eid_) is not None
        assert self.room_id(idxof_eplus) is None
        assert self.room_id(idxof_eplus_) is not None

        assert self.room_id(eprv) is None
        assert self.room_id(enxt_) is not None

        return idxof_eplus


    def same_loop(self, eid0, eid1):
        for eid in self._iterate_loop_edges(eid0):
            if eid == eid1:
                return True
        return False

    def inside_loop(self, query_eid, loop_eid, verts):
        query_pt = verts[self.edges[query_eid].target]
        vert_ids = (self.edges[eid].target for eid in self._iterate_loop_edges(loop_eid))
        polyline = (verts[vid] - query_pt for vid in vert_ids)
        return Geom2.is_origin_inside_polyline(polyline)

    def loop_is_ccw(self, loop_eid, verts):
        vert_ids = (self.edges[eid].target for eid in self._iterate_loop_edges(loop_eid))
        points = tuple(verts[vid] for vid in vert_ids)
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
                for eid in self._iterate_loop_edges(loop):
                    self.edges[eid].room_id = old_room_id

            new_room_id = len(self.rooms)
            for loop in new_loops:
                for eid in self._iterate_loop_edges(loop):
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

        # pick other inbound edges for 2 adjacent verts if necessary
        vid = self.target(eid)
        vid_ = self.target(eid_)
        if self.inbound_edges[vid] == eid:
            self.inbound_edges[vid] = self.prev_edge(eid_)
        if self.inbound_edges[vid_] == eid_:
            self.inbound_edges[vid_] = self.prev_edge(eid)


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
        t.inbound_edges = [None for _ in poly.graph.all_nodes_iterator()]
        loops = []

        for loop_start in poly.graph.loops:
            loop_verts = poly.graph.loop_iterator(loop_start)
            first_eid = len(t.edges)
            loops.append(first_eid + 1)

            src0 = next(loop_verts)
            tgt0 = poly.graph.next[src0]
            t.inbound_edges[tgt0] = len(t.edges)
            t.edges.append(EdgeStruct(target=tgt0, room_id=None, prev=None, next=None))
            t.edges.append(EdgeStruct(target=src0, room_id=0, prev=None, next=None))

            for src in loop_verts:
                tgt = poly.graph.next[src]

                t.edges[-2].next = len(t.edges)
                t.edges[-1].prev = len(t.edges) + 1
                t.inbound_edges[tgt] = len(t.edges)
                t.inbound_edges[src] = len(t.edges) - 2
                t.edges.append(EdgeStruct(target=tgt, room_id=None, prev=len(t.edges)-2, next=None))
                t.edges.append(EdgeStruct(target=src, room_id=0, prev=None, next=len(t.edges)-2))


            t.inbound_edges[src0] = len(t.edges) - 2
            t.edges[-2].next = first_eid
            t.edges[-1].prev = first_eid + 1
            t.edges[first_eid].prev = len(t.edges) - 2
            t.edges[first_eid + 1].next = len(t.edges) - 1

        t.rooms = [Room(outline=loops[0], holes=loops[1:])]
        return t


def delete_redundant_portals(topo, external_angles, convex_relax_thresh):
    deleted = [False] * topo.num_edges()

    def calc_extern_angle_ignoring_deleted(eid):
        eid_adj = topo.prev_edge(topo.opposite(eid))
        angle = external_angles[eid_adj]

        while deleted[eid_adj]:
            eid_adj = topo.prev_edge(topo.opposite(eid_adj))
            angle += math.pi + external_angles[eid_adj]

        angle += math.pi + external_angles[eid]
        eid_adj = topo.opposite(topo.next_edge(eid))
        while deleted[eid_adj]:
            angle += math.pi + external_angles[eid_adj]
            eid_adj = topo.opposite(topo.next_edge(eid_adj))

        return angle


    for eid in topo.iterate_all_internal_edges():
        if not topo.is_portal(eid): continue
        if deleted[eid]: continue

        eid_ = topo.opposite(eid)
        angle0 = calc_extern_angle_ignoring_deleted(eid)
        angle1 = calc_extern_angle_ignoring_deleted(eid_)

        if angle0 <= convex_relax_thresh and angle1 <= convex_relax_thresh:
            deleted[eid] = deleted[eid_] = True

    for eid, d in enumerate(deleted):
        if d and topo.edges[eid] is not None:
            topo.remove_edge(eid)


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

        with timed_exec("create_portals"):
            external_angles = create_portals(self.vertices, topo, convex_relax_thresh, db_visitor)

        delete_redundant_portals(topo, external_angles, convex_relax_thresh)

        self.portals = list()
        for eid in topo.iterate_all_internal_edges():
            if topo.is_portal(eid):
                vid0, vid1 = topo.edge_verts(eid)
                if vid0 < vid1:
                    self.portals.append((vid0, vid1))

        # make rooms (a room is just a list of vertex ids that form a nearly convex polygon)
        self.rooms = []
        self.bboxes = []
        self.adjacent_rooms = []
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


    def get_room_id(self, point):
        for room_id, bbox in enumerate(self.bboxes):
            if not inside_bbox(bbox, point):
                continue

            room = self.rooms[room_id]
            verts = (self.vertices[vid] - point for vid in room)
            if Geom2.is_origin_inside_polyline(verts):
                return room_id
        return None
