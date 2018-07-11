import math

from collections import defaultdict
from operator import itemgetter
from itertools import chain, izip, repeat, tee
from copy import deepcopy
from time import clock
from contextlib import contextmanager

from .vector2 import vec, Geom2
from .utils import pairs, triples

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





class Loops(object):
    def __init__(self):
        self.loops = []
        self.next = []
        self.prev = []


    def add_loop(self, how_many_nodes):
        loop_start = len(self.next)
        ids = range(loop_start, loop_start + how_many_nodes)

        self.loops.append(loop_start)
        self.next.extend(ids[1:] + ids[:1])
        self.prev.extend(ids[-1:] + ids[:-1])
        return loop_start


    def loop_iterator(self, loop_start):
        yield loop_start
        idx = self.next[loop_start]
        while idx != loop_start:
            yield idx
            idx = self.next[idx]


    def all_nodes_iterator(self):
        return chain(*(self.loop_iterator(loop) for loop in self.loops))


    def insert_node(self, edge_to_split):
        new_idx = len(self.next)
        e0, e1 = edge_to_split

        self.next[e0] = self.prev[e1] = new_idx
        self.next.append(e1)
        self.prev.append(e0)
        return new_idx



def is_origin_inside_polyline(polyline):
    # ray from origin along positive x axis
    x_ray = (vec(0, 0), vec(1, 0))
    num_inters = 0
    # iterate over pairs of vertices
    for curr_v, next_v in pairs(polyline):
        if curr_v[1] == 0: curr_v += vec(0, 1e-8)
        if next_v[1] == 0: next_v += vec(0, 1e-8)

        if curr_v[1] * next_v[1] < 0:
            _, b, _ = Geom2.lines_intersect((curr_v, next_v - curr_v), x_ray)
            if b > 0: num_inters += 1

    return num_inters % 2 > 0



class Polygon2d(object):
    def __init__(self, vertices):
        # ensure CCW order - outline must be CCW
        if Geom2.poly_signed_area(vertices) > 0:
            self.vertices = list(vertices)
        else:
            self.vertices = list(vertices[::-1])

        self.graph = Loops()
        self.graph.add_loop(len(self.vertices))



    def add_hole(self, vertices):
        # ensure CW order - holes must be CW
        if Geom2.poly_signed_area(vertices) < 0:
            self.vertices.extend(vertices)
        else:
            self.vertices.extend(vertices[::-1])

        self.graph.add_loop(len(vertices))



    def point_inside_loop(self, point, loop_start):
        # transform vertices so that query point is at the origin, append start vertex at end to wrap
        verts = (self.vertices[idx] - point for idx in \
            chain(self.graph.loop_iterator(loop_start), [loop_start]))
        return is_origin_inside_polyline(verts)
        


    def _get_segment_helper(self, seg_ids):
        return tuple(self.vertices[idx] for idx in seg_ids)


    def _segments_cross_helper(self, seg1, seg2):
        seg1 = self._get_segment_helper(seg1)
        seg2 = self._get_segment_helper(seg2)

        # if segment endpoints match, ignore intersection
        if seg1[0] in seg2 or seg1[1] in seg2:
            return None, None

        line1 = (seg1[0], seg1[1] - seg1[0])
        line2 = (seg2[0], seg2[1] - seg2[0])

        a, b, _ = Geom2.lines_intersect(line1, line2)
        if 0 < a and a < 1 and 0 < b and b < 1:
            return a, b
        else:
            return None, None



    def insert_vertex(self, vertex, edge_to_split):
        new_idx = self.graph.insert_node(edge_to_split)
        assert new_idx == len(self.vertices)
        self.vertices.append(vertex)
        return new_idx



    def add_vertices_to_border(self, edge, vertex_params):
        '''
        Add given list of vertices to the given edge of the polygon border.
        The exact loop that contains the edge is determined automatically.
        This function returns a list of new indices for the list of vertices in the same order.
        '''
        new_ids = [None] * len(vertex_params)
        ascend_params = sorted(enumerate(vertex_params), key = itemgetter(1))

        e0, e1 = edge
        ray = (self.vertices[e0], self.vertices[e1] - self.vertices[e0])
        idx = e0
        for (position_before_sorted, param) in ascend_params:
            idx = self.insert_vertex(ray[0] + param * ray[1], (idx, e1))
            new_ids[position_before_sorted] = idx

        return new_ids


    def point_inside(self, point):
        # first check if inside outline
        if not self.point_inside_loop(point, self.graph.loops[0]):
            return False

        # now check if inside a hole
        for hole in self.graph.loops[1:]:
            if self.point_inside_loop(point, hole):
                return False

        return True


    @staticmethod
    def check_convex(indices, vertices):
        indices_wrap = chain(indices, indices[:2])

        for idx0, idx1, idx2 in triples(indices_wrap):
            if Geom2.signed_area(
                vertices[idx0],
                vertices[idx1],
                vertices[idx2]) < 0: return False

        return True




class Portal(object):
    class ToSegment: pass
    class ToVertex: pass
    class ToPortal: pass

    def calc_endpoint(self, vertices):
        if self.kind == Portal.ToVertex:
            return vertices[self.end_info]

        elif self.kind == Portal.ToSegment:
            edge, para = self.end_info
            idx1, idx2 = edge
            return vertices[idx1] + (para * (vertices[idx2] - vertices[idx1]))

        elif self.kind == Portal.ToPortal:
            other_portal, para = self.end_info
            if para == 0:
                return vertices[other_portal.start_index]
            elif para == 1:
                return other_portal.calc_endpoint(vertices)
            else:
                raise RuntimeError("{} - what is this parameter value? It should be either 0 or 1".format(para))
        else:
             raise RuntimeError("Portal of unknown kind: {}".format(self.kind))




class Mesh2d(object):

    @staticmethod
    def find_spikes(poly, threshold):
        spikes = []
        accum_angle = 0.
        for loop_start in poly.graph.loops:
            indices = chain(poly.graph.loop_iterator(loop_start),
                [loop_start, poly.graph.next[loop_start]])

            for i0, i1, i2 in triples(indices):
                prev_v = poly.vertices[i0]
                cand_v = poly.vertices[i1]
                next_v = poly.vertices[i2]

                signed_area = Geom2.signed_area(prev_v, cand_v, next_v)
                if signed_area < 0.0:
                    side1 = cand_v - prev_v
                    side2 = next_v - cand_v

                    # positive external angle is concavity
                    external_angle = math.acos(Geom2.cos_angle(side1, side2))
                    external_angle = external_angle*180.0 / math.pi
                    accum_angle = max(0., accum_angle + external_angle)
                    if accum_angle > threshold:
                        spikes.append((i0, i1, i2))
                        accum_angle = 0.
        return spikes


    @staticmethod
    def get_sector(prev_v, spike_v, next_v, threshold=0.0):
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

        # degrees to radian
        clearance = math.pi * threshold / 180.0
        sector_angle_plus = sector_angle + clearance

        
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


    @staticmethod
    def find_closest_edge_inside_sector(poly, sector, tip_idx, cutoff1, cutoff2):
        _, tip, _ = sector
        cutoff_plane = None
        closest_para, closest_distSq, closest_edge = None, None, None

        for loop_start in poly.graph.loops:
            indices = chain(poly.graph.loop_iterator(loop_start), [loop_start])

            for (seg_i1, seg_i2) in pairs(indices):
                if tip_idx in (seg_i1, seg_i2):
                    continue

                (seg0, seg1) = (poly.vertices[seg_i1], poly.vertices[seg_i2])

                # with timed_exec("cutoff plane check"):
                #     seg0x, seg1x = seg0.append(1), seg1.append(1)

                #     if (seg0x.dot(cutoff1) < 0 and seg1x.dot(cutoff1) < 0) or (seg0x.dot(cutoff2) < 0 and seg1x.dot(cutoff2) < 0):
                #         continue

                #     elif cutoff_plane and seg0x.dot(cutoff_plane) > 0 and seg1x.dot(cutoff_plane) > 0:
                #         continue

                with timed_exec("_segment_closest_point_inside_sector"):
                    para, point, distSq = Mesh2d._segment_closest_point_inside_sector((seg0, seg1), sector)

                if para is None: continue
                if closest_distSq is None or distSq < closest_distSq:
                    closest_distSq = distSq
                    closest_para = para
                    closest_edge = (seg_i1, seg_i2)

                    # make cutoff line
                    n = point - tip
                    cutoff_plane = n.append(-n.dot(point))
                    # print("cutoff plane: {}".format(cutoff_plane))

        # print("{} considered, {} skipped".format(considered_num, skipped_num))
        return closest_edge, closest_para, closest_distSq



    @staticmethod
    def find_closest_portal_inside_sector(poly, sector, tip_idx, portals, cutoff1, cutoff2):
        _, tip, _ = sector
        cutoff_plane = None
        closest_portal, closest_distSq, closest_para = None, None, None

        for portal in portals:
            if tip_idx == portal.start_index: continue

            (seg0, seg1) = (poly.vertices[portal.start_index], portal.calc_endpoint(poly.vertices))
            if seg0 == seg1: continue

            # with timed_exec("cutoff plane check"):
            #     seg0x, seg1x = seg0.append(1), seg1.append(1)

            #     if (seg0x.dot(cutoff1) < 0 and seg1x.dot(cutoff1) < 0) or (seg0x.dot(cutoff2) < 0 and seg1x.dot(cutoff2) < 0):
            #         continue

            #     elif cutoff_plane and seg0x.dot(cutoff_plane) > 0 and seg1x.dot(cutoff_plane) > 0:
            #         continue

            with timed_exec("_segment_closest_point_inside_sector"):
                para, point, distSq = Mesh2d._segment_closest_point_inside_sector((seg0, seg1), sector)

            if para is None: continue

            # update closest portal
            if closest_distSq is None or distSq < closest_distSq:
                closest_distSq = distSq
                closest_para = para
                closest_portal = portal

                # make cutoff line
                n = point - tip
                cutoff_plane = n.append(-n.dot(point))

        return closest_portal, closest_para, closest_distSq


    @staticmethod
    def _segment_closest_point_inside_sector(segment, sector):
        dir1, tip, dir2 = sector

        def pt_inside(pt):
            rel_pt = pt - tip
            return vec.cross2(dir1, rel_pt) > 0 and vec.cross2(dir2, rel_pt) < 0

        # make ray-like line
        with timed_exec("section 1 (finding line intersections)"):
            line = (segment[0], segment[1] - segment[0])
            (linepara1, raypara1, _) = Geom2.lines_intersect(line, (tip, dir1))
            (linepara2, raypara2, _) = Geom2.lines_intersect(line, (tip, dir2))

        # check whether line containing segment passes through sector:
        line_through_sector = raypara1 > 0 or raypara2 > 0

        if not line_through_sector:
            return None, None, None

        # now we know that line goes through sector
        with timed_exec("section 2"):
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

        with timed_exec("section 3"):
            # find point on line closest to tip
            projpara = Geom2.project_to_line(tip, line)

            if candid_params[0] < projpara and projpara < candid_params[1]:
                candid_params.append(projpara)


            def para_to_pt(para):
                return line[0] + (para * line[1])

            distances = ((tip - para_to_pt(para)).normSq() for para in candid_params)
            with timed_exec("section 3.1 minimization"):
                p, d = min(izip(candid_params, distances), key = itemgetter(1))
                return p, para_to_pt(p), d





    @staticmethod
    def find_portals(poly, threshold):

        """
        This function uses algorithm from R. Oliva and N. Pelechano - 
        Automatic Generation of Suboptimal NavMeshes
        This is work in progress
        The endpoints of portals can be new vertices,
        but they are guaranteed to lie on the polygon boundary (not inside the polygon)
        """
        spikes = Mesh2d.find_spikes(poly, threshold)

        portals = []

        for spike in spikes:
            (_, spike_idx, _) = spike
            (v0, v1, v2) = (poly.vertices[idx] for idx in spike)

            right_dir, left_dir = Mesh2d.get_sector(v0, v1, v2, threshold)

            n_right = left_dir - left_dir.dot(right_dir) * right_dir
            n_left  = right_dir - right_dir.dot(left_dir) * left_dir

            hp1 = n_right.append(-n_right.dot(v1))
            hp2 = n_left.append(-n_left.dot(v1))

            tip = poly.vertices[spike_idx]
            sector = (right_dir, tip, left_dir)

            # find closest edge
            closest_seg, closest_seg_para, closest_seg_dst = \
                Mesh2d.find_closest_edge_inside_sector(poly, sector, spike_idx, hp1, hp2)

            if closest_seg is None:
                raise RuntimeError("Could not find a single edge inside sector")

            # find closest portal
            closest_portal, closest_portal_para, closest_portal_dst = \
                Mesh2d.find_closest_portal_inside_sector(poly, sector, spike_idx, portals, hp1, hp2)

            # closest edge always exists (unless something's horribly wrong)
            # closest portal - not always (there might be no portals inside the sector
            # or no portals at all)
            portal = Portal()

            # we might want to add a second portal later
            new_portals = [portal]

            portal.start_index = spike_idx
            if closest_seg_para == 0 or closest_seg_para == 1:
                portal.kind = Portal.ToVertex
                portal.end_info = closest_seg[int(closest_seg_para)]

            else:
                portal.kind = Portal.ToSegment
                portal.end_info = closest_seg, closest_seg_para

            closest_dst = closest_seg_dst

            # check if there is a portal closer than the previous closest element
            # TODO When attaching to an existing portal, need to reconsider the necessity of the older portal
            if closest_portal_dst is not None and closest_portal_dst < closest_dst:
                closest_dst = closest_portal_dst

                # figure out if we want to create one or two portals

                # we only want to connect to one or two endpoints,
                # not to an intermediate point of the portal
                portal.kind = Portal.ToPortal

                if closest_portal_para == 0:
                    portal.end_info = closest_portal, 0
                elif closest_portal_para == 1:
                    portal.end_info = closest_portal, 1

                # If closest point is not one of the endpoints,
                # we still create the portal(s) to one or two endpoints.
                else:
                    def pt_inside_sector(sector, pt):
                        dir1, tip, dir2 = sector
                        rel_pt = pt - tip
                        return vec.cross2(dir1, rel_pt) > 0 and vec.cross2(dir2, rel_pt) < 0

                    other_start_pt = poly.vertices[closest_portal.start_index]
                    other_end_pt = closest_portal.calc_endpoint(poly.vertices)

                    start_inside = pt_inside_sector(sector, other_start_pt)
                    end_inside = pt_inside_sector(sector, other_end_pt)

                    # if none of the portal endpoints is inside sector, create 2 portals to both ends:
                    if not start_inside and not end_inside:
                        second_portal = deepcopy(portal)
                        portal.end_info = closest_portal, 0
                        second_portal.end_info = closest_portal, 1
                        new_portals.append(second_portal)

                    elif start_inside:
                        portal.end_info = closest_portal, 0
                    else:
                        portal.end_info = closest_portal, 1

            portals.extend(new_portals)

        return portals



    def __init__(self, poly, convex_relax_thresh = 0.0):
        poly = deepcopy(poly)

        portals = Mesh2d.find_portals(poly, convex_relax_thresh)

        for name, time in timing.items():
            print("{} took on average {:.3f} ms,   called {} times,   {} ms total".format(name, 1000 * time.average(), time.n, 1000 * time.accum))

        # insert new vertices for all 'ToSegment' portals and switch them to 'ToVertex'
        segments_to_split = defaultdict(list)
        for portal in portals:

            if portal.kind == Portal.ToSegment:
                segment, _ = portal.end_info
                segments_to_split[segment].append(portal)

        for segment, seg_portals in segments_to_split.items():
            end_params = list(portal.end_info[1] for portal in seg_portals)
            end_ids = poly.add_vertices_to_border(segment, end_params)

            for end_idx, portal in izip(end_ids, seg_portals):
                portal.kind = Portal.ToVertex
                portal.end_info = end_idx


        def resolve_chain(portal):
            if portal.kind == Portal.ToPortal:
                next_portal, para = portal.end_info
                resolve_chain(next_portal)

                portal.kind = Portal.ToVertex
                if para == 0:
                    portal.end_info = next_portal.start_index
                elif para == 1:
                    portal.end_info = next_portal.end_info
                else:
                    raise RuntimeError(
                        "{} - what is this parameter value?"
                        " It should be either 0 or 1".format(para))

        # convert all 'ToPortal' portals to 'ToVertex' portals
        for portal in portals:
            # travel down the chain of linked portals until arrive to 'ToVertex' portal
            resolve_chain(portal)

        # now all portals are 'ToVertex'

        # convert portals to edges (tuples)
        portals = ((p.start_index, p.end_info) for p in portals)

        # remove degen portals
        portals = list(p for p in portals if p[0] != p[1])

        # for each portal add reversed portal
        fwd, back = tee(portals, 2)
        portals = chain(fwd, (tuple(reversed(p)) for p in back))
        # remove duplicates that might have occured if
        # we already had pairs of opposites among original portals
        portals = list(set(portals))

        # print("{} portals".format(len(portals)))


        # list of edges
        edge_buffer = list()

        # linked lists of indices of outgoing edges for each vertex index
        ways_to_go = Loops()
        first_way_ids = list(None for _ in xrange(len(poly.graph.next)))

        for src in poly.graph.all_nodes_iterator():
            tgt = poly.graph.next[src]
            first_way_ids[src] = ways_to_go.add_loop(1)
            edge_buffer.append((src, tgt))

        for src, tgt in portals:
            way_idx = first_way_ids[src]
            ways_to_go.insert_node((way_idx, ways_to_go.next[way_idx]))
            edge_buffer.append((src, tgt))


        def angle(v0, v1):
            # backtracking gives -180 degrees, not 180
            cosine = Geom2.cos_angle(v0, v1)
            sine = Geom2.sin_angle(v0, v1)
            return math.acos(cosine) if sine > 0 else -math.acos(cosine)


        # make rooms (a room is just a list of vertex ids)
        rooms = list()
        consumed = [False] * len(edge_buffer)
        for idx, _ in enumerate(edge_buffer):
            if consumed[idx]: continue
            room = list()
            rooms.append(room)

            while not consumed[idx]:
                consumed[idx] = True
                src, tgt = edge_buffer[idx]
                room.append(src)
                ways0, ways1 = tee(ways_to_go.loop_iterator(first_way_ids[tgt]))

                in_edge = poly.vertices[tgt] - poly.vertices[src]
                out_edges = (poly.vertices[nxt] - poly.vertices[tgt] \
                    for nxt in (edge_buffer[way][1] for way in ways0))

                out_angles = (angle(in_edge, out_edge) for out_edge in out_edges)
                idx, _ = max(izip(ways1, out_angles), key = itemgetter(1))

        # for r in rooms: print(r)
        self.rooms = rooms
        self.portals = portals
        self.vertices = poly.vertices
        self.outline = list(poly.graph.loop_iterator(poly.graph.loops[0]))
        self.holes = list(list(poly.graph.loop_iterator(h)) for h in poly.graph.loops[1:])
