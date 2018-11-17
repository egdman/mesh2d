import math

from collections import defaultdict
from operator import itemgetter
from itertools import chain, repeat, tee
from copy import deepcopy
from time import clock
from contextlib import contextmanager

try:
    from itertools import izip as zip
except ImportError:
    pass

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
            other_portal = self.end_info
            return other_portal.calc_endpoint(vertices)

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


                seg0x, seg1x = seg0.append(1), seg1.append(1)

                if (seg0x.dot(cutoff1) < 0 and seg1x.dot(cutoff1) < 0) or (seg0x.dot(cutoff2) < 0 and seg1x.dot(cutoff2) < 0):
                    continue

                elif cutoff_plane and seg0x.dot(cutoff_plane) > 0 and seg1x.dot(cutoff_plane) > 0:
                    continue

                para, point, distSq = Mesh2d._segment_closest_point_inside_sector((seg0, seg1), sector)

                if para is None: continue
                if closest_distSq is None or distSq < closest_distSq:
                    closest_distSq = distSq
                    closest_para = para
                    closest_edge = (seg_i1, seg_i2)

                    # make cutoff line
                    n = point - tip
                    cutoff_plane = n.append(-n.dot(point))

        return closest_edge, closest_para, closest_distSq


    @staticmethod
    def find_closest_portal_inside_sector(poly, sector, tip_idx, portals, cutoff1, cutoff2):
        dir1, tip, dir2 = sector
        cutoff_plane = None
        closest_portal, closest_distSq, closest_para = None, None, None

        for portal in portals:
            if tip_idx == portal.start_index: continue

            (seg0, seg1) = (poly.vertices[portal.start_index], portal.calc_endpoint(poly.vertices))
            if seg0 == seg1: continue

            # special case: portal endpoint is at the sector tip
            # if other endpoint is inside sector, return portal as closest with distance = 0
            # if other endpoint is outside sector, skip this portal
            if seg1 == tip:
                dir_p = seg0 - tip
                seg0_inside = vec.cross2(dir1, dir_p) > 0 and vec.cross2(dir2, dir_p) < 0
                if seg0_inside:
                    return portal, 1., 0.
                else:
                    continue

            seg0x, seg1x = seg0.append(1), seg1.append(1)

            if (seg0x.dot(cutoff1) < 0 and seg1x.dot(cutoff1) < 0) or (seg0x.dot(cutoff2) < 0 and seg1x.dot(cutoff2) < 0):
                continue

            elif cutoff_plane and seg0x.dot(cutoff_plane) > 0 and seg1x.dot(cutoff_plane) > 0:
                continue

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





    @staticmethod
    def find_portals(poly, threshold, db_visitor=None):

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

            # if db_visitor is not None:
            #     pts = v1 + 50. * right_dir, v1, v1 + 50. * left_dir
            #     db_visitor.add_polygon(pts, '#ffff22')

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

            # if db_visitor is not None:
            #     idx0, idx1 = closest_seg
            #     p0 = poly.vertices[idx0]
            #     p1 = poly.vertices[idx1]
            #     db_visitor.add_polygon((p0, tip, p1, tip), color='#119875')


            # closest edge always exists (unless something's horribly wrong)
            # closest portal - not always (there might be no portals inside the sector
            # or no portals at all)
            portal = Portal()

            # we might want to add a second portal later
            new_portals = [portal]
            portal.start_index = spike_idx
            
            # check if there is a portal closer than the closest edge
            # TODO When attaching to an existing portal, need to reconsider the necessity of the older portal
            if closest_portal_dst is not None and closest_portal_dst < closest_seg_dst:
                if closest_portal_dst == 0:
                    continue

                # figure out if we want to create one or two portals
                # we only want to connect to one or two endpoints,
                # not to an intermediate point of the portal
                if closest_portal_para == 0:
                    portal.kind = Portal.ToVertex
                    portal.end_info = closest_portal.start_index
                elif closest_portal_para == 1:
                    portal.kind = Portal.ToPortal
                    portal.end_info = closest_portal

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
                        portal.kind = Portal.ToVertex
                        portal.end_info = closest_portal.start_index

                        second_portal = Portal()
                        second_portal.start_index = portal.start_index
                        if closest_portal.kind == Portal.ToVertex:
                            second_portal.kind = Portal.ToVertex
                            second_portal.end_info = closest_portal.end_info
                        else:
                            second_portal.kind = Portal.ToPortal
                            second_portal.end_info = closest_portal

                        new_portals.append(second_portal)

                    elif start_inside:
                        portal.kind = Portal.ToVertex
                        portal.end_info = closest_portal.start_index # attach to portal startPt
                    else:
                        portal.kind = Portal.ToPortal
                        portal.end_info = closest_portal

            # if no portals are in the way, attach to edge/vertex
            else:
                if closest_seg_para == 0 or closest_seg_para == 1:
                    portal.kind = Portal.ToVertex
                    portal.end_info = closest_seg[int(closest_seg_para)]
                else:
                    portal.kind = Portal.ToSegment
                    portal.end_info = closest_seg, closest_seg_para

            portals.extend(new_portals)
        return portals



    def __init__(self, poly, convex_relax_thresh = 0.0, db_visitor=None):
        poly = deepcopy(poly)

        portals = Mesh2d.find_portals(poly, convex_relax_thresh, db_visitor)

        # insert new vertices for all 'ToSegment' portals and switch them to 'ToVertex'
        segments_to_split = defaultdict(list)
        for portal in portals:
            if portal.kind == Portal.ToSegment:
                segment, _ = portal.end_info
                segments_to_split[segment].append(portal)

        for segment, seg_portals in segments_to_split.items():
            split_params = list(portal.end_info[1] for portal in seg_portals)
            split_ids = poly.add_vertices_to_border(segment, split_params)

            for split_idx, portal in zip(split_ids, seg_portals):
                portal.kind = Portal.ToVertex
                portal.end_info = split_idx


        def resolve_chain(portal):
            if portal.kind == Portal.ToPortal:
                next_portal = portal.end_info
                resolve_chain(next_portal)
                portal.kind = Portal.ToVertex
                portal.end_info = next_portal.end_info

        # convert all 'ToPortal' portals to 'ToVertex' portals
        for portal in portals:
            # travel down the chain of linked portals until arrive to 'ToVertex' portal
            resolve_chain(portal)
        # now all portals are 'ToVertex'

        portals = list((p.start_index, p.end_info) for p in portals)

        # list of edges
        edge_buffer = list()
        is_border = list()

        def opposite(eid):
            return eid - 1 if eid & 1 else eid + 1

        # lists of indices of outgoing edges for each vertex index
        ways_to_go = list([] for _ in xrange(len(poly.graph.next)))

        for src in poly.graph.all_nodes_iterator():
            tgt = poly.graph.next[src]

            ways_to_go[src].append(len(edge_buffer))
            edge_buffer.append(tgt)
            is_border.append(True)

            ways_to_go[tgt].append(len(edge_buffer))
            edge_buffer.append(src)
            is_border.append(False)

        for src, tgt in portals:

            ways_to_go[src].append(len(edge_buffer))
            edge_buffer.append(tgt)
            is_border.append(False)

            ways_to_go[tgt].append(len(edge_buffer))
            edge_buffer.append(src)
            is_border.append(False)


        next_edges = list(None for _ in xrange(len(edge_buffer)))
        prev_edges = list(None for _ in xrange(len(edge_buffer)))
        inbound_edges = list(None for _ in xrange(len(poly.graph.next)))

        if db_visitor is not None:
            for idx in poly.graph.all_nodes_iterator():
                db_visitor.add_text(loc=poly.vertices[idx], text=str(idx), scale=False)

        # make rooms (a room is just a list of vertex ids that form a nearly convex polygon)
        rooms = list()
        consumed = [False] * len(edge_buffer)
        for idx, _ in enumerate(edge_buffer):
            if consumed[idx]: continue
            room = list()
            rooms.append(room)

            while not consumed[idx]:
                consumed[idx] = True
                tgt = edge_buffer[idx]
                src = edge_buffer[opposite(idx)]
                if is_border[idx]:
                    inbound_edges[tgt] = idx

                room.append(src)

                in_dir = poly.vertices[tgt] - poly.vertices[src]

                e_ids = ways_to_go[tgt]
                v_ids = list(edge_buffer[edge_idx] for edge_idx in e_ids)

                direcs = list((e_idx, (poly.vertices[v_idx] - poly.vertices[tgt]).normalized())
                    for e_idx, v_idx in zip(e_ids, v_ids) if v_idx != src) # backtracking not allowed

                # pick the leftmost direction as the next edge
                left, right = [], []
                for e_idx, out_dir in direcs:
                    if vec.cross2(in_dir, out_dir) < 0:
                        left.append((e_idx, out_dir))
                    else:
                        right.append((e_idx, out_dir))

                if left:
                    candidates = left
                    pick = min
                else:
                    candidates = right
                    pick = max

                projections = (out_dir.dot(in_dir) for _, out_dir in candidates)
                picked_idx, _ = pick(enumerate(projections), key=itemgetter(1))
                next_idx, _ = candidates[picked_idx]

                next_edges[idx] = next_idx
                prev_edges[next_idx] = idx
                idx = next_idx

        def next_edge(eid):
            return next_edges[eid]

        def prev_edge(eid):
            return prev_edges[eid]

        # def verts_around_vert(vid):
        #     eid = inbound_edges[vid]
        #     op_eid = opposite(eid)
        #     yield edge_buffer[op_eid]
        #     next_eid = next_edge(eid)
        #     while next_eid != op_eid:
        #         yield edge_buffer[next_eid]
        #         next_eid = next_edge(opposite(next_eid))

        def inbound_edges_around_vert(vid):
            eid = inbound_edges[vid]
            yield eid
            next_eid = prev_edge(opposite(eid))
            while next_eid != eid:
                yield next_eid
                next_eid = prev_edge(opposite(next_eid))


        def verts_around_face(eid):
            yield edge_buffer[eid]
            next_eid = next_edge(eid)
            while next_eid != eid:
                yield edge_buffer[next_eid]
                next_eid = next_edge(next_eid)

        border_verts = list(verts_around_face(next(eid for eid in next_edges if is_border[eid])))
        print("border: {}".format(border_verts))

        redundant = list(False for _ in xrange(len(edge_buffer)))
        for vid in poly.graph.all_nodes_iterator():
            inbound_portals = list(inbound_edges_around_vert(vid))[1:-1]

            for p in inbound_portals:
                if redundant[p]:
                    continue

                # calc sum of angles before and after the portal p
                eid_before = next_edge(p)
                while redundant[eid_before]:
                    eid_before = next_edge(opposite(eid_before))

                eid_after = prev_edge(opposite(p))
                while redundant[eid_after]:
                    eid_after = prev_edge(opposite(eid_after))

                vid_before = edge_buffer[eid_before]
                vid_after = edge_buffer[opposite(eid_after)]

                


            in_border_eid = inbound_edges[vid]
            in_border_vid = edge_buffer[opposite(in_border_eid)]
            in_dir = poly.vertices[vid] - poly.vertices[in_border_vid]


            inbound = list(inbound_edges_around_vert(vid))
            if len(inbound) < 3:
                continue

            # in_dir = poly.vertices[vid] - poly.vertices[edge_buffer[opposite(inbound[0])]]
            # for eid in inbound[1:-1]:
            #     portal_vid = edge_buffer[opposite(eid)]
            #     portal_dir = poly.vertices[portal_vid] - poly.vertices[vid]
            #     after_portal_vid = edge_buffer[opposite(prev_edge(opposite(eid)))]
            #     after_portal_dir = poly.vertices[after_portal_vid] - poly.vertices[vid]



            verts_around = list(edge_buffer[opposite(eid)] for eid in inbound)            
            verts_around = list(str(idx) for idx in verts_around)
            print("around {}: [{}]".format(vid, ', '.join(verts_around)))





        print("---------------------------------------------------------------------")
        self.rooms = rooms
        self.portals = portals
        self.vertices = poly.vertices
        self.outline = list(poly.graph.loop_iterator(poly.graph.loops[0]))
        self.holes = list(list(poly.graph.loop_iterator(h)) for h in poly.graph.loops[1:])
