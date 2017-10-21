import math
import random

from collections import deque
from operator import itemgetter
from itertools import chain
from copy import deepcopy
from rtree import index

from .vector2 import vec, Geom2
from .utils import debug_draw_room


def print_p(p):
    print("start : {}".format(p['start_index']))
    print("end   : {}, {}".format(p['end_index'], p['end_point']))
    print ("----------------------------------------------")



class Polygon2d(object):
    def __init__(self, vertices, indices):

        # need to find self-intersections
        if Polygon2d.check_ccw(vertices, indices):
            self.outline = indices[:]
        else:
            self.outline = indices[::-1]

        self.vertices = vertices[:]

        self.rti = index.Index()
        self.rti.interleaved = True

        # insert vertices into spatial index
        for vid in self.outline:
            vert = self.vertices[vid]
            self.rti.insert(vid, (vert.x, vert.y, vert.x, vert.y))

        # resolve self-intersections (sinters)
        self._resolve_sinters()

        self.holes = []



    def copy(self):
        res = Polygon2d(self.vertices, self.outline)
        for hole in self.holes:
            res.add_hole((self.vertices[idx] for idx in hole))
        return res



    def outline_coordinates(self, indices=None):
        if indices is None: indices = self.outline
        crds = []

        for ind in indices:
            crds.append(self.vertices[ind].x)
            crds.append(self.vertices[ind].y)
        crds.append(self.vertices[indices[0]].x)
        crds.append(self.vertices[indices[0]].y)
        return crds



    def get_adjacent_edges(self, index):
        vert_loc = self.outline.index(index)
        prev_idx = self.outline[(vert_loc - 1)]
        next_idx = self.outline[(vert_loc + 1) % len(self.outline)]
        return (prev_idx, index), (index, next_idx)



    def find_verts_in_bbox(self, vect_min, vect_max):
        return self.rti.intersection((vect_min.x, vect_min.y, vect_max.x, vect_max.y))



    def add_hole(self, vertices):
        hole = []
        for vert in vertices:
            new_idx = len(self.vertices)
            self.vertices.append(vert)
            hole.append(new_idx)

            # add vertex to spatial index:
            self.rti.insert(new_idx, (vert.x, vert.y, vert.x, vert.y))

        # Holes must be CW
        if Polygon2d.check_ccw(self.vertices, hole):
            hole = hole[::-1]

        self.holes.append(hole)



    def _segments_cross_helper(self, seg1, seg2):
        seg1 = (self.vertices[seg1[0]], self.vertices[seg1[1]])
        seg2 = (self.vertices[seg2[0]], self.vertices[seg2[1]])
        seg_x = Polygon2d._segments_intersect(seg1, seg2)
        # if segments intersect at endpoints, ignore
        if seg_x in chain(seg1, seg2):
            return None
        else:
            return seg_x



    def _find_first_sinter(self, segments):
        # first segment with every other
        seg1 = segments[0]
        num_seg = len(segments)
        for j in range(2, num_seg - 1):
            seg2 = segments[j]
            seg_x = self._segments_cross_helper(seg1, seg2)
            if seg_x is not None:
                return (seg1, seg2, seg_x)

        # remaining segments
        for i in range(1, num_seg - 2):
            for j in range(i+2, num_seg):
                seg1 = segments[i]
                seg2 = segments[j]
                seg_x = self._segments_cross_helper(seg1, seg2)
                if seg_x is not None:
                    return (seg1, seg2, seg_x)

        return (None, None, None)



    def _pull_direction(self, edge1, edge2, intersect_vector):
        e11 = self.vertices[edge1[0]]
        e21 = self.vertices[edge2[0]]

        vec1 = e11 - intersect_vector
        vec2 = e21 - intersect_vector

        vec1 /= vec1.length()
        vec2 /= vec2.length()

        dir1 = vec1 + vec2
        dir2 = vec1 - vec2

        ldir1 = dir1.length()
        ldir2 = dir2.length()

        if ldir1 > ldir2:
            return dir1 / ldir1
        else:
            return dir2 / ldir2



    def _resolve_sinters(self):
        while True:
            segments = Polygon2d.get_segments([self.outline])
            (seg1, seg2, seg_x) = self._find_first_sinter(segments)

            # stop when there are no more sinters
            if seg_x is None: break

            # pull vertices apart:
            # determine direction:
            pull_dir = self._pull_direction(seg1, seg2, seg_x)
            nv1 = seg_x + pull_dir * .5
            nv2 = seg_x - pull_dir * .5

            # insert 2 new vertices at the intersection
            new_idx1 = self.add_vertex_to_outline(nv1, seg1)
            new_idx2 = self.add_vertex_to_outline(nv2, seg2)

            # reverse indices:
            self.outline = self._mirror_indices(self.outline, new_idx1, new_idx2)


            # flip new verts if necessary:
            if self._segments_cross_helper((seg1[0], new_idx1), (new_idx2, seg2[1])):
                self.vertices[new_idx1], self.vertices[new_idx2] = \
                self.vertices[new_idx2], self.vertices[new_idx1]


        # check that points are ccw, if not - reverse
        if not Polygon2d.check_ccw(self.vertices, self.outline):
            self.outline = self.outline[::-1]




    def add_vertex_to_outline(self, vertex, edge):
        return self.add_vertex_to_loop(vertex, edge, self.outline)



    def add_vertex_to_border(self, vertex, edge):
        '''
        Add the given vertex to the given edge of the polygon border.
        The exact loop that contains the edge is determined automatically.
        '''
        wanted_loop = next((loop for loop in chain([self.outline], self.holes) \
                if edge[0] in loop and edge[1] in loop), None)

        if wanted_loop is not None:
            return self.add_vertex_to_loop(vertex, edge, wanted_loop)

        raise ValueError("Adding vertex to borders: invalid edge")




    def add_vertices_to_border(self, vertices, edge):
        '''
        Add given list of vertices to the given edge of the polygon border.
        The exact loop that contains the edge is determined automatically.
        This function returns a list of new indices for the list of vertices in the same order.
        '''
        wanted_loop = next((loop for loop in chain([self.outline], self.holes) \
                if edge[0] in loop and edge[1] in loop), None)

        if wanted_loop is not None:

            # sort vertices by distance from smaller-index end of edge
            sorted_pos_verts = sorted(enumerate(vertices),
                key = lambda elem: (elem[1] - self.vertices[edge[0]]).norm())

            new_ids = [0]*len(vertices)

            last_added_idx = edge[0]
            for (pos, vert) in sorted_pos_verts:
                last_added_idx = self.add_vertex_to_loop(vert, (last_added_idx, edge[1]), wanted_loop)
                new_ids[pos] = last_added_idx

            return new_ids 

        raise ValueError("Adding vertex to borders: invalid edge")




    def add_vertex_to_loop(self, vertex, edge, loop):
        '''
        Add vertex to the index buffer passed as 'loop' argument.
        '''
        e1_idx = edge[0]
        e2_idx = edge[1]

        e1_pos = loop.index(e1_idx)
        e2_pos = loop.index(e2_idx)

        # e1_pos and e2_pos can either differ by 1
        # or loop around the index buffer
        if e1_pos > e2_pos: e1_pos, e2_pos = e2_pos, e1_pos

        if e1_pos == e2_pos: raise ValueError("Adding vertex to loop: invalid edge")

        if e2_pos - e1_pos > 1:
            if e1_pos != 0: raise ValueError("Adding vertex to loop: invalid edge")
            insert_at = e2_pos + 1

        else:
            insert_at = e2_pos

        # check if this vertex is too close to a segment endpoint
        if vertex == self.vertices[e1_idx]: return e1_idx
        if vertex == self.vertices[e2_idx]: return e2_idx

        new_vert_index = len(self.vertices)
        self.vertices.append(vertex)
        loop.insert(insert_at, new_vert_index)

        # add new vertex to spatial index:
        self.rti.insert(new_vert_index,
            (vertex.x, vertex.y, vertex.x, vertex.y))

        return new_vert_index



    def _split_border(self, loops, index_1, index_2):
        '''
        Either split the outline loop in 2 parts, or connect 2 loops into one
        '''
        if index_1 == index_2:
            raise ValueError("Split indices must not be equal to each other")

        # if splitting the outline (it comes first in 'loops'):
        if index_1 in loops[0] and index_2 in loops[0]:
            outline1, outline2 = Polygon2d._split_index_buffer(loops[0], index_1, index_2)

            # if we tried to split using an existing edge, return None, None
            if len(outline1) < 3 or len(outline2) < 3:
                return None, None

            # put pieces of the outline first into resulting loops
            loops1 = [outline1]
            loops2 = [outline2]

            # figure out which holes go where
            for hole in loops[1:]:
                if Polygon2d.point_inside_loop(self.vertices, outline1, self.vertices[hole[0]]):
                    loops1.append(hole)
                else:
                    loops2.append(hole)

            return loops1, loops2

        # if we are connecting 2 loops into one (hole + hole or hole + outline)
        else:
            
            # find first loop that has index_1 (there should be only one)
            index_1_in = next((num for (num, loop) in enumerate(loops) if index_1 in loop), None)
            # find first loop that has index_2 (there should be only one)
            index_2_in = next((num for (num, loop) in enumerate(loops) if index_2 in loop), None)

            if index_1_in is None or index_2_in is None:
                raise ValueError("Split indices must be present in the index buffers")

            if index_1_in == index_2_in:
                raise ValueError("You cannot split a hole")

            merged_loop = Polygon2d._merge_loops(
                loops[index_1_in],
                loops[index_2_in],
                index_1,
                index_2)


            loops1 = loops[:]

            # delete old loops:
            if index_1_in > index_2_in: index_1_in, index_2_in = index_2_in, index_1_in
            del loops1[index_2_in]
            del loops1[index_1_in]

            # if one of loops used to be the outline, prepend the new outline at the start
            if index_1_in == 0 or index_2_in == 0:
                loops1 = [merged_loop] + loops1

            # otherwise, add the new hole at the end
            else:
                loops1.append(merged_loop)

            return loops1, None



    def point_inside(self, point):
        # first check if inside outline
        inside_outline = Polygon2d.point_inside_loop(self.vertices, self.outline, point)
        if not inside_outline: return False

        # now check if inside a hole
        for hole in self.holes:
            if Polygon2d.point_inside_loop(self.vertices, hole, point):
                return False
        return True



    @staticmethod
    def point_inside_loop(vertices, indices, point):
        verts = list(vertices[idx] - point for idx in indices)
        nverts = len(verts)
        num_inters = 0
        for loc in range(nverts):
            cur_v = verts[loc]
            next_v = verts[(loc+1) % nverts]

            if cur_v.y == 0: cur_v.y += 0.01

            if next_v.y == 0: next_v.y += 0.01

            if cur_v.y * next_v.y < 0:

                _, b = Geom2.lines_intersect((cur_v, next_v - cur_v), (vec(0, 0), vec(1, 0)))
                if b > 0: num_inters += 1

        return num_inters % 2 > 0





    @staticmethod
    def _split_index_buffer(indices, index_1, index_2):
        if index_1 == index_2:
            raise ValueError("split indices must not be equal to each other")

        if index_1 not in indices or index_2 not in indices:
            raise ValueError("split indices must both be present in the index buffer")

        buffers = ([], [])
        switch = 0

        for index in indices:
            if index == index_1:
                buffers[switch].append(index_1)
                buffers[switch].append(index_2)
                switch = (switch + 1) % 2
            elif index == index_2: 
                buffers[switch].append(index_2)
                buffers[switch].append(index_1)
                switch = (switch + 1) % 2
            else:
                buffers[switch].append(index)

        return buffers



    @staticmethod
    def _merge_loops(loop1, loop2, index1, index2):
        '''
        Creates a 'cut' between two loops (index1 and index2 are the
        endpoints of the cut).
        Assumes that index1 in loops1, index2 in loops2.
        '''
        # shift loop1
        index1_at = loop1.index(index1)
        loop1 = loop1[index1_at:] + loop1[:index1_at] + [index1]

        index2_at = loop2.index(index2)
        return loop2[:index2_at+1] + loop1 + loop2[index2_at:]



    @staticmethod
    def get_segments(loops):
        '''
        Each element in 'loops' is an index buffer representing a separate
        piece of the polygon border. It can either be the outline or a hole.
        '''
        segs = []

        for loop in loops:
            for loc in range(len(loop) - 1):
                segs.append((loop[loc], loop[loc + 1]))
            segs.append((loop[-1], loop[0]))

        return segs



    @staticmethod
    def check_ccw(vertices, indices):
        return Polygon2d.signed_area(vertices, indices) > 0



    @staticmethod
    def signed_area(vertices, indices):
        return Geom2.poly_signed_area((vertices[idx] for idx in indices))



    @staticmethod
    def _mirror_indices(indices, start_after, end_before):
        '''
        input: indices=[0, 1, 2, 3, 4, 5, 6, 7], start_after=2, end_before=6
        output: [0, 1, 2, 5, 4, 3, 6, 7]
        '''
        start_loc = indices.index(start_after)
        end_loc = indices.index(end_before)

        if start_loc >= end_loc:
            raise ValueError("'start_after' must be to the left of 'end_before'")

        before = indices[:start_loc+1]
        middle = indices[start_loc+1:end_loc]
        after = indices[end_loc:]
        return before + middle[::-1] + after


    @staticmethod
    def _segments_intersect(seg1, seg2):
        if seg1[0] in seg2:
            return seg1
        if seg1[1] in seg2:
            return seg2

        line1 = (seg1[0], seg1[1] - seg1[0])
        line2 = (seg2[0], seg2[1] - seg2[0])
        a, b = Geom2.lines_intersect(line1, line2)
        if 0 < a and a < 1 and 0 < b and b < 1:
            return line1[0] + (a * line[1])
        else:
            return None





class Portal(object):
    def __init__(self):
        self.start_index = None
        self.kind = None
        self.created = False

    class ToSegment: pass
    class ToVertex: pass
    class ToPortal: pass

    def calc_endpoint(self, vertices):
        if self.kind == Portal.ToVertex:
            return vertices[self.end_info[0]]

        else if self.kind == Portal.ToSegment:
            idx1, idx2 = self.end_info[0]
            para = self.end_info[1]
            return vertices[idx1] + (para * (vertices[idx2] - vertices[idx1]))

        else if self.kind == Portal.ToPortal:
            other_portal, para = self.end_info
            if para == 0:
                return vertices[other_portal.start_index]
            else if para == 1:
                return other_portal.calc_endpoint(vertices)
            else:
                raise RuntimeError("{} - what is this parameter value? It should be either 0 or 1".format(para))
        else:
             raise RuntimeError("Portal of unknown kind: {}".format(self.kind))



class Mesh2d(Polygon2d):
    def __init__(self, vertices, indices):
        super(Mesh2d, self).__init__(vertices, indices)
        self.rooms = []
        self.portals = []



    @staticmethod
    def from_polygon(poly):
        poly.rooms = []
        poly.portals = []
        poly.__class__ = Mesh2d
        return poly



    def dump(self):
        return self.__dict__



    def break_into_convex(self, threshold = 0.0, cv=None):
        portals = self.get_portals(threshold=threshold)
        '''
        For all the portals that require creating new vertices, create new vertices.
        Multiple portals may have the same endpoint. If we have 5 portals that converge
        to the same endpoint, we only want to create the endpoint once.
        This is why some portals have a 'parent_portal' attribute - we create the endpoint
        for the parent portal only, and all the other portals use its index.
        '''

        for portal in portals:
            if portal.kind == Portal.ToSegment:
                segment, para, dist = portal.end_info
                end_idx = self.add_vertex_to_border(segment, para)
                portal.kind = Portal.ToVertex
                portal.end_info = end_idx, dist

                if cv:
                    vrt = self.vertices[end_idx]
                    cv.create_oval(vrt[0] - 3, vrt[1] - 3, vrt[0] + 3, vrt[1] + 3, fill='green')

            else if portal.kind == Portal.ToVertex:
                if cv:
                    vrt = self.vertices[portal.end_info[0]]
                    cv.create_oval(vrt[0] - 3, vrt[1] - 3, vrt[0] + 3, vrt[1] + 3, fill='blue')


        # now go through portals again to set child portals' end indexes
        for portal in portals:
            if portal.kind == Portal.ToPortal:
                other_portal, para = portal.end_info
                if other_portal.kind != Portal.ToVertex:
                    raise RuntimeError("Why does this portal have no ending vertex created??")
                portal.kind = Portal.ToVertex

                # attach to start of other portal
                if para == 0:
                    portal.end_info = other_portal.start_index, 0

                # attach to end of other portal
                else if para == 1:
                    portal.end_info = other_portal.end_info

                else:
                    raise RuntimeError("{} - what is this parameter value? It should be either 0 or 1".format(para))


        # Now break the mesh border into convex rooms

        # queue of rooms
        room_q = deque()

        # append a copy of the entire outline plus all the holes
        room_q.append([self.outline[:]] + deepcopy(self.holes))

        while len(room_q) > 0:
            room = room_q.popleft()

            room1, room2, new_portal = self._break_in_two(room, portals)

            # if could not split this room, finalize it
            if room1 is None:
                if len(room) > 1: raise RuntimeError("Trying to finalize a room with a hole")
                self.rooms.append(room[0])

            # otherwise add new rooms to the queue
            else:
                room_q.append(room1)
                if room2 is not None: room_q.append(room2)
                self.portals.append(new_portal)



    def _break_in_two(self, loops, portals):
        indices = list(chain(*loops))

        # iterate over portals trying to find the first portal that belongs to this polygon
        for portal in portals:

            # if this portal has already been created, skip it
            if portal.created: continue

            room1 = []
            room2 = []

            start_i = portal.start_index
            end_i, _ = portal.end_info

            # if this portal starts outside of this polygon, skip it
            if start_i not in indices or end_i not in indices:
                continue

            # split index buffer of the outline of this room using the portal
            room1, room2 = self._split_border(loops, start_i, end_i)

            if room1 is None and room2 is None: continue

            # mark this portal as created
            portal.created = True

            return room1, room2, (start_i, end_i)

        # if we did not find any portals to split, this room must be convex
        return None, None, None



    @staticmethod
    def check_convex(indices, vertices):

        vrt = vertices
        num_indices = len(indices)

        is_convex = True

        for cur_pos in range(num_indices):
            next_pos = (cur_pos + 1) % num_indices
            nnext_pos = (next_pos + 1) % num_indices

            cur_ind = indices[cur_pos]
            next_ind = indices[next_pos]
            nnext_ind = indices[nnext_pos]
            if Geom2.signed_area(vrt[cur_ind], vrt[next_ind], vrt[nnext_ind]) < 0:
                is_convex = False
                break

        return is_convex



    def find_spikes(self, threshold = 0.0):
        segments = Polygon2d.get_segments(chain([self.outline], self.holes))
        num_segs = len(segments)

        # find spikes:
        spikes = []
        start_seg = segments[0] # first seg of current loop
        for cur_pos in range(num_segs):

            cur_seg = segments[cur_pos]
            next_seg = segments[(cur_pos + 1) % num_segs]

            # check if we are at the end of a loop
            if cur_seg[1] != next_seg[0]:
                next_seg, start_seg = start_seg, next_seg

            prev_idx = cur_seg[0]
            cur_idx = cur_seg[1]
            next_idx = next_seg[1]

            prev_v = self.vertices[prev_idx]
            cand_v = self.vertices[cur_idx]
            next_v = self.vertices[next_idx]

            signed_area = Geom2.signed_area(prev_v, cand_v, next_v)
            if signed_area < 0.0:

                side1 = cand_v - prev_v
                side2 = next_v - cand_v

                external_angle = math.acos(Geom2.cos_angle(side1, side2))
                external_angle = external_angle*180.0 / math.pi

                if external_angle > threshold:
                    spikes.append((prev_idx, cur_idx, next_idx))

        return spikes



    def get_portals(self, threshold = 0.0, tolerance = 0.000001):

        """
        This function uses algorithm from R. Oliva and N. Pelechano - 
        Automatic Generation of Suboptimal NavMeshes
        This is work in progress
        The endpoints of portals can be new vertices,
        but they are guaranteed to lie on the polygon boundary (not inside the polygon)

        TODO This function is slightly less complex right now
        """
        spikes = self.find_spikes(threshold)
        portals = []

        for spike in spikes:
            (prev_i, spike_i, next_i) = spike

            right_dir, left_dir = self.get_sector(prev_i, spike_i, next_i, threshold)
            tip = self.vertices[spike_i]

            sector = (right_dir, tip, left_dir)
            # find closest edge
            closest_seg, closest_seg_para, closest_seg_dst = \
                self.find_closest_edge_inside_sector(sector, spike_i)

            # find closest vertex
            closest_vert_i, closest_vert_dst = \
                self.find_closest_vert_inside_sector(sector, spike_i)

            # find closest portal
            closest_portal, closest_portal_para, closest_portal_dst = \
                self.find_closest_portal_inside_sector(sector, spike_i, portals)


            # # remove tiny difference between points
            # if closest_portal is not None:
            #     if closest_portal_dst < tolerance:
            #         closest_portal_point.snap_to(tip)
            #         closest_portal_dst = 0.0



            # closest edge always exists
            # closest vertex - not always (there might be no vertices inside the sector)
            # closest portal - not always (there might be no portals inside the sector
            # or no portals at all)
            portal = Portal()

            # we might want to add a second portal later
            new_portals = [portal]

            portal.start_index = spike_i
            portal.kind = Portal.ToSegment
            portal.end_info = closest_seg, closest_seg_para, closest_seg_dst

            closest_dst = closest_seg_dst

            # check if there is a vertex closer or equally close than the closest edge (prefer vertex)
            if closest_vert_dst is not None and closest_vert_dst <= closest_dst:
                closest_dst = closest_vert_dst
                portal.kind = Portal.ToVertex
                portal.end_info = closest_vert_i, closest_vert_dst

            # check if there is a portal closer than the previous closest element
            # TODO When attaching to an existing portal, need to reconsider the necessity of the older portal
            if closest_portal_dst is not None and closest_portal_dst < closest_dst:
                closest_dst = closest_portal_dst

                # portal_start_i = closest_portal.start_index
                # portal_start_p = self.vertices[portal_start_i]

                # portal_end_i = closest_portal.end_index # might be None
                # portal_end_p = closest_portal.end_point

                # now we know:
                # position of starting point of other portal (portal_start_p)
                # index of starting point of other portal (portal_start_i)
                # position of ending point of other portal (portal_end_p)
                # (possibly) index of ending point of other portal (portal_end_i)

                # figure out if we want to create one or two portals

                # closest_portal_point can either be one of the endpoints,
                # or a middle point of the portal

                # we only want to connect to one or two endpoints,
                # not to the middle of the portal

                portal.kind = Portal.ToPortal

                if closest_portal_para == 0:
                    portal.end_info = closest_portal, 0
                else if closest_portal_para == 1:
                    portal.end_info = closest_portal, 1

                # If closest point is not one of the endpoints,
                # we still create the portal(s) to one or two endpoints.
                else:
                    def pt_inside_sector(sector, pt):
                        dir1, tip, dir2 = sector
                        rel_pt = pt - tip
                        return vec.cross2(dir1, rel_pt) > 0 and vec.cross2(dir2, rel_pt) < 0

                    other_start_pt = self.vertices[closest_portal.start_index]
                    other_end_pt = closest_portal.calc_endpoint(self.vertices)

                    start_inside = pt_inside_sector(sector, other_start_pt)
                    end_inside = pt_inside_sector(sector, other_end_pt)

                    # if none of the portal endpoints is inside sector, create 2 portals to both ends:
                    if not start_inside and not end_inside:
                        second_portal = deepcopy(portal)
                        portal.end_info = closest_portal, 0
                        second_portal.end_info = closest_portal, 1
                        new_portals.append(second_portal)

                    else if start_inside:
                        portal.end_info = closest_portal, 0
                    else:
                        portal.end_info = closest_portal, 1


            portals.extend(new_portals)

        return portals




    def find_closest_vert_inside_sector(self, sector, tip_idx):
        borders = chain([self.outline], self.holes)
        closest_idx, closest_distSq = None, None

        right_dir, tip, left_dir = sector

        for idx in chain(*borders):
            if tip_idx == idx: continue

            relative_pt = self.vertices[idx] - tip
            # check if current point is inside sector
            if vec.cross2(right_dir, relative_pt) > 0 and vec.cross2(left_dir, relative_pt) < 0:
                distSq = relative_pt.normSq()
                if closest_distSq is None or distSq < closest_distSq:
                    closest_distSq = distSq
                    closest_idx = idx

        return closest_idx, closest_distSq




    def find_closest_edge_inside_sector(self, sector, tip_idx):
        segments = Polygon2d.get_segments(chain([self.outline], self.holes))

        closest_para, closest_distSq, closest_edge = None, None, None

        for (seg_i1, seg_i2) in segments:

            # skip edges adjacent to the tip of the sector:
            if tip_idx in (seg_i1, seg_i2):
                continue

            segment = (self.vertices[seg_i1], self.vertices[seg_i2])
            para, distSq = self._segment_closest_point_inside_sector(segment, sector)
            if para is None: raise RuntimeError("Could not find a single edge inside sector")

            if closest_distSq is None or distSq < closest_distSq:
                closest_distSq = distSq
                closest_para = para
                closest_edge = (seg_i1, seg_i2)

        return closest_edge, closest_para, closest_distSq



    def find_closest_portal_inside_sector(self, sector, tip_idx, portals):
        closest_portal, closest_distSq, closest_para = None, None, None

        for portal in portals:
            if tip_idx == portal.start_index: continue

            portal_seg = (self.vertices[portal.start_index], portal.calc_endpoint(self.vertices))
            para, distSq = self._segment_closest_point_inside_sector(portal_seg, sector)
            if para is None: continue

            # update closest portal
            if closest_distSq is None or distSq < closest_distSq:
                closest_distSq = distSq
                closest_para = para
                closest_portal = portal

        return closest_portal, closest_para, closest_distSq



    def get_sector(self, prev_idx, spike_idx, next_idx, threshold=0.0):
        '''
        Returns 2 vectors that define the sector rays.
        The vectors have unit lengths
        The vector pair is right-hand
        '''
        spike_v = self.vertices[spike_idx]
        prev_v =  self.vertices[prev_idx]
        next_v =  self.vertices[next_idx]

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
    def _segment_closest_point_inside_sector(segment, sector):
        dir1, tip, dir2 = sector

        def pt_inside(pt):
            rel_pt = pt - tip
            return vec.cross2(dir1, rel_pt) > 0 and vec.cross2(dir2, rel_pt) < 0

        # make ray-like line
        line = (segment[0], segment[1] - segment[0])
        (linepara1, raypara1) = Geom2.lines_intersect(line, (tip, dir1))
        (linepara2, raypara2)= Geom2.lines_intersect(line, (tip, dir2))

        # check whether line containing segment passes through sector:
        line_through_sector = raypara1[1] > 0 or raypara1[1] > 0

        if not line_through_sector:
            return None, None

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
        else if len(intersect_params) == 2:
            candid_params = list(sorted(intersect_params))

        # no intersections => entirely inside or outside
        else:
            if pt_inside(segment[0]):
                candid_params = [0, 1]
            else:
                return None, None

        # find point on line closest to tip
        projpara = Geom2.project_to_line(tip, line)

        if candid_params[0] < projpara and projpara < candid_params[1]:
            candid_params.append(projpara)

        def para_to_pt(para):
            if para == 0:
                return segment[0]
            else if para == 1:
                return segment[1]
            else:
                return line[0] + (para * line[1])

        distances = ((tip - para_to_pt(para)).normSq() for para in candid_params)

        return min(izip(candid_params, distances), key = itemgetter(1))
