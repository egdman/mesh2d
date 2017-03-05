import math
import random

from collections import deque
from operator import itemgetter
from itertools import chain
from copy import deepcopy
from rtree import index

from .vector2 import Vector2, ZeroSegmentError
from .utils import debug_draw_room


def print_p(p):
    print("start : {}".format(p['start_index']))
    print("end   : {}, {}".format(p['end_index'], p['end_point']))
    print ("----------------------------------------------")




def plus_wrap(pos, num):
    return 0 if pos == num - 1 else pos + 1

def minus_wrap(pos, num):
    return num - 1 if pos == 0 else pos - 1


def chain_index_buffers(loops):
    '''
    Return vertex indices of the given loops as a flat list.
    '''
    all_ind = []
    for loop in loops: all_ind.extend(loop)
    return all_ind



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
        seg11 = self.vertices[seg1[0]]
        seg12 = self.vertices[seg1[1]]
        seg21 = self.vertices[seg2[0]]
        seg22 = self.vertices[seg2[1]]
        return Vector2.where_segments_cross_exclusive(
                    seg11, seg12, seg21, seg22)



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
                key = lambda elem: Vector2.distance(elem[1], self.vertices[edge[0]]))

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

                inters = Vector2.lines_intersect(
                    cur_v, next_v,
                    Vector2(0, 0), Vector2(1, 0)).intersection

                if inters.x > 0: num_inters += 1

        return num_inters%2 > 0





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
        return Vector2.poly_signed_area((vertices[idx] for idx in indices))



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


    def get_rooms_as_meshes(self):
        meshes = []
        for room in self.rooms:
            meshes.append(Mesh2d(self.vertices, room))
        return meshes



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
            if portal['end_index'] is None and 'parent_portal' not in portal:

                new_vrt = portal['end_point']
                start_i = portal['start_index']

                intersection = self.trace_ray(self.vertices[start_i], new_vrt)

                if intersection is None:
                    raise RuntimeError("Ray casting failed to find portal endpoint")

                op_edge = intersection[1]

                end_i = self.add_vertex_to_border(new_vrt, op_edge)

                portal['end_index'] = end_i

                if cv:
                    vrt = self.vertices[end_i]
                    cv.create_oval(vrt.x - 3, vrt.y - 3, vrt.x + 3, vrt.y + 3, fill='green')


        if cv:
            for idx in cv.find_all(): cv.delete(idx)
            debug_draw_room(self, [self.outline] + self.holes, cv)


        # now go through portals again to set child portals' end indexes
        for portal in portals:
            if portal['end_index'] is None and 'parent_portal' in portal:
                portal['end_index'] = portal['parent_portal']['end_index']


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
            if 'created' in portal: continue

            room1 = []
            room2 = []

            start_i = portal['start_index']
            end_i = portal['end_index']

            # if this portal starts outside of this polygon, skip it
            if start_i not in indices or end_i not in indices:
                continue

            # split index buffer of the outline of this room using the portal
            # room1, room2 = Polygon2d._split_index_buffer(indices, start_i, end_i)

            room1, room2 = self._split_border(loops, start_i, end_i)

            if room1 is None and room2 is None: continue

            # mark this portal as created
            portal['created'] = True

            return room1, room2, (start_i, end_i)

        # if we did not find any portals to split, this room must be convex
        return None, None, None


    # def get_triangle_coords(self, triangle):

    #     v1 = self.vertices[triangle[0]]
    #     v2 = self.vertices[triangle[1]]
    #     v3 = self.vertices[triangle[2]]
    #     return [v1.x, v1.y, v2.x, v2.y, v3.x, v3.y, v1.x, v1.y]



    @staticmethod
    def check_convex(indices, vertices):

        vrt = vertices
        num_indices = len(indices)

        is_convex = True

        for cur_pos in range(num_indices):
            next_pos = plus_wrap(cur_pos, num_indices)
            nnext_pos = plus_wrap(next_pos, num_indices)

            cur_ind = indices[cur_pos]
            next_ind = indices[next_pos]
            nnext_ind = indices[nnext_pos]
            if Vector2.double_signed_area(vrt[cur_ind], vrt[next_ind], vrt[nnext_ind]) < 0:
                is_convex = False
                break

        return is_convex



    def find_spikes(self, threshold = 0.0):
        vrt = self.vertices
        borders = [self.outline] + self.holes
        segments = Polygon2d.get_segments(borders)
        num_segs = len(segments)

        # find spikes:
        spikes = []
        start_seg = segments[0]
        for cur_pos in range(num_segs):

            cur_seg = segments[cur_pos]
            next_seg = segments[(cur_pos + 1) % num_segs]

            # check if we are at the end of the border
            if cur_seg[1] != next_seg[0]:
                next_seg, start_seg = start_seg, next_seg

            prev_ind = cur_seg[0]
            cur_ind = cur_seg[1]
            next_ind = next_seg[1]

            prev_v = vrt[prev_ind]
            cand_v = vrt[cur_ind]
            next_v = vrt[next_ind]

            signed_area = Vector2.double_signed_area(prev_v, cand_v, next_v)
            if signed_area < 0.0:

                side1 = cand_v - prev_v
                side2 = next_v - cand_v
                try:
                    external_angle = Vector2.angle(side1, side2)

                except ValueError as ve:
                    print(ve)
                    print("tried to find angle between vectors: {}, {}".format(side1, side2))
                    raise ve

                external_angle = external_angle*180.0 / math.pi

                if external_angle > threshold:
                    spikes.append((prev_ind, cur_ind, next_ind))

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

            prev_i = spike[0]
            spike_i = spike[1]
            next_i = spike[2]

            sectorvec1, sectorvec2 = self.get_sector(prev_i, spike_i, next_i, threshold)
            tip = self.vertices[spike_i]
            left = tip + sectorvec2
            right = tip + sectorvec1

            # find closest edge
            closest_seg, closest_seg_point, closest_seg_dst = \
                self.find_closest_edge(left, tip, right)

            # find closest vertex
            closest_vert_i, closest_vert_dst = self.find_closest_vert(left, tip, right)

            # find closest portal
            closest_portal, closest_portal_point, closest_portal_dst = \
                self.find_closest_portal(left, tip, right, portals)

            # remove tiny difference between points
            if closest_portal is not None:
                if closest_portal_dst < tolerance:
                    closest_portal_point.snap_to(tip)
                    closest_portal_dst = 0.0



            # closest edge always exists
            # closest vertex - not always (there might be no vertices inside the sector)
            # closest portal - not always (there might be no portals inside the sector
            # or no portals at all)
            portal = {}

            # we might want to add a second portal later
            new_portals = [portal]

            portal['start_index'] = spike_i
            portal['end_index'] = None # optional, might be added later

            portal['end_point'] = closest_seg_point

            closest_dst = closest_seg_dst

            # check if there is a vertex closer or equally close than the closest edge (prefer vertex)
            if closest_vert_dst is not None and closest_vert_dst <= closest_dst:
                closest_dst = closest_vert_dst
                portal['end_index'] = closest_vert_i
                portal['end_point'] = self.vertices[closest_vert_i]
                

            # check if there is a portal closer than the previous closest element
            # TODO When attaching to an existing portal, need to reconsider the necessity of the older portal
            if closest_portal_dst is not None and closest_portal_dst < closest_dst:
                closest_dst = closest_portal_dst

                portal_start_i = closest_portal['start_index']
                portal_start_p = self.vertices[portal_start_i]

                portal_end_i = closest_portal['end_index'] # might be None
                portal_end_p = closest_portal['end_point']

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

                if closest_portal_point == portal_start_p:
                    portal['end_index'] = portal_start_i
                    portal['end_point'] = portal_start_p
                    # new_portal_endpoint = [portal_v1]

                elif closest_portal_point == portal_end_p:
                    portal['end_index'] = portal_end_i # this might be None
                    portal['end_point'] = portal_end_p

                
                # If closest point is not one of the endpoints,
                # we still create the portal(s) to one or two endpoints.
                
                else:

                    portal_endpoints = (
                        (portal_start_i, portal_start_p),
                        (portal_end_i, portal_end_p)
                    )

                    endpt_dists = tuple((idx, pos, Vector2.distance(pos, tip)) \
                        for (idx, pos) in portal_endpoints \
                        if self._inside_sector(pos, left, tip, right))

                    # if 1 or 2 points are inside sector, pick the closest one
                    if len(endpt_dists) > 0:
                        (cl_idx, cl_pos, cl_dist) = min(endpt_dists, key=itemgetter(2))
                        portal['end_index'] = cl_idx
                        portal['end_point'] = cl_pos


                    # if none of the portal endpoints is inside sector, create 2 portals to both ends:
                    else:
                        second_portal = {}
                        second_portal.update(portal)

                        portal['end_index'] = portal_start_i
                        portal['end_point'] = portal_start_p

                        second_portal['end_index'] = portal_end_i
                        second_portal['end_point'] = portal_end_p

                        # save reference to the portal that we are snapping to
                        second_portal['parent_portal'] = closest_portal
                        new_portals.append(second_portal)


                # save reference to the portal that we are snapping to
                portal['parent_portal'] = closest_portal

            for new_portal in new_portals:

                if not new_portal['end_point'] == tip:
                    portals.append(new_portal)

        return portals




    def find_closest_vert(self, left, tip, right):
        vrt = self.vertices

        borders = [self.outline] + self.holes

        closest_ind = None
        closest_dst = None

        for cur_i in chain(*borders):
            cur_v = vrt[cur_i]

            if cur_v == tip: continue

            if self._inside_sector_inclusive(vrt[cur_i], left, tip, right):
                dst = Vector2.distance(vrt[cur_i], tip)
                if closest_dst is None or dst < closest_dst:
                    closest_dst = dst
                    closest_ind = cur_i

        return closest_ind, closest_dst



    def find_closest_edge(self, left, tip, right):
        vrt = self.vertices
        borders = [self.outline] + self.holes

        segments = Polygon2d.get_segments(borders)

        closest_pt = None
        closest_dist = None
        closest_edge = None

        for seg in segments:
            seg_i1 = seg[0]
            seg_i2 = seg[1]

            seg1 = vrt[seg_i1]
            seg2 = vrt[seg_i2]

            # skip edges adjacent to the tip of the sector:
            if seg2 == tip:
                continue
            if seg1 == tip:
                continue

            candid_pt, candid_dist = self.segment_closest_point_inside_sector(seg1, seg2, left, tip, right)
            if candid_pt is None: continue

            if closest_dist is None or candid_dist < closest_dist:
                closest_dist = candid_dist
                closest_pt = candid_pt
                closest_edge = (seg_i1, seg_i2)

        return closest_edge, closest_pt, closest_dist



    def find_closest_portal(self, left, tip, right, portals):

        closest_portal = None
        closest_dist = None
        closest_pt = None

        for portal in portals:

            port1 = self.vertices[portal['start_index']]
            port2 = portal['end_point']

            candid_pt, candid_dist = self.segment_closest_point_inside_sector(port1, port2, left, tip, right)
            if candid_pt is None: continue

            # update closest portal
            if closest_dist is None or candid_dist < closest_dist:
                closest_dist = candid_dist
                closest_pt = candid_pt
                closest_portal = portal

        return closest_portal, closest_pt, closest_dist



    def _inside_sector(self, vert, left_v, tip_v, right_v):
        '''
        Check if vert is inside sector defined by left_v, tip_v, right_v.
        Exclusive.
        If vert is on the border of the sector, method returns False.
        If vert == the sector tip, method returns False
        '''
        right_area = Vector2.double_signed_area(tip_v, right_v, vert)
        left_area = Vector2.double_signed_area(tip_v, left_v, vert)
        if right_area == 0 or left_area == 0: return False
        return right_area > 0 and left_area < 0



    def _inside_sector_inclusive(self, vert, left_v, tip_v, right_v):
        right_area = Vector2.double_signed_area(tip_v, right_v, vert)
        left_area = Vector2.double_signed_area(tip_v, left_v, vert)

        if left_area == 0 and right_area >= 0: return True
        if right_area == 0 and left_area <= 0: return True
        return right_area > 0 and left_area < 0



    def get_sector(self, prev_ind, spike_ind, next_ind, threshold=0.0):
        '''
        Returns 2 vectors that define the sector rays.
        The vectors have unit lengths
        The vector pair is right-hand
        '''
        vrt = self.vertices
        num_vrt = len(self.vertices)

        spike_v = vrt[spike_ind]
        prev_v = vrt[prev_ind]
        next_v = vrt[next_ind]


        vec1 = spike_v - prev_v
        vec2 = spike_v - next_v
        vec1 /= vec1.length()
        vec2 /= vec2.length()

        sector_angle = math.acos(vec1.dot_product(vec2))
        
        # degrees to radian
        clearance = math.pi * threshold / 180.0
        sector_angle_plus = sector_angle + clearance

        
        # limit sector opening to 180 degrees:
        sector_angle_plus = sector_angle_plus if sector_angle_plus < math.pi else math.pi

        clearance = sector_angle_plus - sector_angle

        clearance = clearance / 2.0

        cosine = math.cos(clearance)
        sine = math.sin(clearance)
        mtx = [cosine, sine, -sine, cosine]
        vec1_new = Vector2.mul_mtx(mtx, vec1)
        mtx = [cosine, -sine, sine, cosine]
        vec2_new = Vector2.mul_mtx(mtx, vec2)

        return vec1_new, vec2_new




    def trace_ray(self, ray1, ray2):
        vrt = self.vertices
        borders = [self.outline] + self.holes
        segments = Polygon2d.get_segments(borders)

        intersections = []
        for seg in segments:
            seg_i1 = seg[0]
            seg_i2 = seg[1]

            seg1 = vrt[seg_i1]
            seg2 = vrt[seg_i2]
            # ignore edges that are adjacent to the ray's starting point
            if seg1 == ray1 or seg2 == ray1:
                continue

            inter_pt = Vector2.where_segment_crosses_ray(seg1, seg2, ray1, ray2)
            if inter_pt is not None:
                # append a tuple of a vertex and an edge
                intersections.append((inter_pt, (seg_i1, seg_i2)))


        if len(intersections) == 0:
            return None

        # find closest intersection:
        inter_dist = ((inter, Vector2.distance(ray1, inter[0])) for inter in intersections)
        res = min(inter_dist, key=itemgetter(1))[0]
        return res



    def segment_closest_point_inside_sector(self, seg1, seg2, left, tip, right):
        '''
        Find point on the segment that lies inside the sector that is closest to the sector tip.
        Returns (Vector2, distance).
        If the segment is entirely outside the sector, returns (None, None)
        '''

        '''
        Need to find closest point that is both inside sector and inside segment
        Points of interest:
        - proj point (check if inside sector AND inside segment)
        - endpoint1 (check if inside sector)
        - endpoint2 (check if inside sector)
        - intersect1 (check if not None)
        - intersect2 (check if not None)
        '''
        pts_of_interest = []
        proj_pt = Vector2.project_to_line(tip, seg1, seg2)

        if Vector2.point_between(proj_pt, seg1, seg2) and \
            self._inside_sector(proj_pt, left, tip, right):
            pts_of_interest.append(proj_pt)

        if self._inside_sector(seg1, left, tip, right):
            pts_of_interest.append(seg1)
        if self._inside_sector(seg2, left, tip, right):
            pts_of_interest.append(seg2)

        inters_left = Vector2.where_segment_crosses_ray(seg1, seg2, tip, left)
        inters_right = Vector2.where_segment_crosses_ray(seg1, seg2, tip, right)

        if inters_left is not None:
            pts_of_interest.append(inters_left)
        if inters_right is not None:
            pts_of_interest.append(inters_right)

        if len(pts_of_interest) == 0: return None, None

        pts_of_interest = list((poi, Vector2.distance(poi, tip)) for poi in pts_of_interest)

        # if all dists are 0, the segment is outside but touches the tip of the sector
        if sum((dist for (poi, dist) in pts_of_interest)) == 0: return None, None

        (closest_pt, dist) = min(pts_of_interest, key=itemgetter(1))
        return closest_pt, dist
