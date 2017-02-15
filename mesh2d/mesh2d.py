import math
import random
from collections import deque
from operator import itemgetter

from .vector2 import Vector2, ZeroSegmentError


def plus_wrap(pos, num):
            return 0 if pos == num - 1 else pos + 1

def minus_wrap(pos, num):
    return num - 1 if pos == 0 else pos - 1



class Polygon2d(object):
    def __init__(self, vertices, indices):

        # need to find self-intersections
        if Polygon2d.check_ccw(vertices, indices):
            self.outline = indices[:]
        else:
            self.outline = indices[::-1]

        self.vertices = vertices[:]
        sinters = self.find_self_intersections()



    def find_self_intersections(self):
        segs = Polygon2d.get_segments(self.outline)
        sinters = []
        for i in range(len(segs)):
            for j in range(i+2, len(segs)):
                seg1 = segs[i]
                seg2 = segs[j]

                seg11 = self.vertices[seg1[0]]
                seg12 = self.vertices[seg1[1]]

                seg21 = self.vertices[seg2[0]]
                seg22 = self.vertices[seg2[1]]

                seg_x = Vector2.where_segments_cross(
                    seg11, seg12, seg21, seg22)

                if seg_x is not None:
                    sinters.append((seg1, seg2, seg_x))
        return sinters



    def add_vertex_to_outline(self, vertex, edge):
        e1_i = edge[0]
        e2_i = edge[1]

        e1_pos = self.outline.index(e1_i)
        e2_pos = self.outline.index(e2_i)

        # e1_pos and e2_pos can either differ by 1
        # or loop around the index buffer
        if e1_pos > e2_pos: e1_pos, e2_pos = e2_pos, e1_pos

        if e1_pos == e2_pos: raise ValueError("Adding vertex to outline: invalid edge")

        if e2_pos - e1_pos > 1:
            if e1_pos != 0: raise ValueError("Adding vertex to outline: invalid edge")
            insert_at = e2_pos + 1

        else:
            insert_at = e2_pos

        new_vert_index = len(self.vertices)
        self.vertices.append(vertex)
        self.outline.insert(insert_at, new_vert_index)
        return new_vert_index




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
    def get_segments(indices):
        segs = []
        for loc in range(len(indices) - 1):
            segs.append((indices[loc], indices[loc + 1]))
        segs.append((indices[-1], indices[0]))
        return segs



    @staticmethod
    def check_ccw(vertices, indices):
        return Polygon2d.signed_area(vertices, indices) > 0



    @staticmethod
    def signed_area(vertices, indices):
        area = 0.0
        for i in range(len(indices) - 1):
            ind1 = indices[i]
            ind2 = indices[i+1]

            vert1 = vertices[ind1]
            vert2 = vertices[ind2]

            area += (vert1.x - vert2.x) * (vert1.y + vert2.y)

        # wrap for last segment:
        vert1 = vertices[indices[-1]]
        vert2 = vertices[indices[0]]

        area += (vert1.x - vert2.x) * (vert1.y + vert2.y)
        return area / 2.0







class Mesh2d(Polygon2d):
    def __init__(self, vertices, indices):
        super(Mesh2d, self).__init__(vertices, indices)

        # just an alias (need to get rid of it later)
        self.indices = self.outline

        self.pieces = []
        self.portals = []


    def dump(self):
        return self.__dict__


    def get_pieces_as_meshes(self):
        meshes = []
        for piece in self.pieces:
            meshes.append(Mesh2d(self.vertices, piece))
        return meshes



    def outline_coordinates(self, indices=None):
        if indices is None: indices = self.indices
        crds = []

        for ind in indices:
            crds.append(self.vertices[ind].x)
            crds.append(self.vertices[ind].y)
        crds.append(self.vertices[indices[0]].x)
        crds.append(self.vertices[indices[0]].y)
        return crds




    def break_into_convex(self, threshold = 0.0, canvas = None):
        portals = self.get_portals(threshold=threshold)

        # draw portals
        if canvas is not None:
            for portal in portals:
                v1 = self.vertices[portal['start_index']]
                v2 = portal['end_point']

                canvas.create_line(v1.x, v1.y, v2.x, v2.y, fill='red', width=1)

        '''
        For all the portals that require creating new vertices, create new vertices.
        Multiple portals may have the same endpoint. If we have 5 portals that converge
        to the same endpoint, we only want to create the endpoint once.
        This is why some portals have a 'parent_portal' attribute - we create the endpoint
        for the parent portal only, and all the other portals use its index.
        '''

        for portal in portals:

            # draw portal endpoints
            if canvas is not None:
                sz = 3
                new_vrt = portal['end_point']
                canvas.create_oval(
                    new_vrt.x - sz, new_vrt.y - sz,
                    new_vrt.x + sz, new_vrt.y + sz,
                    fill='green')


            if portal['end_index'] is None and 'parent_portal' not in portal:

                new_vrt = portal['end_point']
                start_i = portal['start_index']
                intersection = self.trace_ray(self.vertices[start_i], new_vrt)

                if intersection is None:
                    continue

                else:
                    op_edge = intersection[1]
                
                    end_i = self.add_vertex_to_outline(new_vrt, op_edge)

                    portal['end_index'] = end_i

        # now go through portals again to set child portals' end indexes
        for portal in portals:
            if portal['end_index'] is None and 'parent_portal' in portal:
                portal['end_index'] = portal['parent_portal']['end_index']


        # Now break the mesh outline into convex pieces

        # queue of pieces
        piece_q = deque()

        # append a copy of the entire outline
        piece_q.append(self.indices[:])

        while len(piece_q) > 0:
            piece = piece_q.popleft()
            piece1, piece2, new_portal = Mesh2d._break_in_two(piece, portals)

            # if could not split this piece, finalize it
            if piece1 is None:
                self.pieces.append(piece)

            # otherwise add new pieces to the queue
            else:
                piece_q.append(piece1)
                piece_q.append(piece2)
                self.portals.append(new_portal)

        return portals



    @staticmethod
    def _break_in_two(indices, portals):
        # iterate over portals trying to find the first portal that belongs to this polygon
        for portal in portals:

            # if this portal has already been created, skip it
            if 'created' in portal: continue

            piece1 = []
            piece2 = []

            start_i = portal['start_index']
            end_i = portal['end_index']

            # if this portal starts outside of this polygon, skip it
            if start_i not in indices or end_i not in indices:
                continue

            # split index buffer of the outline of this piece using the portal
            piece1, piece2 = Polygon2d._split_index_buffer(indices, start_i, end_i)

            # if this portal is actually an edge, skip it
            if len(piece1) < 3 or len(piece2) < 3:
                continue

            # mark this portal as created
            portal['created'] = True

            return piece1, piece2, (start_i, end_i)

        # if we did not find any portals to split, this piece must be convex
        return None, None, None




    


    def get_triangle_coords(self, triangle):

        v1 = self.vertices[triangle[0]]
        v2 = self.vertices[triangle[1]]
        v3 = self.vertices[triangle[2]]
        return [v1.x, v1.y, v2.x, v2.y, v3.x, v3.y, v1.x, v1.y]



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

        indices = self.indices
        num_indices = len(indices)
        # find spikes:
        spikes = []
        for cur_pos in range(num_indices):
            next_pos = plus_wrap(cur_pos, num_indices)
            prev_pos = minus_wrap(cur_pos, num_indices)

            next_ind = indices[next_pos]
            prev_ind = indices[prev_pos]
            cur_ind = indices[cur_pos]
            
            prev_v = vrt[prev_ind]
            cand_v = vrt[cur_ind]
            next_v = vrt[next_ind]

            signed_area = Vector2.double_signed_area(prev_v, cand_v, next_v)
            if signed_area < 0.0:

                side1 = cand_v - prev_v
                side2 = next_v - cand_v
                external_angle = Vector2.angle(side1, side2)
                external_angle = external_angle*180.0 / math.pi

                if external_angle > threshold:
                    spikes.append((prev_ind, cur_ind, next_ind))
        return spikes




    def get_portals(self, threshold = 0.0, tolerance = 0.000001, canvas=None):

        """
        This function uses algorithm from R. Oliva and N. Pelechano - 
        Automatic Generation of Suboptimal NavMeshes
        This is work in progress
        The endpoints of portals can be new vertices,
        but they are guaranteed to lie on the polygon boundary (not inside the polygon)

        TODO This function is ridiculously complex right now
        """
        spikes = self.find_spikes(threshold)
        portals = []

        for spike in spikes:

            prev_i = spike[0]
            spike_i = spike[1]
            next_i = spike[2]

            conevec1, conevec2 = self.get_anticone(prev_i, spike_i, next_i, threshold)
            tip = self.vertices[spike_i]
            left = tip + conevec2
            right = tip + conevec1


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
            # closest vertex - not always (there might be no vertices inside the anticone)
            # closest portal - not always (there might be no portals inside the anticone
            # or no portals at all)

            portal = {
                'start_index': None,
                'end_index': None,
                'end_point': None
            }

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

                    # if portal_end_i is None: portal['parent_portal'] = closest_portal

                # if closest point is not one of the endpoints, make 2 portals to both endpoints
                else:
                
                    start_inside = self._inside_cone(portal_start_p, left, tip, right)
                    end_inside = self._inside_cone(portal_end_p, left, tip, right)

                    # if both portal endpoints are inside
                    if start_inside and end_inside:
                        # pick the closest one
                        dst_start = Vector2.distance(portal_start_p, tip)
                        dst_end = Vector2.distance(portal_end_p, tip)

                        if dst_start <= dst_end:
                            portal['end_index'] = portal_start_i
                            portal['end_point'] = portal_start_p
                        else:
                            portal['end_index'] = portal_end_i # this might be None
                            portal['end_point'] = portal_end_p


                    # if only the start of other portal is inside:
                    elif start_inside:
                        portal['end_index'] = portal_start_i
                        portal['end_point'] = portal_start_p

                    # if only the end of other portal is inside:
                    elif end_inside:
                        portal['end_index'] = portal_end_i # this might be None
                        portal['end_point'] = portal_end_p

                    # if none of the portal endpoints is inside, create 2 portals to both ends:
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

        indices = self.indices

        closest_ind = None
        closest_dst = None

        for cur_i in indices:
            cur_v = vrt[cur_i]

            if cur_v != tip:

                if self._inside_cone(vrt[cur_i], left, tip, right):
                    dst = Vector2.distance(vrt[cur_i], tip)
                    if closest_dst is None or dst < closest_dst:
                        closest_dst = dst
                        closest_ind = cur_i

        return closest_ind, closest_dst



    def find_closest_edge(self, left, tip, right):
        vrt = self.vertices

        indices = self.indices
        num_indices = len(indices)

        closest_pt = None
        closest_dist = None
        closest_edge = None

        for pos in range(num_indices):
            # here we assume that indices go counter-clockwise,
            # and seg2 comes after seg1
            seg_i1 = indices[pos]
            seg_i2 = indices[plus_wrap(pos, num_indices)]

            seg1 = vrt[seg_i1]
            seg2 = vrt[seg_i2]

            # skip edges adjacent to the tip of the cone:
            if seg2 == tip:
                continue
            if seg1 == tip:
                continue

            
            if self._segment_inside_cone(seg1, seg2, left, tip, right):

                candid_pt, candid_dist = self.segment_closest_point_inside_cone(seg1, seg2, left, tip, right)

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

            if self._segment_inside_cone(port1, port2, left, tip, right):

                candid_pt, candid_dist = self.segment_closest_point_inside_cone(port1, port2, left, tip, right)

                # # portal's closest point can only be one of the endpoints
                # if candid_pt != port1 and candid_pt != port2:

                    # start_inside = self._inside_cone(port1, left, tip, right)
                    # end_inside = self._inside_cone(port2, left, tip, right)

                    # endpts_inside = (
                    #     (port1, self._inside_cone(port1, left, tip, right)),
                    #     (port2, self._inside_cone(port2, left, tip, right))
                    # )

                    # # calculate distances to endpoints that are inside cone
                    # endpt_dists = tuple((endpt, Vector2.distance(endpt, tip)) \
                    #     for (endpt, inside) in endpts_inside if inside)

                    # # if more than one endpoint is inside cone, picke the closest one
                    # if len(endpt_dists) > 0:
                    #     (candid_pt, candid_dist) = min(endpt_dists, key=itemgetter(1))
                    # # if both endpoints are outside cone, leave it like this (we'll create 2 portals later)
                                           

                # update closest portal
                if closest_dist is None or candid_dist < closest_dist:
                    closest_dist = candid_dist
                    closest_pt = candid_pt
                    closest_portal = portal

        return closest_portal, closest_pt, closest_dist




    def _inside_cone(self, vert, left_v, tip_v, right_v):
        '''
        Check if vert is inside cone defined by left_v, tip_v, right_v.
        If vert is on the border of the cone, method returns False.
        '''
        right_area = Vector2.double_signed_area(tip_v, right_v, vert)
        left_area = Vector2.double_signed_area(tip_v, left_v, vert)
        if right_area == 0 or left_area == 0: return False
        return right_area > 0 and left_area < 0



    def _segment_inside_cone(self, seg1, seg2, left, tip, right):
        v1_inside = self._inside_cone(seg1, left, tip, right)
        v2_inside = self._inside_cone(seg2, left, tip, right)
        # if at least one vertex is inside the cone:
        if v1_inside or v2_inside:
            return True
        # if none of the endpoints is inside cone, the middle of the segment might still be:
        else:
            left_x = Vector2.where_segment_crosses_ray(seg1, seg2, tip, left)
            right_x = Vector2.where_segment_crosses_ray(seg1, seg2, tip, right)

            if left_x is not None and right_x is not None:
                if left_x == tip and right_x == tip: return False
                return True
            else:
                return False




    def get_anticone(self, prev_ind, spike_ind, next_ind, threshold=0.0):
        '''
        Returns 2 vectors that define the cone rays.
        The vectors have unit lengths
        The vector pair is right-hand
        '''
        vrt = self.vertices
        num_vrt = len(self.vertices)
        # next_ind = plus_wrap(spike_ind, num_vrt)
        # prev_ind = minus_wrap(spike_ind, num_vrt)

        spike_v = vrt[spike_ind]
        prev_v = vrt[prev_ind]
        next_v = vrt[next_ind]


        vec1 = spike_v - prev_v
        vec2 = spike_v - next_v
        vec1 /= vec1.length()
        vec2 /= vec2.length()

        anticone_angle = math.acos(vec1.dot_product(vec2))
        
        # degrees to radian
        clearance = math.pi * threshold / 180.0
        anticone_angle_plus = anticone_angle + clearance

        

        # limit anticone opening to 180 degrees:
        anticone_angle_plus = anticone_angle_plus if anticone_angle_plus < math.pi else math.pi

        clearance = anticone_angle_plus - anticone_angle

        clearance = clearance / 2.0

        cosine = math.cos(clearance)
        sine = math.sin(clearance)
        mtx = [cosine, sine, -sine, cosine]
        vec1_new = Vector2.mul_mtx(mtx, vec1)
        mtx = [cosine, -sine, sine, cosine]
        vec2_new = Vector2.mul_mtx(mtx, vec2)

        return vec1_new, vec2_new


    def trace_ray(self, ray1, ray2):
        num_indices = len(self.indices)
        intersections = []
        for pos1 in range(num_indices):
            pos2 = plus_wrap(pos1, num_indices)
            seg1 = self.vertices[self.indices[pos1]]
            seg2 = self.vertices[self.indices[pos2]]

            # ignore edges that are adjacent to the ray's starting point
            if seg1 == ray1 or seg2 == ray1:
                continue

            inter_pt = Vector2.where_segment_crosses_ray(seg1, seg2, ray1, ray2)
            if inter_pt:
                # append a tuple of a vertex and an edge
                intersections.append((inter_pt, (self.indices[pos1], self.indices[pos2])))

        if len(intersections) == 0:
            return None

        # find closest intersection:
        min_dst = Vector2.distance(ray1, intersections[0][0])
        closest_inters = intersections[0]

        for inters in intersections:
            dst = Vector2.distance(ray1, inters[0])
            if dst < min_dst:
                min_dst = dst
                closest_inters = inters

        return closest_inters




    def segment_closest_point_inside_cone(self, seg1, seg2, left, tip, right):
        '''
        Find point on the segment that lies inside the cone that is closest to the cone tip.
        This method assumes that the segment is at least partially inside the cone.
        Use _segment_inside_cone method to verify that before calling this method.
        '''

        '''
        Need to find closest point that is both inside cone and inside segment
        Points of interest:
        - proj point (check if inside cone AND inside segment)
        - endpoint1 (check if inside cone)
        - endpoint2 (check if inside cone)
        - intersect1 (check if not None)
        - intersect2 (check if not None)
        '''
        pts_of_interest = []
        proj_pt = Vector2.project_to_line(tip, seg1, seg2)

        if self._inside_cone(proj_pt, left, tip, right) and \
            Vector2.point_between(proj_pt, seg1, seg2):
            pts_of_interest.append(proj_pt)

        if self._inside_cone(seg1, left, tip, right):
            pts_of_interest.append(seg1)
        if self._inside_cone(seg2, left, tip, right):
            pts_of_interest.append(seg2)

        inters_left = Vector2.where_segment_crosses_ray(seg1, seg2, tip, left)
        inters_right = Vector2.where_segment_crosses_ray(seg1, seg2, tip, right)

        if inters_left is not None:
            pts_of_interest.append(inters_left)
        if inters_right is not None:
            pts_of_interest.append(inters_right)


        pts_of_interest = [(poi, Vector2.distance(poi, tip)) for poi in pts_of_interest]
        (closest_pt, dist) = min(pts_of_interest, key=itemgetter(1))

        return closest_pt, dist
