import math
import random
import yaml

from .vector2 import Vector2, ZeroSegmentError


def plus_wrap(pos, num):
			return 0 if pos == num - 1 else pos + 1

def minus_wrap(pos, num):
	return num - 1 if pos == 0 else pos - 1

class Mesh2d:

	def __init__(self, index_buffers, vertex_buffer):
		self.polygons = index_buffers
		self.vertices = vertex_buffer


class Bbox:
	def __init__(self, vertices, indices):
		vrt = vertices
		ind = indices
		self.xmin = self.xmax = vrt[ind[0]].x
		self.ymin = self.ymax = vrt[ind[0]].y

		for i in ind:
			v = vrt[i]
			if v.x < self.xmin:
				self.xmin = v.x
			elif v.x > self.xmax:
				self.xmax = v.x

			if v.y < self.ymin:
				self.ymin = v.y
			elif v.y > self.ymax:
				self.ymax = v.y


		def point_inside(self, point):
			return point.x < self.xmax and point.x > self.xmin and \
				point.y < self.ymax and point.y > self.ymin





class Polygon2d:
	def __init__(self, vertices, indices):
		if self.check_ccw(vertices, indices):
			self.indices = indices[:]
		else:
			self.indices = list(reversed(indices))

		# self.vertices = vertices[:]
		self.vertices = vertices

		self.bbox = Bbox(vertices, indices)


	# def point_inside(self, point):
	# 	if not self.bbox.point_inside(point): return False



	def outline_coordinates(self):
		crds = []

		for ind in self.indices:
			crds.append(self.vertices[ind].x)
			crds.append(self.vertices[ind].y)
		crds.append(self.vertices[self.indices[0]].x)
		crds.append(self.vertices[self.indices[0]].y)
		return crds


	def break_into_convex(self, polys, threshold = 0.0, canvas=None):
		portals = self.get_portals(threshold=threshold)
		print "num portals = {0}".format(len(portals))

		# draw portals
		if canvas is not None:
			for portal in portals:
				v1 = self.vertices[portal['start_index']]
				v2 = portal['end_point']

				canvas.create_line(v1.x, v1.y, v2.x, v2.y, fill='red', width=2)

		# break polygons by portals recursively
		ysize = self.bbox.ymax - self.bbox.ymin
		self._break_in_two(polys, portals, canvas, 1, ysize)



	def draw_self(self, canvas, displacement):
		coords = []
		for i in self.indices:
			vrt = self.vertices[i] + displacement
			coords.append(vrt.x)
			coords.append(vrt.y)

		rnd = lambda: random.randint(0,255)
		color = '#%02X%02X%02X' % (rnd(),rnd(),rnd())
		canvas.create_polygon(coords, fill=color)



	def add_vertex_to_edge(self, vertex, edge):
		e1_i = edge[0]
		e2_i = edge[1]

		e1_pos = self.indices.index(e1_i)
		e2_pos = self.indices.index(e2_i)

		

		# e1_pos and e2_pos can either differ by 1
		# or loop around the index buffer
		if e1_pos > e2_pos: e1_pos, e2_pos = e2_pos, e1_pos

		if e1_pos == e2_pos: raise ValueError("Adding vertex to edge: invalid edge")

		if e2_pos - e1_pos > 1:
			if e1_pos != 0: raise ValueError("Adding vertex to edge: invalid edge")
			insert_at = e2_pos + 1

		else:
			insert_at = e2_pos

		new_vert_index = len(self.vertices)
		self.vertices.append(vertex)
		self.indices.insert(insert_at, new_vert_index)
		return new_vert_index

	


	def _split_index_buffer(self, index_1, index_2):
		buffers = ([], [])
		switch = 0

		for index in self.indices:
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




	def _break_in_two(self, parts, portals, canvas, level, ysize):
		
		poly1 = None
		poly2 = None

		disp = Vector2(0, ysize * level)

		self.draw_self(canvas, disp)

		# iterate over portals trying to find the first portal that belongs to this polygon
		for portal in portals:

			piece1 = []
			piece2 = []

			start_i = portal['start_index']
			end_i = portal['end_index'] # may be None

			# if this portal starts outside this polygon
			if not start_i in self.indices:
				continue

			# if portal ends in a new vertex, add this vertex to the polygon
			if end_i is None:
				new_vrt = portal['end_point']
				intersection = self.trace_ray(self.vertices[start_i], new_vrt)

				if intersection is None:
					continue

				else:
					op_edge = intersection[1]
				
					end_i = self.add_vertex_to_edge(new_vrt, op_edge)
					if canvas is not None:
						sz = 3
						new_vrt = self.vertices[end_i]
						canvas.create_oval(new_vrt.x - sz, new_vrt.y - sz, new_vrt.x + sz, new_vrt.y + sz, fill='green')
					portal['end_index'] = end_i


			# find positions of starting and ending vertices of the portal
			start_pos = None
			end_pos = None

			# print 'portal starts at {0}'.format(start_i)
			# print 'portal ends   at {0}'.format(end_i)
			# print "indices : {0}".format(self.indices)
			
			# split index buffer of the polygon using the portal
			piece1, piece2 = self._split_index_buffer(start_i, end_i)

			# print "piece 1: {0}".format(piece1)
			# print "piece 2: {0}\n\n".format(piece2)

			# if this portal is actually an edge, skip it
			if len(piece1) < 3 or len(piece2) < 3:
				continue

			# create 2 new polygons
			poly1 = Polygon2d(self.vertices, piece1)
			poly2 = Polygon2d(self.vertices, piece2)

			# break them recursively
			poly1._break_in_two(parts, portals, canvas, level+1, ysize)
			poly2._break_in_two(parts, portals, canvas, level+1, ysize)
			return

		# if we did not find any portals, this polygon must be convex
		# append it to the list
		parts.append(self)
		return





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


	@staticmethod
	def check_ccw(vertices, indices):
		return Polygon2d.signed_area(vertices, indices) > 0


	def get_triangle_coords(self, triangle):

		v1 = self.vertices[triangle[0]]
		v2 = self.vertices[triangle[1]]
		v3 = self.vertices[triangle[2]]
		return [v1.x, v1.y, v2.x, v2.y, v3.x, v3.y, v1.x, v1.y]


	# @staticmethod
	# def triangulate(vertices):

	# 	# make index array:
	# 	indices = range(len(vertices))
	# 	triangles = []
	# 	diagonals = []
	# 	while len(indices) > 2:
	# 		ear = Polygon2d.find_ear(indices, vertices)
	# 		triangles.append([ear[0], ear[1], ear[2]])

	# 		diagonals.append([(ear[0], ear[2])])

	# 		# chop off ear:
	# 		indices.remove(ear[1])

	# 	return triangles, diagonals


				
	# @staticmethod
	# def find_ear(indices, vertices):

	# 	vrt = vertices
	# 	num_indices = len(indices)
					
	# 	cur_pos = 0

	# 	ear_found = False
	# 	while not ear_found:
	# 		next_pos = plus_wrap(cur_pos, num_indices)
	# 		prev_pos = minus_wrap(cur_pos, num_indices)

	# 		cur_ind = indices[cur_pos]
	# 		next_ind = indices[next_pos]
	# 		prev_ind = indices[prev_pos]

	# 		if Vector2.are_points_ccw(vrt[prev_ind], vrt[cur_ind], vrt[next_ind]):
	# 			# corner is convex
	# 			# check if any vertices are inside the corner
	# 			ear_found = True

	# 			for diag_pos in range(num_indices):

	# 				diag_ind = indices[diag_pos]

	# 				if diag_ind != cur_ind and diag_ind != next_ind and diag_ind != prev_ind:
						
	# 					if Vector2.point_inside(
	# 						vrt[diag_ind],
	# 						vrt[prev_ind],
	# 						vrt[cur_ind],
	# 						vrt[next_ind]):

	# 						ear_found = False
	# 						# ear has a vertex inside, continue search
	# 						cur_pos = plus_wrap(cur_pos, num_indices)
	# 						break
			
	# 		else:
	# 			# corner is concave, continue search
	# 			cur_pos = plus_wrap(cur_pos, num_indices)
	# 	return [prev_ind, cur_ind, next_ind]




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
			if not Vector2.are_points_ccw(vrt[cur_ind], vrt[next_ind], vrt[nnext_ind]):
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
		"""
		sz = 5
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

			portal = {}
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

						new_portals.append(second_portal)

			for new_portal in new_portals:

				if portal['start_index'] is None:
					print 'start_index is None'

				if portal['end_index'] is None:
					print 'end_index is None (it\'s ok)'

				if portal['end_point'] is None:
					print 'end_point is None'


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


				if closest_dist is None or candid_dist < closest_dist:
					closest_dist = candid_dist
					closest_pt = candid_pt
					closest_portal = portal

		return closest_portal, closest_pt, closest_dist 




	def _inside_cone(self, vert, left_v, tip_v, right_v):
			return Vector2.are_points_ccw(tip_v, right_v, vert) and not \
				Vector2.are_points_ccw(tip_v, left_v, vert)



	def _segment_inside_cone(self, seg1, seg2, left, tip, right):
		v1_inside = self._inside_cone(seg1, left, tip, right)
		v2_inside = self._inside_cone(seg2, left, tip, right)
		# if at least one vertex is inside the cone:
		if v1_inside or v2_inside:
			return True
		# if none of the endpoints is inside cone, the middle of the segment might still be:
		else:
			if Vector2.segment_crosses_ray(seg1, seg2, tip, left) and \
			Vector2.segment_crosses_ray(seg1, seg2, tip, right):
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
		proj = Vector2.project_to_line(tip, seg1, seg2)

		dist = None
		closest_pt = None
		# if projected point is inside anticone:
		if self._inside_cone(proj, left, tip, right):

			# 1st case: projected point is inside anticone and inside segment:
			if Vector2.point_between(proj, seg1, seg2):
				dist = Vector2.distance(tip, proj)
				closest_pt = proj

			# 2nd case: projected point is inside anticone and outside segment:
			else:
				dist1 = Vector2.distance(seg1, tip)
				dist2 = Vector2.distance(seg2, tip)
				dist = min(dist1, dist2)
				closest_pt = seg1 if dist1 < dist2 else seg2

		# 3rd case: projected point is outside anticone:
		else:
			inters_left = Vector2.where_segment_crosses_ray(seg1, seg2, tip, left)
			inters_right = Vector2.where_segment_crosses_ray(seg1, seg2, tip, right)

			dist_left = Vector2.distance(tip, inters_left) if inters_left is not None else 9999999.0
			dist_right = Vector2.distance(tip, inters_right) if inters_right is not None else 9999999.0

			dist = min(dist_left, dist_right)
			closest_pt = inters_left if dist_left < dist_right else inters_right

		return closest_pt, dist