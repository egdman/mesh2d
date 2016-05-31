import math
from .vector2 import Vector2, ZeroSegmentError


def plus_wrap(pos, num):
			return 0 if pos == num - 1 else pos + 1

def minus_wrap(pos, num):
	return num - 1 if pos == 0 else pos - 1


class Polygon2d:
	def __init__(self, vertices):
		if self.check_ccw(vertices):
			self.vertices = vertices[:]
		else:
			self.vertices = list(reversed(vertices))

		tris, diags = self.triangulate(self.vertices)
		self.triangles = tris
		self.diagonals = diags
		self.convex_parts = self.triangles[:]



	@staticmethod
	def signed_area(vertices):
		area = 0.0
		for i in range(len(vertices) - 1):
			vert1 = vertices[i]
			vert2 = vertices[i+1]
			area += (vert1.x - vert2.x) * (vert1.y + vert2.y)
		# wrap for last segment:
		vert1 = vertices[-1]
		vert2 = vertices[0]
		area += (vert1.x - vert2.x) * (vert1.y + vert2.y)
		return area / 2.0


	@staticmethod
	def check_ccw(vertices):
		return Polygon2d.signed_area(vertices) > 0


	def get_triangle_coords(self, triangle):

		v1 = self.vertices[triangle[0]]
		v2 = self.vertices[triangle[1]]
		v3 = self.vertices[triangle[2]]
		return [v1.x, v1.y, v2.x, v2.y, v3.x, v3.y, v1.x, v1.y]


	@staticmethod
	def triangulate(vertices):

		# make index array:
		indices = range(len(vertices))
		triangles = []
		diagonals = []
		while len(indices) > 2:
			ear = Polygon2d.find_ear(indices, vertices)
			triangles.append([ear[0], ear[1], ear[2]])

			diagonals.append([(ear[0], ear[2])])

			# chop off ear:
			indices.remove(ear[1])

		return triangles, diagonals


				
	@staticmethod
	def find_ear(indices, vertices):

		vrt = vertices
		num_indices = len(indices)
					
		cur_pos = 0

		ear_found = False
		while not ear_found:
			next_pos = plus_wrap(cur_pos, num_indices)
			prev_pos = minus_wrap(cur_pos, num_indices)

			cur_ind = indices[cur_pos]
			next_ind = indices[next_pos]
			prev_ind = indices[prev_pos]

			if Vector2.are_points_ccw(vrt[prev_ind], vrt[cur_ind], vrt[next_ind]):
				# corner is convex
				# check if any vertices are inside the corner
				ear_found = True

				for diag_pos in range(num_indices):

					diag_ind = indices[diag_pos]

					if diag_ind != cur_ind and diag_ind != next_ind and diag_ind != prev_ind:
						
						if Vector2.point_inside(
							vrt[diag_ind],
							vrt[prev_ind],
							vrt[cur_ind],
							vrt[next_ind]):

							ear_found = False
							# ear has a vertex inside, continue search
							cur_pos = plus_wrap(cur_pos, num_indices)
							break
			
			else:
				# corner is concave, continue search
				cur_pos = plus_wrap(cur_pos, num_indices)
		return [prev_ind, cur_ind, next_ind]




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
		num_verts = len(vrt)
		indices = range(num_verts)

		# find spikes:
		spikes = []
		for cur_ind in indices:
			next_ind = plus_wrap(cur_ind, num_verts)
			prev_ind = minus_wrap(cur_ind, num_verts)

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
					spikes.append(cur_ind)
		return spikes






	def get_portals(self, threshold = 0.0, canvas=None):

		"""
		This function uses algorithm from R. Oliva and N. Pelechano - 
		Automatic Generation of Suboptimal NavMeshes
		This is work in progress
		"""
		sz = 5
		spikes = self.find_spikes(threshold)
		portals = []

		for spike_i in spikes:

			conevec1, conevec2 = self.get_anticone(spike_i, threshold)
			tip_v = self.vertices[spike_i]
			left_v = tip_v + conevec2
			right_v = tip_v + conevec1

			(closest_seg_i1, closest_seg_i2), closest_seg_point, closest_seg_dst = \
				self.find_closest_edge(spike_i)

			closest_vert_i, closest_vert_dst = self.find_closest_vert(left_v, tip_v, right_v)

			if canvas is not None and closest_vert_i is not None:
				vrt = self.vertices[closest_vert_i]
				canvas.create_oval(vrt.x - sz, vrt.y - sz, vrt.x + sz, vrt.y + sz, fill='cyan')

			closest_portal, closest_portal_point, closest_portal_dst = \
				self.find_closest_portal(spike_i, portals)

			if closest_portal is not None:
				closest_portal_v1 = closest_portal[0]
				closest_portal_v2 = closest_portal[1]
				# if canvas is not None:
				# 	x = closest_portal_point.x
				# 	y = closest_portal_point.y
					
				# 	spike_v = self.vertices[spike_i]
				# 	canvas.create_oval(x - sz, y - sz, x + sz, y + sz, fill='cyan')
				# 	canvas.create_line(x, y, spike_v.x, spike_v.y, fill='cyan')


			# closest edge always exist
			# closest vertex - not always (there might be no vertices inside the anticone)
			# closest portal - not always (there might be no portals inside the anticone
			#                              or no portals at all)
			
			new_portal_endpoint = [closest_seg_point]
			closest_dst = closest_seg_dst

			# check if there is a vertex closer than the closest edge
			if closest_vert_dst is not None and closest_vert_dst < closest_dst:
				new_portal_endpoint = [self.vertices[closest_vert_i]]
				closest_dst = closest_vert_dst


			if closest_portal_dst is not None and closest_portal_dst < closest_dst:
				closest_dst = closest_portal_dst
				# figure out if we want to create one or two portals

				# closest_portal_point can either be one of the endpoints,
				# or a middle point of the portal

				# we only want to connect to one or two endpoints,
				# not to the middle of the portal

				if closest_portal_point == closest_portal_v1:
					new_portal_endpoint = [closest_portal_v1]

				elif closest_portal_point == closest_portal_v2:
					new_portal_endpoint = [closest_portal_v2]

				else:
					spike_v = self.vertices[spike_i]
					prev_v = self.vertices[minus_wrap(spike_i, len(self.vertices))]
					next_v = self.vertices[plus_wrap(spike_i, len(self.vertices))]

					v1_inside = self._inside_anticone(closest_portal_v1, prev_v, spike_v, next_v)
					v2_inside = self._inside_anticone(closest_portal_v2, prev_v, spike_v, next_v)

					# if both portal endpoints are inside
					if v1_inside and v2_inside:
						# pick the closest one
						dst1 = Vector2.distance(closest_portal_v1, spike_v)
						dst2 = Vector2.distance(closest_portal_v2, spike_v)

						new_portal_endpoint = [closest_portal_v1] \
							if dst1 < dst2 else [closest_portal_v2]

					# if only one portal endpoint is inside
					elif v1_inside:
						new_portal_endpoint = [closest_portal_v1]

					elif v2_inside:
						new_portal_endpoint = [closest_portal_v2]

					# if none of the portal endpoints is inside
					else:
						new_portal_endpoint = [closest_portal_v1, closest_portal_v2]


			for portal_endpoint in new_portal_endpoint:
				portals.append( (self.vertices[spike_i], portal_endpoint) )

		return portals






	def find_closest_vert(self, left_v, tip_v, right_v):
		vrt = self.vertices
		num_verts = len(vrt)
		indices = range(num_verts)

		closest_ind = None
		closest_dst = None

		for cur_i in range(num_verts):
			cur_v = vrt[cur_i]

			if cur_v != tip_v:

				if self._inside_cone(vrt[cur_i], left_v, tip_v, right_v):
					dst = Vector2.distance(vrt[cur_i], tip_v)
					if closest_dst is None or dst < closest_dst:
						closest_dst = dst
						closest_ind = cur_i

		return closest_ind, closest_dst




	def _inside_cone(self, vert, left_v, tip_v, right_v):
		return Vector2.are_points_ccw(tip_v, right_v, vert) and not \
			Vector2.are_points_ccw(tip_v, left_v, vert)


	def find_closest_edge(self, spike_i, threshold=0.0):
		vrt = self.vertices
		num_verts = len(vrt)
		indices = range(num_verts)

		prev_i = minus_wrap(spike_i, num_verts)
		next_i = plus_wrap(spike_i, num_verts)

		spike_v = vrt[spike_i]
		prev_v = vrt[prev_i]
		next_v = vrt[next_i]

		closest_seg = None
		closest_dst = None
		closest_point = None

		for ind in range(num_verts):
			seg_i1 = ind
			seg_i2 = plus_wrap(ind, num_verts)

			# skip edges that belong to the spike:
			if seg_i2 == spike_i:
				continue
			if seg_i2 == next_i:
				continue


			seg_v1 = vrt[seg_i1]
			seg_v2 = vrt[seg_i2]

			if self._segment_inside_anticone(seg_v1, seg_v2, prev_v, spike_v, next_v):

				candidate_point, dst = \
				self.segment_closest_point_inside_anticone(seg_v1, seg_v2, prev_v, spike_v, next_v)

				if closest_dst is None or dst < closest_dst:
					closest_dst = dst
					closest_seg = (seg_i1, seg_i2)
					closest_point = candidate_point

		return closest_seg, closest_point, closest_dst





	def find_closest_portal(self, spike_i, portals):

		num_verts = len(self.vertices)
		prev_i = minus_wrap(spike_i, num_verts)
		next_i = plus_wrap(spike_i, num_verts)

		spike_v = self.vertices[spike_i]
		prev_v = self.vertices[prev_i]
		next_v = self.vertices[next_i]

		closest_portal = None
		closest_dst = None
		closest_point = None

		for portal in portals:

			port1 = portal[0]
			port2 = portal[1]

			candidate_point = None
			dst = None

			if self._segment_inside_anticone(port1, port2, prev_v, spike_v, next_v):

				candidate_point, dst = \
				self.segment_closest_point_inside_anticone(port1, port2, prev_v, spike_v, next_v)

			else:
				continue


			if closest_dst is None or dst < closest_dst:
				closest_dst = dst
				closest_point = candidate_point
				closest_portal = portal

		return closest_portal, closest_point, closest_dst





	def _segment_inside_anticone(self, seg_v1, seg_v2, prev_v, spike_v, next_v):
		v1_inside = self._inside_anticone(seg_v1, prev_v, spike_v, next_v)
		v2_inside = self._inside_anticone(seg_v2, prev_v, spike_v, next_v)

		# if at least one vertex is inside the cone:
		if v1_inside or v2_inside:
			return True
		# if none of the endpoints is inside cone, the middle of the segment might still be:
		else:
			if Vector2.segment_crosses_ray(seg_v1, seg_v2, prev_v, spike_v) and \
			Vector2.segment_crosses_ray(seg_v1, seg_v2, next_v, spike_v):
				return True
			else:
				return False



	def _inside_anticone(self, vert, prev_v, spike_v, next_v):
		return Vector2.are_points_ccw(prev_v, spike_v, vert) and \
			Vector2.are_points_ccw(spike_v, next_v, vert)



	def get_anticone(self, spike_ind, threshold=0.0):
		'''
		Returns 2 vectors that define the cone rays.
		The vectors have unit lengths
		The vector pair is right-hand
		'''
		vrt = self.vertices
		num_vrt = len(self.vertices)
		next_ind = plus_wrap(spike_ind, num_vrt)
		prev_ind = minus_wrap(spike_ind, num_vrt)

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




	def segment_closest_point_inside_anticone(self, seg1, seg2, prev, spike, next):
		proj_vert = Vector2.project_to_line(spike, seg1, seg2)

		# if projected point is inside anticone:
		if self._inside_anticone(proj_vert, prev, spike, next):

			# 1st case: projected point is inside anticone and inside segment:
			if Vector2.point_between(proj_vert, seg1, seg2):

				dst = Vector2.distance(proj_vert, spike)
				closest_point = proj_vert

			# 2nd case: projected point is inside anticone and outside segment:
			else:

				# choose the closest endpoint of the segment:
				dist1 = Vector2.distance(seg1, spike)
				dist2 = Vector2.distance(seg2, spike)
				dst = min(dist1, dist2)
				closest_point = seg1 if dist1 < dist2 else seg2

		# 3rd case: projected point is outside anticone:
		else:
			new_v1 = Vector2.where_segment_crosses_ray(seg1, seg2, prev, spike)
			new_v2 = Vector2.where_segment_crosses_ray(seg1, seg2, next, spike)

			# sometimes segment and ray intersect at the starting point of the ray:
			# we treat these cases as non-intersections:
			if new_v1 is not None and new_v1 == prev:
				new_v1 = None

			if new_v2 is not None and new_v2 == next:
				new_v2 = None

			dist1 = Vector2.distance(spike, new_v1) if new_v1 is not None else 9999999.0
			dist2 = Vector2.distance(spike, new_v2) if new_v2 is not None else 9999999.0
			dst = min(dist1, dist2)
			closest_point = new_v1 if dist1 < dist2 else new_v2

		return closest_point, dst