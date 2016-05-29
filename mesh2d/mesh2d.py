import math
from .vector2 import Vector2

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


	def find_spikes(self):
		vrt = self.vertices
		num_verts = len(vrt)
		indices = range(num_verts)

		# find spikes:
		spikes = []
		for cur_ind in indices:
			next_ind = plus_wrap(cur_ind, num_verts)
			prev_ind = minus_wrap(cur_ind, num_verts)
			if not Vector2.are_points_ccw(vrt[prev_ind], vrt[cur_ind], vrt[next_ind]):
				spikes.append(cur_ind)

		return spikes


	def break_into_convex(self):

		"""
		This function uses algorithm from R. Oliva and N. Pelechano - 
		Automatic Generation of Suboptimal NavMeshes
		This is work in progress
		"""

		spikes = self.find_spikes()
		# for spike_i in spikes:
		# 	clos_vert_i = self.find_closest_vert(spike_i)
		# 	clos_seg = self.find_closest_edge(spike_i)



	def find_closest_vert(self, spike_i):
		vrt = self.vertices
		num_verts = len(vrt)
		indices = range(num_verts)

		prev_i = minus_wrap(spike_i, num_verts)
		next_i = plus_wrap(spike_i, num_verts)

		closest_ind = None
		closest_dst = None

		for cur_i in range(num_verts):
			if cur_i != spike_i:
				if self._inside_anticone(vrt[cur_i], vrt[prev_i], vrt[spike_i], vrt[next_i]):
					dst = Vector2.distance(vrt[cur_i], vrt[spike_i])
					if closest_dst is None or dst < closest_dst:
						closest_dst = dst
						closest_ind = cur_i

		return closest_ind, closest_dst


	def find_closest_edge(self, spike_i):
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
				proj_vert = Vector2.project_to_line(spike_v, seg_v1, seg_v2)

				# if projected point is inside anticone:
				if self._inside_anticone(proj_vert, prev_v, spike_v, next_v):

					# 1st case: projected point is inside anticone and inside segment:
					if Vector2.point_between(proj_vert, seg_v1, seg_v2):

						dst = Vector2.distance(proj_vert, spike_v)
						candidate_point = proj_vert

					# 2nd case: projected point is inside anticone and outside segment:
					else:

						# choose the closest endpoint of the segment:
						dist1 = Vector2.distance(seg_v1, spike_v)
						dist2 = Vector2.distance(seg_v2, spike_v)
						dst = min(dist1, dist2)
						candidate_point = seg_v1 if dist1 < dist2 else seg_v2

				# 3rd case: projected point is outside anticone:
				else:

					# if this edge is adjacent to the spike (previous egde)
					if seg_i2 == prev_i:
						dst = Vector2.distance(spike_v, seg_v1)
						candidate_point = seg_v1

					# if this edge is adjacent to the spike (next egde)
					elif seg_i1 == next_i:
						dst = Vector2.distance(spike_v, seg_v2)
						candidate_point = seg_v2

					# if this edge is not adjacent to the spike:
					else:

						new_v1 = Vector2.where_segment_crosses_ray(seg_v1, seg_v2, prev_v, spike_v)
						new_v2 = Vector2.where_segment_crosses_ray(seg_v1, seg_v2, next_v, spike_v)

						dist1 = Vector2.distance(spike_v, new_v1) if new_v1 is not None else 9999999.0
						dist2 = Vector2.distance(spike_v, new_v2) if new_v2 is not None else 9999999.0
						dst = min(dist1, dist2)
						candidate_point = new_v1 if dist1 < dist2 else new_v2

				if closest_dst is None or dst < closest_dst:
					closest_dst = dst
					closest_seg = (seg_i1, seg_i2)
					closest_point = candidate_point

		return closest_seg, closest_point, closest_dst




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