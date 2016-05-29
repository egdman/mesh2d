from mesh2d import *


import Tkinter as tk




class Application(tk.Frame):
	def __init__(self, master=None):
		tk.Frame.__init__(self, master)
		self.grid()
		self.createWidgets()

		self._new_vertices = []
		self._polygons = []
		self._dots_ids = []
		self.dot_size = 4


	def createWidgets(self):
		self.canvas = tk.Canvas(self, background='#000000', width=1200, height=600)
		self.canvas.bind('<Button-1>', self._add_vertex)
		self.canvas.bind('<Button-3>', self._add_polygon)
		self.canvas.grid()
		self.quitButton = tk.Button(self, text='Quit', command=self.quit)
		self.quitButton.grid()

		# vertices = [(50, 50), (70, 70), (10, 100)]
		# self.canvas.create_line(vertices, fill='#FFFFFF')


	def _add_vertex(self, event):
		# reflect y to transform into right-hand coordinates
		x = float(event.x)
		y = float(event.y)

		new_vert = Vector2(x, y)
		self._new_vertices.append(Vector2(x, y))
		sz = self.dot_size
		
		if len(self._new_vertices) > 1:
			prev_x = self._new_vertices[-2].x
			prev_y = self._new_vertices[-2].y
			self.canvas.create_line(x, y, prev_x, prev_y, fill='#A0A0A0')

		id = self.canvas.create_oval(x - sz, y - sz, x + sz, y + sz, fill='#FFFFFF')
		self._dots_ids.append(id)


	def _add_polygon(self, event):
		new_poly = Polygon2d(self._new_vertices)
		del self._new_vertices[:]

		# for triangle in new_poly.triangles:
		# 	coords = new_poly.get_triangle_coords(triangle)
		# 	tri_id = self.canvas.create_line(coords , fill='#FFFFFF')


		# draw outline:
		crds = []
		for ind in range(len(new_poly.vertices)):
			vrt1 = new_poly.vertices[ind]
			crds.append(vrt1.x)
			crds.append(vrt1.y)
		crds.append(new_poly.vertices[0].x)
		crds.append(new_poly.vertices[0].y)

		self.canvas.create_line(crds, fill='#FFFFFF')



		self._polygons.append(new_poly)

		for d_id in self._dots_ids:
			self.canvas.delete(d_id)

		spikes = new_poly.find_spikes()
		print "Num spikes = {0}".format(len(spikes))
		sz = self.dot_size
		for sp in spikes:
			sp_v = new_poly.vertices[sp]
			x = sp_v.x
			y = sp_v.y
			self.canvas.create_oval(x - sz, y - sz, x + sz, y + sz, fill='red')

			clos_i, v_dst = new_poly.find_closest_vert(sp)
			clos_seg, clos_point, e_dst = new_poly.find_closest_edge(sp)

			portal_point = None

			if clos_i is not None:
				if v_dst < e_dst:
					portal_point = new_poly.vertices[clos_i]
				else:
					portal_point = clos_point
			else:
					portal_point = clos_point

			seg_v1 = new_poly.vertices[clos_seg[0]]
			seg_v2 = new_poly.vertices[clos_seg[1]]
			
			self.canvas.create_line([sp_v.x, sp_v.y, portal_point.x, portal_point.y], fill='yellow')
			self.canvas.create_line([seg_v1.x, seg_v1.y, seg_v2.x, seg_v2.y], fill='green')

			# if clos_i is not None:
			# 	clos_v = new_poly.vertices[clos_i]
			# 	# self.canvas.create_oval(clos_v.x - sz, clos_v.y - sz, clos_v.x + sz, clos_v.y + sz, fill='green')
			# 	self.canvas.create_line([x, y, clos_v.x, clos_v.y], fill='yellow')

			# if clos_seg is not None:

			# 	print clos_seg[0]
			# 	print clos_seg[1]

			# 	seg_v1 = new_poly.vertices[clos_seg[0]]
			# 	seg_v2 = new_poly.vertices[clos_seg[1]]

			# 	self.canvas.create_oval(clos_point.x - sz, clos_point.y - sz,
			# 		clos_point.x + sz, clos_point.y + sz, fill='green')

			# 	self.canvas.create_line([seg_v1.x, seg_v1.y, seg_v2.x, seg_v2.y], fill='purple')
			# 	self.canvas.create_line([sp_v.x, sp_v.y, clos_point.x, clos_point.y], fill='green')




def main():
	app = Application()
	app.master.title('Triangulation test')
	app.mainloop() 



if __name__ == '__main__':
	main()