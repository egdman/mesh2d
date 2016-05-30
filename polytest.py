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
		self.dot_size = 3


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


		try:

			portals = new_poly.get_portals()
		
			for portal in portals:
				p1 = portal[0]
				p2 = portal[1]
				self.canvas.create_line([p1.x, p1.y, p2.x, p2.y], fill='yellow')

			spikes = new_poly.find_spikes()
			for spike in spikes:
				sp_v = new_poly.vertices[spike]
				sz = self.dot_size
				self.canvas.create_oval(sp_v.x - sz, sp_v.y - sz, sp_v.x + sz, sp_v.y + sz, fill='red')


		except ZeroSegmentError as ex:
			print ex.message()
			p1 = ex.segment()[0]
			p2 = ex.segment()[1]
			print "{0}, {1}\n{2}, {3}".format(p1.x, p1.y, p2.x, p2.y)
			self.canvas.create_oval([p1.x - 3, p1.y - 3, p1.x + 3, p1.y + 3], fill='red')




def main():
	app = Application()
	app.master.title('Triangulation test')
	app.mainloop() 



if __name__ == '__main__':
	main()