from mesh2d import *
import random

import Tkinter as tk



class Application(tk.Frame):
	def __init__(self, master=None):
		tk.Frame.__init__(self, master)
		self.grid()
		self.createWidgets()

		self._new_vertices = []
		self._polygons = []
		self._dots_ids = []
		self.dot_size = 6


	def createWidgets(self):
		self.canvas = tk.Canvas(self, background='#000000', width=1200, height=600,
			scrollregion=(0, 0, 3000, 30000))

		# horiz scrollbar
		self.hbar = tk.Scrollbar(self, orient = tk.HORIZONTAL)
		self.hbar.config(command = self.canvas.xview)

		# vert scrollbar
		self.vbar = tk.Scrollbar(self, orient = tk.VERTICAL)
		self.vbar.config(command = self.canvas.yview)

		self.canvas.config(xscrollcommand = self.hbar.set, yscrollcommand=self.vbar.set)

		self.canvas.bind('<Button-1>', self._add_vertex)
		self.canvas.bind('<Button-3>', self._add_polygon)

		# self.canvas.grid()
		# self.hbar.grid()
		# self.vbar.grid()

		self.canvas.bind('<Button-4>', self._mousewheel_up)
		self.canvas.bind('<Button-5>', self._mousewheel_down)

		self.hbar.pack(side = tk.BOTTOM, fill = tk.X)
		self.vbar.pack(side = tk.RIGHT, fill = tk.Y)
		self.canvas.pack(side = tk.LEFT, expand = True, fill = tk.BOTH)

		self.quitButton = tk.Button(self, text='Quit', command=self.quit)
		self.quitButton.pack()

		# vertices = [(50, 50), (70, 70), (10, 100)]
		# self.canvas.create_line(vertices, fill='#FFFFFF')


	def _mousewheel_up(self, event):
		self.canvas.yview_scroll(-1, "units")

	def _mousewheel_down(self, event):
		self.canvas.yview_scroll(1, "units")


	def _add_vertex(self, event):
		# reflect y to transform into right-hand coordinates
		x = float( self.canvas.canvasx(event.x) )
		y = float( self.canvas.canvasy(event.y) )

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
		threshold = 10.0 # degrees
		new_poly = Polygon2d(self._new_vertices, range(len(self._new_vertices)))
		del self._new_vertices[:]

		# for triangle in new_poly.triangles:
		# 	coords = new_poly.get_triangle_coords(triangle)
		# 	tri_id = self.canvas.create_line(coords , fill='#FFFFFF')


		# draw outline:
		crds = []
		for ind in new_poly.indices:
			vrt1 = new_poly.vertices[ind]
			crds.append(vrt1.x)
			crds.append(vrt1.y)
		crds.append(new_poly.vertices[0].x)
		crds.append(new_poly.vertices[0].y)

		self.canvas.create_line(crds, fill='#FFFFFF', width=3)

		polys = []
		new_poly.break_into_convex(polys, threshold, self.canvas)

		print "number of convex parts = {0}".format(len(polys))
		for poly in polys:

			rnd = lambda: random.randint(0,255)
			color = '#%02X%02X%02X' % (rnd(),rnd(),rnd())

			coords = poly.outline_coordinates()

			# self.canvas.create_line(coords, fill='green')
			self.canvas.create_polygon(coords[:-2], activefill=color, outline='', fill='')
		# self.canvas.create_line(poly2.outline_coordinates(), fill='red')
		
		self._polygons.append(new_poly)

		# for d_id in self._dots_ids:
		# 	self.canvas.delete(d_id)


		# try:

			# portals = new_poly.get_portals(threshold, canvas=self.canvas)

			# for portal in portals:
			# 	p1 = new_poly.vertices[portal['start_index']]
			# 	p2 = portal['end_point']

				# self.canvas.create_line([p1.x, p1.y, p2.x, p2.y], fill='yellow')

			# spikes = new_poly.find_spikes(threshold)
			# for spike in spikes:
			# 	sp_v = new_poly.vertices[spike[1]]
			# 	sz = self.dot_size
			# 	self.canvas.create_oval(sp_v.x - sz, sp_v.y - sz, sp_v.x + sz, sp_v.y + sz, fill='red')

			# 	antic1, antic2 = new_poly.get_anticone(spike[0], spike[1], spike[2], threshold)
			# 	ant_v1 = sp_v + antic1*5.0
			# 	ant_v2 = sp_v + antic2*5.0
			# 	self.canvas.create_line(ant_v1.x, ant_v1.y, sp_v.x, sp_v.y, ant_v2.x, ant_v2.y, fill='magenta')



		# except ZeroSegmentError as ex:
		# 	print ex.message()
		# 	p1 = ex.segment()[0]
		# 	p2 = ex.segment()[1]
		# 	print "{0}, {1}\n{2}, {3}".format(p1.x, p1.y, p2.x, p2.y)
		# 	self.canvas.create_oval([p1.x - 3, p1.y - 3, p1.x + 3, p1.y + 3], fill='red')




def main():
	app = Application()
	app.master.title('Triangulation test')
	app.mainloop() 



if __name__ == '__main__':
	main()