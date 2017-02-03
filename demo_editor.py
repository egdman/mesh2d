import os
from mesh2d import *
import random
import yaml

import Tkinter as tk

pkg_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(pkg_dir, 'resources')
button_dir = os.path.join(resource_dir, 'buttons')



class Tool(object):

	def __init__(self, parent):
		self.parent = parent

	def right_click(self, event):
		pass

	def left_click(self, event):
		pass


class Create(Tool):

	def right_click(self, event):
		self.parent._add_polygon(event)

	def left_click(self, event):
		self.parent._add_vertex(event)


class Select(Tool):

	def right_click(self, event):
		print("SELECT: right click")


	def left_click(self, event):
		# print("SELECT: left click")
		obj_ids = self.parent.canvas.find_overlapping(event.x, event.y, event.x, event.y)
		print (obj_ids)
		print ("mouse at {}, {}".format(event.x, event.y))

		# x = float( self.canvas.canvasx(event.x) )
		# y = float( self.canvas.canvasy(event.y) )
		# pointer = Vector2(x, y)

		# for poly in parent._polygons:
		# 	if poly.point_inside(pointer): 



class Application(tk.Frame):
	def __init__(self, master=None):
		tk.Frame.__init__(self, master)
		self.grid()
		self.createWidgets()

		self._new_vertices = []
		self._polygons = []
		self._dots_ids = []
		self.dot_size = 3

		self.select_tool = Select(self)
		self.create_tool = Create(self)
		# ......
		#.......

		self.active_tool = self.create_tool

		self.last_created_poly = None
		self.pan_mode = False
		self.rotate_mode = False
		self.last_x = 0
		self.last_y = 0



	def createWidgets(self):
		self.canvas = tk.Canvas(self, background='#000000', width=1000, height=900,
			scrollregion=(0, 0, 90000, 90000))

		self.canvas_center = Vector2(float(self.canvas['width']) / 2., float(self.canvas['height']) / 2.)

		# horiz scrollbar
		self.hbar = tk.Scrollbar(self, orient = tk.HORIZONTAL)
		self.hbar.config(command = self.canvas.xview)

		# vert scrollbar
		self.vbar = tk.Scrollbar(self, orient = tk.VERTICAL)
		self.vbar.config(command = self.canvas.yview)

		self.canvas.config(
			xscrollcommand = self.hbar.set, yscrollcommand=self.vbar.set,
			xscrollincrement=1, yscrollincrement=1)

		self.canvas.bind('<ButtonRelease-1>', self._left_up)
		self.canvas.bind('<ButtonRelease-3>', self._right_up)


		# camera controls
		self.canvas.bind('<Control-Button-1>', self._ctrl_left_down)
		self.canvas.bind('<Control-Button-3>', self._ctrl_right_down)
		self.canvas.bind('<Motion>', self._mouse_moved)

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

		
		# butens
		self.selectToolIcon = tk.PhotoImage(file=os.path.join(button_dir, 'select.gif'))
		self.selectToolBtn = tk.Button(
			self,
			image=self.selectToolIcon,
			height=31,
			width=31,
			command = self._select_cb)
		self.selectToolBtn.pack()


		self.createToolIcon = tk.PhotoImage(file=os.path.join(button_dir, 'create.gif'))
		self.createToolBtn = tk.Button(
			self,
			image=self.createToolIcon,
			height=31,
			width=31,
			command = self._create_cb)
		self.createToolBtn.pack()


		self.saveToolIcon = tk.PhotoImage(file=os.path.join(button_dir, 'save.gif'))
		self.saveToolBtn = tk.Button(
			self,
			image=self.saveToolIcon,
			height=31,
			width=31,
			command = self._save_last)
		self.saveToolBtn.pack()


		# vertices = [(50, 50), (70, 70), (10, 100)]
		# self.canvas.create_line(vertices, fill='#FFFFFF')


	def _save_last(self):
		with open("saved_poly.yaml", 'w') as savef:
			yaml.dump(self.last_created_poly, savef)



	def _select_cb(self):
		self.active_tool = self.select_tool

	def _create_cb(self):
		self.active_tool = self.create_tool		


	def _get_event_modifiers(self, event):
		state = {
			'ctrl': event.state & 0x0004 != 0,
			'ralt': event.state & 0x0080 != 0,
			'lalt': event.state & 0x0008 != 0,
			'shift': event.state & 0x0001 != 0,
		}
		return set(modname for modname in state if state[modname])


	def _left_up(self, event):
		self.pan_mode = False
		
		mods = self._get_event_modifiers(event)

		# do something only if no modifiers
		if len(mods) == 0:
			self.active_tool.left_click(event)


	def _right_up(self, event):
		self.rotate_mode = False

		mods = self._get_event_modifiers(event)

		# do something only if no modifiers
		if len(mods) == 0:
			self.active_tool.right_click(event)



	def _ctrl_left_down(self, event):
		self.pan_mode = True


	def _ctrl_right_down(self, event):
		self.rotate_mode = True



	def _mouse_moved(self, event):

		if self.pan_mode:
			delta_x = event.x - self.last_x
			delta_y = event.y - self.last_y
			self.canvas.xview_scroll(-delta_x, 'units')
			self.canvas.yview_scroll(-delta_y, 'units')

		if self.rotate_mode:

			sangle = -Vector2.signed_angle(
				Vector2(event.x, event.y) - self.canvas_center,
				Vector2(self.last_x, self.last_y) - self.canvas_center
			)

			for dot_id in self._dots_ids:
				crds = self.canvas.coords(dot_id)

				if len(crds) < 4: continue

				old_pos = Vector2((crds[0] + crds[2])/2., (crds[1] + crds[3])/2.)

				new_pos = Matrix.rotate2d(
					rotation_center = self.canvas_center,
					angle = sangle).multiply(old_pos)
				
					# Matrix((3,1), (old_pos.x, old_pos.y, 1.0)))

				new_pos = Vector2(new_pos.values[0], new_pos.values[1])

				dif = new_pos - old_pos
				self.canvas.move(dot_id, dif.x, dif.y)


		self.last_x = event.x
		self.last_y = event.y



	def _mousewheel_up(self, event):
		self.canvas.yview_scroll(-1, "units")

	def _mousewheel_down(self, event):
		self.canvas.yview_scroll(1, "units")


	def _add_vertex(self, event):
		# reflect y to transform into right-hand coordinates
		x = float( self.canvas.canvasx(event.x) )
		y = float( self.canvas.canvasy(event.y) )

		self._new_vertices.append(Vector2(x, y))
		sz = self.dot_size
		
		if len(self._new_vertices) > 1:
			prev_x = self._new_vertices[-2].x
			prev_y = self._new_vertices[-2].y
			self.canvas.create_line(x, y, prev_x, prev_y, fill='#A0A0A0')

		dot_id = self.canvas.create_oval(x - sz, y - sz, x + sz, y + sz, fill='#FFFFFF')
		self._dots_ids.append(dot_id)


	def _add_polygon(self, event):
		if len(self._new_vertices) < 3: return

		threshold = 10.0 # degrees
		new_poly = Mesh2d(self._new_vertices[:], range(len(self._new_vertices)))

		# to save in case of failure
		self.last_created_poly = Mesh2d(self._new_vertices[:], range(len(self._new_vertices)))

		del self._new_vertices[:]

		# for triangle in new_poly.triangles:
		# 	coords = new_poly.get_triangle_coords(triangle)
		# 	tri_id = self.canvas.create_line(coords , fill='#FFFFFF')

		# draw whole polygon:
		coords = new_poly.outline_coordinates()
		self.canvas.create_polygon(coords[:-2], fill='#1A1A1A', outline='', activefill='#1A1A9A')

		# break into convex parts:
		# new_poly.break_into_convex(polys, threshold, self.canvas)

		def error_dump(poly):
			with open("debug_dump.yaml", 'w') as debf:
				yaml.dump(poly, debf)			

		try:
			portals = new_poly.break_into_convex(threshold, self.canvas)
			polys = new_poly.get_pieces_as_meshes()

			print ("number of portals      = {0}".format(len(portals)))
			print ("number of convex parts = {0}".format(len(polys)))

			if len(polys) != len(portals) + 1:
				print ("Error!")
				error_dump(self.last_created_poly)
	
		except ValueError as ve:
			error_dump(self.last_created_poly)
			raise ve
 

		
		for poly in polys:

			rnd = lambda: random.randint(0,255)
			color = '#%02X%02X%02X' % (rnd(),rnd(),rnd())

			coords = poly.outline_coordinates()

			self.canvas.create_polygon(coords[:-2], activefill='#111111', outline='', fill='')
		
		
		# draw outline:
		coords = new_poly.outline_coordinates()
		self.canvas.create_line(coords, fill='#FFFFFF', width=1)


		self._polygons.append(new_poly)

		self.active_tool = self.select_tool

		for d_id in self._dots_ids:
			self.canvas.delete(d_id)




class ProvideException(object):
    def __init__(self, func):
        self._func = func

    def __call__(self, *args):

        try:
            return self._func(*args)

        except Exception, e:
            print 'Exception was thrown', str(e)
            # Optionally raise your own exceptions, popups etc


@ProvideException
def main():
	app = Application()
	app.master.title('Triangulation test')
	app.mainloop() 



if __name__ == '__main__':
	main()