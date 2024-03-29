from __future__ import print_function
import os
import random
import fnmatch
import platform
import struct
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk
from argparse import ArgumentParser
import traceback
from math import floor
from mesh2d import *
from views import *

pkg_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(pkg_dir, 'resources')
button_dir = os.path.join(resource_dir, 'buttons')

try:
    AnyError = StandardError
except NameError:
    AnyError = Exception


RELAX_ANGLE = 0.0001 # degrees


class RecordType:
    class AddVertex: pass
    class AddPoly: pass

def v2col(x, y):
    return Matrix.column_vec((x, y, 1.))


def polys_to_ascii(polys):
    def _poly_to_ascii(poly):
        s = ""
        loops = ((poly.vertices[idx] for idx in poly.graph.loop_iterator(loop)) for loop in poly.graph.loops)
        for loop in loops:
            s += " ".join("{} {}".format(ieee754_ser(v[0]), ieee754_ser(v[1])) for v in loop)
            s += "\n"
        return s

    return "&\n".join(_poly_to_ascii(poly) for poly in polys)


def ascii_to_polys(text, scale):
    def read_float(s):
        try:
            return float(s)
        except ValueError:
            return ieee754_unser(s)

    def _ascii_to_poly(text):
        loops = []
        for line in text.splitlines():
            loop = []
            loops.append(loop)
            crds = (scale * read_float(el.strip()) for el in line.strip().split())
            try:
                while True:
                    v0 = next(crds)
                    v1 = next(crds)
                    loop.append(vec(v0, v1))
            except StopIteration:
                pass
        poly = Polygon2d(loops[0])
        for hole in loops[1:]:
            poly.add_hole(hole)
        return poly

    return list(_ascii_to_poly(substring.strip()) for substring in text.split("&"))


def ieee754_ser(floatNumber):
    i, = struct.unpack('<Q', struct.pack('<d', floatNumber))
    return format(i, '#x')

def ieee754_unser(hexString):
    return struct.unpack('<d', struct.pack('<Q', int(hexString, base=16)))[0]


def dump_history(history):
    import uuid
    fname = "hist_{}".format(uuid.uuid4().hex)
    with open(fname, 'w') as stream:
        for kind, data in history:
            if kind == RecordType.AddVertex:
                x, y = data
                stream.write(ieee754_ser(x) + " : " + str(x) + '\n')
                stream.write(ieee754_ser(y) + " : " + str(y) + '\n')
            elif kind == RecordType.AddPoly:
                mode = '+' if data == Bool.Add else '-'
                stream.write("AddPoly " + mode + '\n')


class Tool(object):

    def __init__(self, parent):
        self.parent = parent

    def right_click(self, event):
        pass

    def left_click(self, event):
        pass

    def mouse_moved(self, event, dx, dy):
        pass


class CreateWallTool(Tool):
    def __init__(self, parent):
        self.width_mode = False
        self.start = None
        self.end = None
        self.width = 25.
        super(CreateWallTool, self).__init__(parent)


    # toggle width adjustment mode
    def right_click(self, event):
        if self.start is None: return
        self.width_mode = not self.width_mode

        # When exiting width adjustment mode, snap cursor back to endpoint.
        # This might not work on all platforms.
        if not self.width_mode:
            endpt_screen = self.parent.get_screen_crds(self.end[0], self.end[1])
            self.parent.canvas.event_generate('<Motion>', warp=True,
                x=endpt_screen[0], y=endpt_screen[1])

        self.redraw()


    # if not in width adjust mode, set start- and end-points
    def left_click(self, event):
        app = self.parent
        if not self.width_mode:
            if self.start is not None:
                c0, c1, c2, c3 = app.draw_objects['tool_helpers/wall'].find_corners()
                app.add_polygon((c0, c1, c2, c3))
                app.remove_draw_objects_glob('tool_helpers/wall')
                self.width_mode = False

            self.start = app.get_world_crds(event.x, event.y)
            self.end = self.start + vec(1, 0)
            app.add_draw_object('tool_helpers/wall',
                WallHelperView(self.start, self.end, self.width))

            app.draw_all()


    def mouse_moved(self, event, dx, dy):
        app = self.parent
        if self.width_mode:
            self.width += 1.*dx
            if self.width < 0.05: self.width = 0.05
            self.redraw()
        elif self.start is not None:
            self.end = app.get_world_crds(event.x, event.y)
            self.redraw()


    # draw temporary wall helpers
    def redraw(self):
        app = self.parent
        app.draw_objects['tool_helpers/wall'].modify(
            self.start, self.end, self.width)
        app.draw_all()


class FreePolyTool(Tool):
    def __init__(self, parent):
        self.vertices = []
        self.half_spacing = 0.005 #2.5
        super(FreePolyTool, self).__init__(parent)

    def grid_snap(self, p):
        s = self.half_spacing
        x, y = p
        x = floor((x+s) / (2*s))
        y = floor((y+s) / (2*s))
        return vec(x*s*2, y*s*2)


    def right_click(self, event):
        app = self.parent
        app.history.append((RecordType.AddPoly, app.get_bool_mode()))

        def remove_duplicates(vertices):
            seen = set()
            for v in vertices:
                if v not in seen:
                    seen.add(v)
                    yield v

        self.vertices = list(remove_duplicates(self.vertices))
        if len(self.vertices) < 3: return

        app.add_polygon(self.vertices)

        del self.vertices[:]
        # remove helper views
        app.remove_draw_objects_glob('tool_helpers/free_poly/*')
        app.draw_all()


    def left_click(self, event):
        app = self.parent
        new_vrt = app.get_world_crds(event.x-10, event.y-10)
        new_vrt = self.grid_snap(new_vrt)

        app.history.append((RecordType.AddVertex, new_vrt))

        app.add_draw_object(
            'tool_helpers/free_poly/point_{}'.format(len(self.vertices)),
            PointHelperView(loc=new_vrt))

        self.vertices.append(new_vrt)

        if len(self.vertices) == 1:
            app.add_draw_object(
                'tool_helpers/free_poly/future_segment',
                SegmentHelperView(new_vrt, new_vrt, color="#99dd99"))

        if len(self.vertices) > 1:
            prev_vrt = self.vertices[-2]

            app.add_draw_object(
                'tool_helpers/free_poly/segment_{}'.format(len(self.vertices)),
                SegmentHelperView(prev_vrt, new_vrt))

        app.draw_all()


    def mouse_moved(self, event, dx, dy):
        app = self.parent
        loc = app.get_world_crds(event.x-10, event.y-10)
        loc = self.grid_snap(loc)

        if 'tool_helpers/free_poly/cursor' not in app.draw_objects:
            app.add_draw_object('tool_helpers/free_poly/cursor',
                PlusView(loc=loc, size=12, color="#99dd99"))
        else:
            app.draw_objects['tool_helpers/free_poly/cursor'].modify(loc)

        if len(self.vertices) > 0:
            prev_vrt = self.vertices[-1]
            app.draw_objects['tool_helpers/free_poly/future_segment'].modify(prev_vrt, loc)

        app.draw_all()



class SelectTool(Tool):

    def right_click(self, event):
        print("SELECT: right click")


    def left_click(self, event):
        print("SELECT: left click, {}".format(self.parent.get_world_crds(event.x, event.y)))


def get_event_modifiers(event):
    state = (
        ('ctrl',  event.state & 0x0004),
        ('ralt',  event.state & 0x0080),
        ('shift', event.state & 0x0001),
    )
    return set(modname for (modname, hit) in state if hit != 0)

class Bool:
    class Subtract: pass
    class Add: pass


class VisualDebug(object):
    def __init__(self, app):
        self.app = app
        self.counter = 0

    def add_text(self, loc, text, size=18, scale=False, color='#ffffff'):
        self.app.add_draw_object(
            'debug_text_{}'.format(self.counter),
            TextView(loc=loc, text=text, size=size, scale=scale, color=color))
        self.counter += 1

    def add_polygon(self, points, color='#ffffff'):
        self.app.add_draw_object('debug_text_{}'.format(self.counter),
            PolygonView(Polygon2d(points), outline_color=color))
        self.counter += 1

    def add_plus(self, loc, color="#ffffff", size=20):
        self.app.add_draw_object('debug_text_{}'.format(self.counter),
            PlusView(loc=loc, size=size, color=color))
        self.counter += 1


class Application(tk.Frame):
    def __init__(self, master=None, input_polys=(), db_mode=False, camera_pos=vec(0, 0)):
        tk.Frame.__init__(self, master)
        self.focus_set()

        self.debugger = VisualDebug(self) if db_mode else None

        self.history = []
        self.this_is_windows = "windows" in platform.system().lower()
        self.this_is_osx = "darwin" in platform.system().lower()

        self.bind('<Control-z>', self.undo_or_redo)
        self.bind('<Control-Z>', self.undo_or_redo)

        self.grid()
        self.createWidgets()

        self._polygons = []
        self.navmeshes = []
        self._dots_ids = []
        self.dot_size = 3

        self.active_tool = FreePolyTool(self)

        # camera control modes
        self.pan_mode = False
        self.rotate_mode = False
        self.zoom_mode = False

        # last coordinates of mouse pointer
        self.last_x = 0
        self.last_y = 0

        self.draw_objects = {}

        # info about camera position
        self.camera_pos = camera_pos
        self.camera_rot = 0.
        self.camera_size = 1.
        self.rot_marker_world = vec(0, 0)

        self.zoom_rate = 1.1 #1.2

        # draw big cross marker at the world coordinate origin
        self.add_draw_object("origin_marker_x",
            SegmentHelperView(vec(-250, 0), vec(250, 0), color='#2f2f2f'))
        self.add_draw_object("origin_marker_y",
            SegmentHelperView(vec(0, -250), vec(0, 250), color='#2f2f2f'))

        self.pointer_over_poly = False

        for poly in input_polys:
            self._polygons.append(poly)
            num_polys = len(self.find_draw_objects_glob('polys/*'))

            navmesh = subdivide_polygon(poly, RELAX_ANGLE, self.debugger)
            if navmesh:
                self.add_draw_object('polys/poly_{}'.format(num_polys), NavMeshView(navmesh))
                self.navmeshes.append(navmesh)
            else:
                self.add_draw_object('polys/poly_{}'.format(num_polys), PolygonView(poly))

        self.undo_stack = [self._polygons]
        self.undo_depth = 0
        self.draw_all()


    def _set_polygons(self, polygons):
        self.remove_draw_objects_glob('polys/*')
        self.remove_draw_objects_glob('debug_*')

        self._polygons = []
        self.navmeshes = []
        for poly in polygons:
            self._polygons.append(poly)
            num_polys = len(self.find_draw_objects_glob('polys/*'))

            navmesh = subdivide_polygon(poly, RELAX_ANGLE, self.debugger)
            if navmesh:
                self.add_draw_object('polys/poly_{}'.format(num_polys), NavMeshView(navmesh))
                self.navmeshes.append(navmesh)
            else:
                self.add_draw_object('polys/poly_{}'.format(num_polys), PolygonView(poly))


    def undo_or_redo(self, event):
        mods = get_event_modifiers(event)
        if "shift" in mods:
            self.redo()
        else:
            self.undo()


    def undo(self):
        if self.undo_depth + 1 < len(self.undo_stack):
            self.undo_depth += 1
            self._set_polygons(self.undo_stack[len(self.undo_stack) - self.undo_depth - 1])
            self.draw_all()

    def redo(self):
        if self.undo_depth > 0:
            self.undo_depth -= 1
            self._set_polygons(self.undo_stack[len(self.undo_stack) - self.undo_depth - 1])
            self.draw_all()


    def createWidgets(self):
        cwidth = 1200
        cheight = 900

        self.canvas = tk.Canvas(self, background='#000000', width=cwidth, height=cheight,
            scrollregion=(0, 0, cwidth, cheight))

        self.canvas_center = vec(float(self.canvas['width']) / 2., float(self.canvas['height']) / 2.)


        # pack root window into OS window and make it fill the entire window
        self.pack(side = tk.LEFT, expand = True, fill = tk.BOTH)

        self.debug_canvas = tk.Canvas(self, background='#000020', width=cwidth, height=cheight,
            scrollregion=(-1000, -1000, 1000, 1000))
        self.debug_canvas.scan_dragto(cwidth//2, cheight//2, gain=1)

        self.canvas.config(xscrollincrement=1, yscrollincrement=1)

        self.debug_canvas.bind('<Button-1>', lambda ev: self.debug_canvas.scan_mark(ev.x, ev.y))
        self.debug_canvas.bind('<Motion>', self._debug_pan_mouse)

        # bind right and left mouse clicks
        self.canvas.bind('<ButtonRelease-1>', self._left_up)
        self.canvas.bind('<ButtonRelease-2>', self._right_up)
        self.canvas.bind('<ButtonRelease-3>', self._right_up)


        # bind camera controls
        self.canvas.bind('<Control-Button-1>', self._ctrl_left_down)
        self.canvas.bind('<Control-Button-2>', self._ctrl_right_down)
        self.canvas.bind('<Control-Button-3>', self._ctrl_right_down)
        self.canvas.bind('<Motion>', self._mouse_moved)


        # mouse wheel
        if self.this_is_windows or self.this_is_osx:
            self.canvas.bind_all('<MouseWheel>', self._windows_mousewheel)
        else:
            self.canvas.bind_all('<Button-4>', self._mousewheel_up)
            self.canvas.bind_all('<Button-5>', self._mousewheel_down)

        self.canvas.bind_all('-', self._zoom_out)
        self.canvas.bind_all('=', self._zoom_in)

        # escape button resets the active tool
        self.bind_all('<Escape>', lambda _: self.change_active_tool(type(self.active_tool)))

        self.canvas.pack(side = tk.RIGHT, expand = True, fill = tk.BOTH)


        # butens
        # self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        # self.quitButton.pack()

        self.selectToolIcon = tk.PhotoImage(file=os.path.join(button_dir, 'select.gif'))
        self.selectToolBtn = tk.Button(
            self,
            image=self.selectToolIcon,
            height=31,
            width=31,
            command = lambda: self.change_active_tool(SelectTool))
        self.selectToolBtn.pack()


        self.createToolIcon = tk.PhotoImage(file=os.path.join(button_dir, 'create.gif'))
        self.createToolBtn = tk.Button(
            self,
            image=self.createToolIcon,
            height=31,
            width=31,
            command = lambda: self.change_active_tool(FreePolyTool))
        self.createToolBtn.pack()


        self.wallToolIcon = tk.PhotoImage(file=os.path.join(button_dir, 'wall.gif'))
        self.wallToolBtn = tk.Button(
            self,
            image=self.wallToolIcon,
            command = lambda: self.change_active_tool(CreateWallTool)
        )
        self.wallToolBtn.pack()


        self.saveToolIcon = tk.PhotoImage(file=os.path.join(button_dir, 'save.gif'))
        self.saveToolBtn = tk.Button(
            self,
            image=self.saveToolIcon,
            height=31,
            width=31,
            command = self.save_polygons)
        self.saveToolBtn.pack()


        self.minusIcon = tk.PhotoImage(file=os.path.join(button_dir, 'minus.gif'))
        self.minusBtn = tk.Button(
            self,
            image=self.minusIcon,
            command = self._switch_subtract_mode
        )
        self.minusBtn.pack(side = tk.BOTTOM)

        self.plusIcon = tk.PhotoImage(file=os.path.join(button_dir, 'plus.gif'))
        self.plusBtn = tk.Button(
            self,
            image=self.plusIcon,
            relief=tk.SUNKEN,
            command = self._switch_add_mode
        )
        self.plusBtn.pack(side = tk.BOTTOM)


    def save_polygons(self):
        # dump_history(self.history)
        if len(self._polygons) > 0:
            with open("poly.txt", 'w') as stream:
                stream.write(polys_to_ascii(self._polygons))


    def get_bool_mode(self):
        if self.plusBtn.cget('relief') == tk.SUNKEN:
            return Bool.Add
        return Bool.Subtract



    def _switch_add_mode(self):
        self.plusBtn.config(relief=tk.SUNKEN)
        self.minusBtn.config(relief=tk.RAISED)



    def _switch_subtract_mode(self):
        self.plusBtn.config(relief=tk.RAISED)
        self.minusBtn.config(relief=tk.SUNKEN)



    def _debug_pan_mouse(self, event):
        if event.state & 0x0100 != 0:
            self.debug_canvas.scan_dragto(event.x, event.y, 1)


    def add_draw_object(self, name, draw_obj):
        # print("adding draw object \'{}\'".format(name))
        self.draw_objects[name] = draw_obj



    def remove_draw_object(self, name):
        try:
            self.draw_objects[name].cleanup(self.canvas)
            del self.draw_objects[name]
            # print("removing draw object \'{}\'".format(name))
        except KeyError:
            pass


    def remove_draw_objects_glob(self, pattern):
        for matched_name in self.find_draw_objects_glob(pattern):
            self.remove_draw_object(matched_name)



    def find_draw_objects_glob(self, pattern):
        return list(name for name in self.draw_objects.keys() \
            if fnmatch.fnmatch(name, pattern))


    def change_active_tool(self, tool):
        self.remove_draw_objects_glob("tool_helpers/*")
        self.active_tool = tool(self)


    def _left_up(self, event):
        self.pan_mode = False
        self.remove_draw_object('pan_marker')
        
        mods = get_event_modifiers(event)

        # do something only if no modifiers
        if len(mods) == 0:
            self.active_tool.left_click(event)


    def _right_up(self, event):
        self.rotate_mode = False
        self.remove_draw_object('rotate_marker')

        mods = get_event_modifiers(event)

        # do something only if no modifiers
        if len(mods) == 0:
            self.active_tool.right_click(event)



    def _ctrl_left_down(self, event):
        self.pan_mode = True
        pointer_world_crds = self.get_world_crds(
            event.x, event.y)

        self.add_draw_object(
            'pan_marker',
            PlusView(pointer_world_crds, size=15, color='yellow')
        )



    def _ctrl_right_down(self, event):
        self.rotate_mode = True
        # remember world crds of the marker so that we can rotate camera around it
        self.rot_marker_world = self.get_world_crds(event.x, event.y)

        self.add_draw_object(
            'rotate_marker',
            PlusView(self.rot_marker_world, size=15, color='red')
        )



    def _mouse_moved(self, event):
        delta_x = event.x - self.last_x
        delta_y = event.y - self.last_y

        last_pointer_world = self.get_world_crds(self.last_x, self.last_y)
        pointer_world = self.get_world_crds(event.x, event.y)

        self.last_x = event.x
        self.last_y = event.y


        ptr_over_poly_now = False
        for poly in self._polygons:
            if poly.point_inside(pointer_world):
                ptr_over_poly_now = True
                break

        if ptr_over_poly_now and not self.pointer_over_poly:
            self.canvas.config(cursor='center_ptr')
        elif not ptr_over_poly_now and self.pointer_over_poly:
            self.canvas.config(cursor='left_ptr')
        self.pointer_over_poly = ptr_over_poly_now


        # tell the active tool that mouse has moved
        self.active_tool.mouse_moved(event, delta_x, delta_y)

        # draw camera UI only when in camera control mode:
        if self.rotate_mode or self.pan_mode or self.zoom_mode:

            if self.pan_mode:

                delta = (
                    Matrix.rotate2d((0,0), self.camera_rot)
                    .dot(Matrix.scale2d((0,0), (self.camera_size, self.camera_size)))
                    .dot(v2col(delta_x, delta_y))
                )

                self.camera_pos -= vec(*delta[:2])


            elif self.rotate_mode:
                angle = 0.003*delta_x
                self.camera_rot += angle

                # make camera rotate around the marker rather than screen center
                rot_mtx = Matrix.rotate2d(self.rot_marker_world, angle)
                camera_new_pos = rot_mtx.dot(v2col(*self.camera_pos[:2]))
                self.camera_pos = vec(*camera_new_pos[:2])

            self.draw_all()


    def _windows_mousewheel(self, event):
        if event.delta > 0:
            self._mousewheel_up(event)
        else:
            self._mousewheel_down(event)


    def _mousewheel_up(self, event):
        self._scale_around_pointer(1./self.zoom_rate, event.x, event.y)
        

    def _mousewheel_down(self, event):
        self._scale_around_pointer(self.zoom_rate, event.x, event.y)


    def _zoom_in(self, event):
        x = self.canvas.winfo_pointerx()
        y = self.canvas.winfo_pointery()
        self._scale_around_pointer(.8/self.zoom_rate, x, y)


    def _zoom_out(self, event):
        x = self.canvas.winfo_pointerx()
        y = self.canvas.winfo_pointery()
        self._scale_around_pointer(1.25*self.zoom_rate, x, y)


    def _scale_around_pointer(self, rate, x_screen, y_screen):
        self.camera_size *= rate

        scale_cntr_world = self.get_world_crds(x_screen, y_screen)
        camera_new_pos = (
            Matrix.scale2d(scale_cntr_world, (rate, rate))
            .dot(v2col(*self.camera_pos[:2]))
        )
        self.camera_pos = vec(*camera_new_pos[:2])
        self.draw_all()



    def get_world_to_screen_mtx(self):
        '''
        Apply the opposite transformations from those of the camera
        1. Un-translate
        2. Un-rotate
        3. Un-zoom
        4. un-offset
        '''

        tran_mtx = Matrix.translate2d(-self.camera_pos)

        rot_mtx = Matrix.rotate2d((0,0), -self.camera_rot)

        # un-zoom
        zoom_mtx = Matrix.scale2d((0,0), (1./self.camera_size, 1./self.camera_size))

        offset_mtx = Matrix.translate2d(self.canvas_center)

        return offset_mtx.dot(zoom_mtx).dot(rot_mtx).dot(tran_mtx)



    def get_screen_to_world_mtx(self):
        '''
        Apply the same transformations that we applied to the camera
        1. Offset
        2. Zoom
        3. Rotate
        4. Translate
        '''

        # offset coords to move origin to center of screen
        offset_mtx = Matrix.translate2d(-self.canvas_center)

        # zoom
        zoom_mtx = Matrix.scale2d((0,0), (self.camera_size, self.camera_size))
        # zoom_mtx = Matrix.identity(3)

        # rotate
        rot_mtx = Matrix.rotate2d((0,0), self.camera_rot)

        # translate
        tran_mtx = Matrix.translate2d(self.camera_pos)

        return tran_mtx.dot(rot_mtx).dot(zoom_mtx).dot(offset_mtx)




    def get_world_crds(self, screen_x, screen_y):
        world_crds = self.get_screen_to_world_mtx().dot(v2col(screen_x, screen_y))
        return vec(world_crds[0], world_crds[1])



    def get_screen_crds(self, world_x, world_y):
        screen_crds = self.get_world_to_screen_mtx().dot(v2col(world_x, world_y))
        return vec(screen_crds[0], screen_crds[1])


    def _do_boolean_ops(self, mode, polygons, poly_b):
        new_polys = []

        if mode == Bool.Subtract:
            for poly_a in polygons:
                try:
                    subtracted = bool_subtract(poly_a, poly_b, self.debug_canvas)
                except AnyError as err:
                    self.save_polygons()
                    print("Exception was thrown inside method 'bool_subtract': {}".format(err))
                    print(traceback.format_exc())
                    subtracted = (poly_a,)

                new_polys.extend(subtracted)

        elif mode == Bool.Add:
            for poly_a in polygons:
                # add 2 polys, get either 1 or 2 polys
                try:
                    added = bool_add(poly_a, poly_b, self.debug_canvas)
                except AnyError as err:
                    self.save_polygons()
                    print("Exception was thrown inside method 'bool_add': {}".format(err))
                    print(traceback.format_exc())
                    added = poly_a, poly_b

                if len(added) == 1:
                    (poly_b,) = added
                else:
                    new_polys.append(poly_a)

            new_polys.append(poly_b)

        return new_polys


    def add_polygon(self, vertices):
        mode = self.get_bool_mode()

        self.remove_draw_objects_glob('polys/*')
        self.remove_draw_objects_glob('debug_*')

        self._polygons = self._do_boolean_ops(mode, self._polygons, Polygon2d(vertices[:]))

        del self.undo_stack[len(self.undo_stack) - self.undo_depth :]
        self.undo_depth = 0
        self.undo_stack.append(self._polygons)


        self.navmeshes = []
        # draw navmeshes
        for poly in self._polygons:
            num_polys = len(self.find_draw_objects_glob('polys/*'))
            navmesh = Mesh2d(poly, RELAX_ANGLE, self.debugger)
            self.add_draw_object('polys/poly_{}'.format(num_polys), NavMeshView(navmesh))
            self.navmeshes.append(navmesh)
            # self.add_draw_object('polys/poly_{}'.format(num_polys), PolygonView(poly))


    def draw_all(self):
        camera_trans = self.get_world_to_screen_mtx()

        # draw debug objects on top
        for name in self.find_draw_objects_glob('debug_*'):
            self.canvas.tag_raise(self.draw_objects[name].tag)

        for draw_object in self.draw_objects.values():
            draw_object.draw_self(camera_trans, self.canvas)


def read_polygons(poly_path, scale):
    try:
        with open(poly_path, 'r') as stream:
            return list(ascii_to_polys(stream.read(), scale))
           
    except IOError:
        print("given file cannot be opened")
        return ()


def subdivide_polygon(polygon, convex_threshold, debugger=None):
    try:
        return Mesh2d(polygon, convex_threshold, debugger)

    except AnyError as err:
        print("error occured during navmesh construction: {}".format(err))
        print(traceback.format_exc())


class ProvideException(object):
    def __init__(self, func):
        self._func = func

    def __call__(self, *args):

        try:
            return self._func(*args)

        except Exception as e:
            print('Exception was thrown: {}'.format(e))
            print(traceback.format_exc())
            # Optionally raise your own exceptions, popups etc


def list_dir(dir_path):
    if os.path.isfile(dir_path):
        yield dir_path
    elif os.path.isdir(dir_path):
        for n in os.listdir(dir_path):
            n = os.path.join(dir_path, n)
            if os.path.isfile(n):
                yield n
    else:
        raise RuntimeError("{} - no such file or directory".format(dir_path))


@ProvideException
def main():
    parser = ArgumentParser()

    parser.add_argument('input', metavar="FILE", nargs='?', default=None, type=str, help='open this polygon file')
    parser.add_argument('-d', '--debug', action='store_true', help='start in debug mode')
    parser.add_argument('-z', '--no-gui', action='store_true', help='open file and perform subdivision without GUI')
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--test-dir', type=str, default="", help='open all files in the given directory for testing')

    args = parser.parse_args()

    # test mode
    if args.test_dir != "":
        for name in sorted(list_dir(args.test_dir)):
            print("testing {}".format(name))
            input_polygons = read_polygons(name, 1.)
            for poly in input_polygons:
                subdivide_polygon(poly, RELAX_ANGLE)
        return

    input_polygons = read_polygons(args.input, args.scale) if args.input else ()

    # from itertools import chain
    # verts = chain(*(p.vertices for p in input_polygons))
    # bbox = vec.aabb(verts)
    # offset = vec(bbox[1][0], 0)

    camera_pos = vec(473.95899454023555, -18.10075326094288)*2097152 # FOR DEBUG of problem15.txt
    camera_pos = vec(0,0)

    if args.no_gui:
        for poly in input_polygons:
            subdivide_polygon(poly, RELAX_ANGLE)
        return

    else:
        app = Application(db_mode=args.debug, input_polys=input_polygons, camera_pos=camera_pos)
        app.master.title('Map editor')
        app.mainloop()

if __name__ == '__main__':
    main()
