import os
import random
import fnmatch
import platform
import struct
import Tkinter as tk
from argparse import ArgumentParser
import traceback
from mesh2d import *
from views import *

pkg_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(pkg_dir, 'resources')
button_dir = os.path.join(resource_dir, 'buttons')

try:
    AnyError = StandardError
except NameError:
    AnyError = Exception

class RecordType:
    class AddVertex: pass
    class AddPoly: pass

def v2col(x, y):
    return Matrix.column_vec((x, y, 1.))

def poly_to_ascii(poly):
    s = ""
    loops = ((poly.vertices[idx] for idx in poly.graph.loop_iterator(loop)) for loop in poly.graph.loops)
    for loop in loops:
        s += " ".join("{} {}".format(v[0], v[1]) for v in loop)
        s += "\n"
    return s

def ascii_to_poly(text):
    loops = []
    for line in text.splitlines():
        loop = []
        loops.append(loop)
        crds = (float(el.strip()) for el in line.strip().split())
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


def ieee754_ser(floatNumber):
    return hex(struct.unpack('<Q', struct.pack('<d', floatNumber))[0])[:-1]

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
        super(FreePolyTool, self).__init__(parent)

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

        try:
            app.add_polygon(self.vertices)
        except AnyError:
            dump_history(app.history)


        del self.vertices[:]
        # remove helper views
        app.remove_draw_objects_glob('tool_helpers/free_poly/*')
        app.draw_all()


    def left_click(self, event):
        app = self.parent
        new_vrt = app.get_world_crds(event.x, event.y)

        app.history.append((RecordType.AddVertex, new_vrt))
        app.add_draw_object(
            'tool_helpers/free_poly/point_{}'.format(len(self.vertices)),
            PointHelperView(loc=new_vrt))

        self.vertices.append(new_vrt)

        if len(self.vertices) > 1:
            prev_vrt = self.vertices[-2]

            app.add_draw_object(
                'tool_helpers/free_poly/segment_{}'.format(len(self.vertices)),
                SegmentHelperView(prev_vrt, new_vrt))

        app.draw_all()


class SelectTool(Tool):

    def right_click(self, event):
        print("SELECT: right click")


    def left_click(self, event):
        # print("SELECT: left click")
        obj_ids = self.parent.canvas.find_overlapping(event.x, event.y, event.x, event.y)
        print (obj_ids)

        world_c = self.parent.get_world_crds(event.x, event.y)
        screen_c = self.parent.get_screen_crds(world_c[0], world_c[1])

        print ("mouse at {}, {} from event".format(event.x, event.y))
        print ("mouse at {} in world".format(world_c))
        print ("mouse at {} on screen".format(screen_c))


def get_event_modifiers(event):
    state = [
        ('ctrl',  event.state & 0x0004),
        ('ralt',  event.state & 0x0080),
        ('shift', event.state & 0x0001),
    ]
    return set(modname for (modname, hit) in state if hit != 0)

class Bool:
    class Subtract: pass
    class Add: pass


class Application(tk.Frame):
    def __init__(self, master=None, poly_path=None, db_mode=False):
        tk.Frame.__init__(self, master)
        self.db_mode = db_mode
        self.history = []
        self.this_is_windows = "windows" in platform.system().lower()

        self.grid()
        self.createWidgets()

        self._polygons = []
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
        self.camera_pos = vec(0,0)
        self.camera_rot = 0.
        self.camera_size = 1.
        self.rot_marker_world = vec(0, 0)

        self.zoom_rate = 1.04

        # draw big cross marker at the world coordinate origin
        self.add_draw_object(
            'origin_marker',
            PlusView(250)
        )

        self.pointer_over_poly = False

        if poly_path is not None:
            try:
                with open(poly_path, 'r') as stream:
                    poly = ascii_to_poly(stream.read())
                    self._polygons.append(poly)
                    navmesh = Mesh2d(poly, 15)
                    num_polys = len(self.find_draw_objects_glob('polys/*'))
                    self.add_draw_object('polys/poly_{}'.format(num_polys),
                        NavMeshView(navmesh))
            except IOError:
                print("given file cannot be opened")

        self.draw_all()



    def createWidgets(self):
        cwidth = 1200
        cheight = 900
        if self.db_mode: cwidth /= 2


        self.canvas = tk.Canvas(self, background='#000000', width=cwidth, height=cheight,
            scrollregion=(0, 0, cwidth, cheight))

        self.canvas_center = vec(float(self.canvas['width']) / 2., float(self.canvas['height']) / 2.)


        # pack root window into OS window and make it fill the entire window
        self.pack(side = tk.LEFT, expand = True, fill = tk.BOTH)

        self.debug_canvas = tk.Canvas(self, background='#000020', width=cwidth, height=cheight,
            scrollregion=(-1000, -1000, 1000, 1000))
        self.debug_canvas.scan_dragto(cwidth/2, cheight/2, gain=1)
        
        if self.db_mode:
            self.debug_canvas.pack(side = tk.RIGHT, expand = True, fill = tk.BOTH)


        self.canvas.config(xscrollincrement=1, yscrollincrement=1)

        self.debug_canvas.bind('<Button-1>', lambda ev: self.debug_canvas.scan_mark(ev.x, ev.y))
        self.debug_canvas.bind('<Motion>', self._debug_pan_mouse)

        # bind right and left mouse clicks
        self.canvas.bind('<ButtonRelease-1>', self._left_up)
        self.canvas.bind('<ButtonRelease-3>', self._right_up)


        # bind camera controls
        self.canvas.bind('<Control-Button-1>', self._ctrl_left_down)
        self.canvas.bind('<Control-Button-3>', self._ctrl_right_down)
        self.canvas.bind('<Motion>', self._mouse_moved)


        # mouse wheel
        if self.this_is_windows:
            self.canvas.bind_all('<MouseWheel>', self._windows_mousewheel)
        else:
            self.canvas.bind_all('<Button-4>', self._mousewheel_up)
            self.canvas.bind_all('<Button-5>', self._mousewheel_down)

        

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
            command = self.save_polygon)
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


    def save_polygon(self):
        dump_history(self.history)
        if self._polygons:
            with open("poly.txt", 'w') as stream:
                stream.write(poly_to_ascii(self._polygons[0]))


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
            PlusView(15, loc=pointer_world_crds, color='yellow')
        )



    def _ctrl_right_down(self, event):
        self.rotate_mode = True
        # remember world crds of the marker so that we can rotate camera around it
        self.rot_marker_world = self.get_world_crds(event.x, event.y)

        self.add_draw_object(
            'rotate_marker',
            PlusView(15, loc=self.rot_marker_world, color='red')
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
                angle = 0.008*delta_x
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
        self._scale_around_pointer(1./self.zoom_rate, event)
        

    def _mousewheel_down(self, event):
        self._scale_around_pointer(self.zoom_rate, event)


    def _scale_around_pointer(self, rate, event):
        self.camera_size *= rate

        scale_cntr_world = self.get_world_crds(event.x, event.y)
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


    def add_polygon(self, vertices):
        mode = self.get_bool_mode()

        # delete all polygon views (we'll add them after boolean operation)
        self.remove_draw_objects_glob('polys/*')

        new_poly = Polygon2d(vertices[:])

        # do boolean operations
        new_polys = []

        if mode == Bool.Subtract:
            for old_poly in self._polygons:
                new_polys.extend(bool_subtract(old_poly, new_poly, self.debug_canvas))
            
        elif mode == Bool.Add:
            while len(self._polygons):
                old_poly = self._polygons.pop()

                # add 2 polys, get either 1 or 2 polys
                added = bool_add(old_poly, new_poly, self.debug_canvas)
                if len(added) == 1:
                    (new_poly,) = added
                else:
                    new_polys.append(old_poly)

            new_polys.append(new_poly)

        else:
            return

        self._polygons = new_polys

        # draw navmeshes
        for poly in self._polygons:
            navmesh = Mesh2d(poly, 15)
            num_polys = len(self.find_draw_objects_glob('polys/*'))
            self.add_draw_object('polys/poly_{}'.format(num_polys),
                NavMeshView(navmesh))

        # # draw polygons
        # for poly in self._polygons:
        #     num_polys = len(self.find_draw_objects_glob('polys/*'))
        #     self.add_draw_object('polys/poly_{}'.format(num_polys),
        #         PolygonView(poly))


    def draw_all(self):
        camera_trans = self.get_world_to_screen_mtx()

        for draw_object in self.draw_objects.values():
            draw_object.draw_self(camera_trans, self.canvas)




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


@ProvideException
def main():
    parser = ArgumentParser()

    parser.add_argument('-d', '--debug', action='store_true', help='Start in debug mode')
    parser.add_argument('-p', '--poly', type=str, default=None, help='open this polygon file')

    args = parser.parse_args()

    app = Application(db_mode=args.debug, poly_path=args.poly)
    app.master.title('Map editor')
    app.mainloop() 



if __name__ == '__main__':
    main()
