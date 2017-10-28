import os
import random
import yaml
import fnmatch
import Tkinter as tk
from argparse import ArgumentParser

from mesh2d import *
from views import *

pkg_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(pkg_dir, 'resources')
button_dir = os.path.join(resource_dir, 'buttons')

parser = ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='Start in debug mode')

def v2col(v):
    return Matrix.column_vec(v.append(1))

class Tool(object):

    def __init__(self, parent):
        self.parent = parent

    def right_click(self, event):
        pass

    def left_click(self, event):
        pass

    def mouse_moved(self, event, dx, dy):
        pass



class CreateWall(Tool):
    def __init__(self, parent):
        self.width_mode = False
        self.start = None
        self.end = None
        self.width = 1.
        super(CreateWall, self).__init__(parent)


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


    # if not in window adj mode, set start- and end-points
    def left_click(self, event):
        app = self.parent
        if not self.width_mode:
            click_world = app.get_world_crds(event.x, event.y)

            if self.start is None:
                self.start = click_world
                self.end = click_world + vec(1, 0)

                app.add_draw_object('wall_tool_helpers/wall_helper',
                    WallHelperView(self.start, self.end, self.width))

                app.draw_all()

            else:
                self.end = click_world
                app.add_wall(self.start, self.end, self.width)
                # app.remove_draw_objects_glob('wall_tool_helpers/*')
                self.width_mode = False
                self.start = self.end
                self.end = None



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
        app.draw_objects['wall_tool_helpers/wall_helper'].modify(
            self.start, self.end, self.width)
        app.draw_all()



class Create(Tool):

    def right_click(self, event):
        # self.parent._add_navmesh(event)
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

        world_c = self.parent.get_world_crds(event.x, event.y)
        screen_c = self.parent.get_screen_crds(world_c[0], world_c[1])

        print ("mouse at {}, {} from event".format(event.x, event.y))
        print ("mouse at {} in world".format(world_c))
        print ("mouse at {} on screen".format(screen_c))

        # x = float( self.canvas.canvasx(event.x) )
        # y = float( self.canvas.canvasy(event.y) )
        # pointer = vec(x, y)

        # for poly in parent._polygons:
        #   if poly.point_inside(pointer): 



class Application(tk.Frame):
    def __init__(self, master=None, db_mode = False):
        tk.Frame.__init__(self, master)
        self.db_mode = db_mode

        self.grid()
        self.createWidgets()

        self._new_vertices = []
        self._polygons = []
        self._dots_ids = []
        self.dot_size = 3

        self.select_tool = Select(self)
        self.create_tool = Create(self)
        # ......
        # ......

        self.active_tool = self.create_tool

        self.last_created_poly = None

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
        self.canvas.bind('<Button-4>', self._mousewheel_up)
        self.canvas.bind('<Button-5>', self._mousewheel_down)


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


        self.wallToolIcon = tk.PhotoImage(file=os.path.join(button_dir, 'wall.gif'))
        self.wallToolBtn = tk.Button(
            self,
            image=self.wallToolIcon,
            command = self._create_wall_cb
        )
        self.wallToolBtn.pack()

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



    def get_bool_mode(self):
        if self.plusBtn.cget('relief') == tk.SUNKEN:
            return 'add'
        return 'subtract'



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



    def _save_last(self):
        with open("saved_poly.yaml", 'w') as savef:
            yaml.dump(self.last_created_poly, savef)


    def _create_wall_cb(self):
        self.active_tool = CreateWall(self)


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
        self.remove_draw_object('pan_marker')
        
        mods = self._get_event_modifiers(event)

        # do something only if no modifiers
        if len(mods) == 0:
            self.active_tool.left_click(event)


    def _right_up(self, event):
        self.rotate_mode = False
        self.remove_draw_object('rotate_marker')

        mods = self._get_event_modifiers(event)

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


        # find polygon vertices near the mouse pointer
        # self.remove_draw_objects_glob("highlight_points/*")
        # if len(self._polygons) > 0:
        #     ptr_world = self.get_world_crds(event.x, event.y)
        #     vect_nw = ptr_world - vec(20., 20.)
        #     vect_se = ptr_world + vec(20., 20.)

        #     num = 0
        #     for poly in self._polygons:
        #         idxs = poly.find_verts_in_bbox(vect_nw, vect_se)
        #         for idx in idxs:
        #             vect = poly.vertices[idx]
        #             self.add_draw_object("highlight_points/{}".format(num),
        #                 PointHelperView(loc = vect, color = 'green'))
        #             num += 1
        #     if num > 0: self.draw_all()

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

                # rotate and zoom delta to get correct pan direction:
                delta = (
                    Matrix.rotate2d((0,0), self.camera_rot)
                    .multiply(Matrix.scale2d((0,0), (self.camera_size, self.camera_size)))
                    .multiply(v2col(vec(delta_x, delta_y))).values
                )

                self.camera_pos -= vec(delta[0], delta[1])


            elif self.rotate_mode:
                angle = 0.008*delta_x

                self.camera_rot += angle

                # make camera rotate around the marker rather than screen center
                rot_center_world = vec(self.rot_marker_world[0], self.rot_marker_world[0])
                rot_mtx = Matrix.rotate2d(rot_center_world, angle)
                camera_new_pos = rot_mtx.multiply(v2col(self.camera_pos)).values
                self.camera_pos = vec(camera_new_pos[0], camera_new_pos[1])

            self.draw_all()





    def _mousewheel_up(self, event):
        self._scale_around_pointer(1./self.zoom_rate, event)
        

    def _mousewheel_down(self, event):
        self._scale_around_pointer(self.zoom_rate, event)


    def _scale_around_pointer(self, rate, event):
        self.camera_size *= rate

        scale_cntr_world = self.get_world_crds(event.x, event.y)
        camera_new_pos = (
            Matrix.scale2d(scale_cntr_world, (rate, rate))
            .multiply(v2col(self.camera_pos)).values
        )
        self.camera_pos = vec(camera_new_pos[0], camera_new_pos[1])

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
        # zoom_mtx = Matrix.identity(3)

        offset_mtx = Matrix.translate2d(self.canvas_center)

        return offset_mtx.multiply(zoom_mtx).multiply(rot_mtx).multiply(tran_mtx)



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

        return tran_mtx.multiply(rot_mtx).multiply(zoom_mtx).multiply(offset_mtx)




    def get_world_crds(self, screen_x, screen_y):
        screen_crds = vec(screen_x, screen_y)
        world_crds = self.get_screen_to_world_mtx().multiply(v2col(screen_crds)).values
        return vec(world_crds[0], world_crds[1])



    def get_screen_crds(self, world_x, world_y):
        world_crds = vec(world_x, world_y)
        screen_crds = self.get_world_to_screen_mtx().multiply(v2col(world_crds)).values
        return vec(screen_crds[0], screen_crds[1])



    def _add_vertex(self, event):
        x = event.x
        y = event.y
        new_vrt = self.get_world_crds(event.x, event.y)

        self.add_draw_object(
            'obj_creation_helpers/point_{}'.format(len(self._new_vertices)),
            PointHelperView(loc=new_vrt))
        
        self._new_vertices.append(new_vrt)

        if len(self._new_vertices) > 1:
            prev_vrt = self._new_vertices[-2]

            self.add_draw_object(
                'obj_creation_helpers/segment_{}'.format(len(self._new_vertices)),
                SegmentHelperView(prev_vrt, new_vrt)
            )

        self.draw_all()



    def _add_polygon(self, event):
        if len(self._new_vertices) < 3: return

        mode = self.get_bool_mode()

        # remove helper views
        self.remove_draw_objects_glob('obj_creation_helpers/*')

        # # delete all polygon views (we'll add them after boolean operation)
        # self.remove_draw_objects_glob('polys/*')

        # new_poly = Polygon2d(self._new_vertices[:], range(len(self._new_vertices)))
        new_poly = Polygon2d(self._new_vertices[:], range(len(self._new_vertices)))

        del self._new_vertices[:]

        new_polys = [new_poly]

        # if mode == 'subtract' and len(self._polygons) > 0:
        #     new_polys = bool_subtract(self._polygons[-1], new_poly)
            

        # elif mode == 'add':
        #     if len(self._polygons) == 0:
        #         new_polys = [new_poly]
        #     else:
        #         new_polys = bool_add(self._polygons[-1], new_poly)

        # else:
        #     return

        del self._polygons[:]
        for poly in new_polys:
            poly = Mesh2d.from_polygon(poly)
            poly.break_into_convex(10., self.debug_canvas)
            num_polys = len(self.find_draw_objects_glob('polys/*'))
            self.add_draw_object('polys/poly_{}'.format(num_polys),
                NavMeshView(poly))

            self._polygons.append(poly)

        # set current tool on 'Select'
        # self.active_tool = self.select_tool

        self.draw_all()




    def _add_navmesh(self, event):
        if len(self._new_vertices) < 3: return

        threshold = 10.0 # degrees

        new_poly = Mesh2d(self._new_vertices[:], range(len(self._new_vertices)))

        # to save in case of failure
        self.last_created_poly = Mesh2d(self._new_vertices[:], range(len(self._new_vertices)))

        del self._new_vertices[:]

        # break into convex parts:

        def error_dump(poly):
            with open("debug_dump.yaml", 'w') as debf:
                yaml.dump(poly, debf)           

        try:
            new_poly.break_into_convex(threshold, self.debug_canvas)

            print ("number of portals      = {0}".format(len(new_poly.portals)))
            print ("number of convex rooms = {0}".format(len(new_poly.rooms)))

            if len(new_poly.rooms) != len(new_poly.portals) + 1:
                print ("Error!")
                error_dump(new_poly)
    
        except ValueError as ve:
            error_dump(self.last_created_poly)
            raise ve

        self._polygons.append(new_poly)

        # add navmesh view
        self.add_draw_object(
            'navmesh_{}'.format(len(self._polygons)),
            NavMeshView(new_poly)
        )

        # remove helper views
        self.remove_draw_objects_glob('obj_creation_helpers/*')

        # set current tool on 'Select'
        # self.active_tool = self.select_tool

        self.draw_all()



    def add_wall(self, start_vec, end_vec, width):
        # self.active_tool = self.select_tool
        # print ("added wall {}, {}, {}".format(start, end, width))
        class cc: pass
        start, end = cc(), cc()
        start.x, start.y = start_vec.comps
        end.x, end.y = end_vec.comps

        # add placeholder code to test polys with holes
        if start.x > end.x: start.x, end.x = end.x, start.x
        if start.y > end.y: start.y, end.y = end.y, start.y
        v0 = vec(start.x, start.y)
        v1 = vec(end.x, start.y)
        v2 = vec(end.x, end.y)
        v3 = vec(start.x, end.y)

        width = end.x - start.x
        height = end.y - start.y
        cntr_x = start.x + width / 2.
        cntr_y = start.y + height / 2.
        cntr = vec(cntr_x, cntr_y)

        sz = min(height, width) / 2.

        h10 = (vec(.1, .9) - vec(.5, .5) )*sz + cntr
        h11 = (vec(.1, .1) - vec(.5, .5) )*sz + cntr
        h12 = (vec(.4, .1) - vec(.5, .5) )*sz + cntr
        h13 = (vec(.4, .9) - vec(.5, .5) )*sz + cntr

        h20 = (vec(.9, .9) - vec(.5, .5) )*sz + cntr
        h21 = (vec(.5, .5) - vec(.5, .5) )*sz + cntr
        h22 = (vec(.9, .1) - vec(.5, .5) )*sz + cntr

        h30 = (vec(.55, .72) - vec(.5, .5) )*sz + cntr
        h31 = (vec(.79, .92) - vec(.5, .5) )*sz + cntr
        h32 = (vec(.60, .85) - vec(.5, .5) )*sz + cntr


        new_poly = Mesh2d([v0, v1, v2, v3], range(4))
        new_poly.add_hole([h10, h11, h12, h13])
        new_poly.add_hole([h20, h21, h22])
        new_poly.add_hole([h30, h31, h32])

        cv = self.debug_canvas
        for cid in cv.find_all(): cv.delete(cid)

        new_poly.break_into_convex(10., self.debug_canvas)
        self._polygons.append(new_poly)

        num_polys = len(self.find_draw_objects_glob('polys/*'))
        self.add_draw_object('polys/poly_{}'.format(num_polys),
            NavMeshView(new_poly))

        self.draw_all()



    def draw_all(self):
        camera_trans = self.get_world_to_screen_mtx()

        for obj_name in self.draw_objects:
            self.draw_objects[obj_name].draw_self(camera_trans, self.canvas)




class ProvideException(object):
    def __init__(self, func):
        self._func = func

    def __call__(self, *args):

        try:
            return self._func(*args)

        except Exception as e:
            print('Exception was thrown: {}'.format(e))
            # Optionally raise your own exceptions, popups etc


@ProvideException
def main():

    args = parser.parse_args()

    app = Application(db_mode=args.debug)
    app.master.title('Map editor')
    app.mainloop() 



if __name__ == '__main__':
    main()
