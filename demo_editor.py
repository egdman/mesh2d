import os
import random
import yaml
import fnmatch
import Tkinter as tk

from mesh2d import *
from views import *

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

    def mouse_moved(self, event, dx, dy):
        pass



class CreateWall(Tool):
    def __init__(self, parent):
        self.width_mode = False
        self.start = None
        self.end = None
        self.width = 10.
        super(CreateWall, self).__init__(parent)


    # toggle width adjustment mode
    def right_click(self, event):
        if self.start is None: return
        self.width_mode = not self.width_mode

        # When exiting width adjustment mode, snap cursor back to endpoint.
        # This might not work on all platforms.
        if not self.width_mode:
            endpt_screen = self.parent.get_screen_crds(self.end.x, self.end.y)
            self.parent.canvas.event_generate('<Motion>', warp=True,
                x=endpt_screen.x, y=endpt_screen.y)

        self.redraw()


    # if not in window adj mode, set start- and end-points
    def left_click(self, event):
        app = self.parent
        if not self.width_mode:
            click_world = app.get_world_crds(event.x, event.y)

            if self.start is None:
                self.start = click_world
                self.end = click_world + Vector2(1, 0)

                app.add_draw_object('wall_tool_helpers/wall_helper',
                    WallHelperView(self.start, self.end, self.width))

                app.draw_all()

            else:
                self.end = click_world
                app.add_wall(self.start, self.end, self.width)
                app.remove_draw_objects_glob('wall_tool_helpers/*')



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
        screen_c = self.parent.get_screen_crds(world_c.x, world_c.y)

        print ("mouse at {}, {} from event".format(event.x, event.y))
        print ("mouse at {} in world".format(world_c))
        print ("mouse at {} on screen".format(screen_c))

        # x = float( self.canvas.canvasx(event.x) )
        # y = float( self.canvas.canvasy(event.y) )
        # pointer = Vector2(x, y)

        # for poly in parent._polygons:
        #   if poly.point_inside(pointer): 



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
        self.camera_pos = Vector2(0,0)
        self.camera_rot = 0.
        self.camera_size = 1.
        self.rot_marker_world = Vector2(0, 0)

        self.zoom_rate = 1.04

        # draw big cross marker at the world coordinate origin
        self.add_draw_object(
            'origin_marker',
            OriginView(250)
        )

        self.draw_all()



    def createWidgets(self):
        self.canvas = tk.Canvas(self, background='#000000', width=1200, height=900,
            scrollregion=(0, 0, 1200, 900))

        self.canvas_center = Vector2(float(self.canvas['width']) / 2., float(self.canvas['height']) / 2.)


        # pack root window into OS window and make it fill the entire window
        self.pack(side = tk.LEFT, expand = True, fill = tk.BOTH)

        self.canvas.config(xscrollincrement=1, yscrollincrement=1)

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


        self.wallToolBtn = tk.Button(
            self,
            text='wa',
            command = self._create_wall_cb
        )
        self.wallToolBtn.pack()



    def add_draw_object(self, name, draw_obj):
        print("adding draw object \'{}\'".format(name))
        self.draw_objects[name] = draw_obj



    def remove_draw_object(self, name):
        try:
            self.draw_objects[name].cleanup(self.canvas)
            del self.draw_objects[name]
            print("removing draw object \'{}\'".format(name))
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
            OriginView(15, loc=pointer_world_crds, color='yellow')
        )



    def _ctrl_right_down(self, event):
        self.rotate_mode = True
        # remember world crds of the marker so that we can rotate camera around it
        self.rot_marker_world = self.get_world_crds(event.x, event.y)

        self.add_draw_object(
            'rotate_marker',
            OriginView(15, loc=self.rot_marker_world, color='red')
        )



    def _mouse_moved(self, event):
        delta_x = event.x - self.last_x
        delta_y = event.y - self.last_y

        self.last_x = event.x
        self.last_y = event.y

        # tell the active tool that mouse has moved
        self.active_tool.mouse_moved(event, delta_x, delta_y)

        # draw camera UI only when in camera control mode:
        if self.rotate_mode or self.pan_mode or self.zoom_mode:

            if self.pan_mode:

                # rotate and zoom delta to get correct pan direction:
                delta = (
                    Matrix.rotate2d((0,0), self.camera_rot)
                    .multiply(Matrix.scale2d((0,0), (self.camera_size, self.camera_size)))
                    .multiply(Vector2(delta_x, delta_y)).values
                )

                self.camera_pos -= Vector2(delta[0], delta[1])


            elif self.rotate_mode:
                angle = 0.008*delta_x

                self.camera_rot += angle

                # make camera rotate around the marker rather than screen center
                rot_center_world = Vector2(self.rot_marker_world.x, self.rot_marker_world.y)
                rot_mtx = Matrix.rotate2d(rot_center_world, angle)
                camera_new_pos = rot_mtx.multiply(self.camera_pos).values
                self.camera_pos = Vector2(camera_new_pos[0], camera_new_pos[1])

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
            .multiply(self.camera_pos).values
        )
        self.camera_pos = Vector2(camera_new_pos[0], camera_new_pos[1])

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
        screen_crds = Vector2(screen_x, screen_y)
        world_crds = self.get_screen_to_world_mtx().multiply(screen_crds).values
        return Vector2(world_crds[0], world_crds[1])



    def get_screen_crds(self, world_x, world_y):
        world_crds = Vector2(world_x, world_y)
        screen_crds = self.get_world_to_screen_mtx().multiply(world_crds).values
        return Vector2(screen_crds[0], screen_crds[1])



    def _add_vertex(self, event):
        # reflect y to transform into right-hand coordinates
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

        new_poly = Polygon2d(self._new_vertices[:], range(len(self._new_vertices)))
        del self._new_vertices[:]

        sinters = new_poly.find_self_intersections()
        print("{} sinters".format(len(sinters)))

        num_polys = len(self.find_draw_objects_glob('polys/*'))
        self.add_draw_object('polys/poly_{}'.format(num_polys),
            PolygonView(new_poly))

        # remove helper views
        self.remove_draw_objects_glob('obj_creation_helpers/*')

        for si_num, sinter in enumerate(sinters):
            pt = sinter[2]
            self.add_draw_object('poly_sinters/sinter_{}_{}'.format(num_polys, si_num),
                OriginView(16, loc=pt, color='cyan'))

        # set current tool on 'Select'
        self.active_tool = self.select_tool

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
            portals = new_poly.break_into_convex(threshold)
            polys = new_poly.get_pieces_as_meshes()

            print ("number of portals      = {0}".format(len(portals)))
            print ("number of convex parts = {0}".format(len(polys)))

            if len(polys) != len(portals) + 1:
                print ("Error!")
                error_dump(self.last_created_poly)
    
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
        self.active_tool = self.select_tool

        self.draw_all()



    def add_wall(self, start, end, width):
        self.active_tool = self.select_tool
        print ("added wall {}, {}, {}".format(start, end, width))



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

        except Exception, e:
            print('Exception was thrown: {}'.format(e))
            # Optionally raise your own exceptions, popups etc


@ProvideException
def main():
    app = Application()
    app.master.title('Map editor')
    app.mainloop() 



if __name__ == '__main__':
    main()