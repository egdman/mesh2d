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


class ObjectView(object):
    def __init__(self, tags=None):
        self.crd_buf = []
        self.element_ids = []

        if tags is not None:
            self.tags = tags
        else:
            self.tags = []



    def draw_self(self, camera_transform, canvas):
        if len(self.crd_buf) == 0:
            self.first_time_draw(canvas)

            # remember coordinate buffer
            for eid in self.element_ids:
                self.crd_buf.extend(canvas.coords(eid))

            print("new object with {} coordinates created".format(len(self.crd_buf)))

        self.redraw(camera_transform, canvas)



    def first_time_draw(self, canvas):
        '''
        first time draw (add elements to canvas)
        '''
        raise NotImplementedError("Implement method 'first_time_draw' in your subclass")



    def redraw(self, camera_transform, canvas):
        obj_view = camera_transform
            
        obj_vertices = []
        for idx in range(len(self.crd_buf)/2):
            x = self.crd_buf[2*idx]
            y = self.crd_buf[2*idx+1]
            obj_vertices.append(Vector2(x, y))

        screen_vertices = list(self.apply_transform(obj_view, obj_vertices))

        screen_vert_buf = []
        for screen_v in screen_vertices:
            screen_vert_buf.append(screen_v[0])
            screen_vert_buf.append(screen_v[1])

        notch = 0
        for eid in self.element_ids:
            num_crds = len(canvas.coords(eid))
            new_crds = screen_vert_buf[notch:notch+num_crds]
            canvas.coords(eid, *new_crds)
            notch += num_crds



    @staticmethod
    def get_open_crds(vertices, indices):
        crds = []

        for ind in indices:
            crds.append(vertices[ind][0])
            crds.append(vertices[ind][1])
        return crds


    @staticmethod
    def get_closed_crds(vertices, indices):
        crds = []

        for ind in indices:
            crds.append(vertices[ind][0])
            crds.append(vertices[ind][1])
        crds.append(vertices[indices[0]][0])
        crds.append(vertices[indices[0]][1])
        return crds


    @staticmethod
    def apply_transform(transform, vertices):
        return (transform.multiply(vert).values[:-1] for vert in vertices)



class PointHelperView(ObjectView):
    def __init__(self, loc, color='#ffffff', tags=None):
        self.loc = loc
        self.color = color
        super(PointHelperView, self).__init__(tags)



    def first_time_draw(self, canvas):
        sz = 3
        Id = canvas.create_polygon(
            (
                self.loc[0] - sz,
                self.loc[1] - sz,

                self.loc[0] + sz,
                self.loc[1] - sz,

                self.loc[0] + sz,
                self.loc[1] + sz,

                self.loc[0] - sz,
                self.loc[1] + sz,
            ),

            fill = self.color,
            tags = self.tags
        )
        self.element_ids.append(Id)



class SegmentHelperView(ObjectView):
    def __init__(self, vert1, vert2, color='#ffffff', width=1, tags=None):
        self.v1 = vert1
        self.v2 = vert2
        self.width = width

        self.color = color
        super(SegmentHelperView, self).__init__(tags)



    def first_time_draw(self, canvas):
        Id = canvas.create_line(
                (
                    self.v1[0], self.v1[1],
                    self.v2[0], self.v2[1]
                ),
                fill = self.color,
                width=self.width,
                tags = self.tags
        )
        self.element_ids.append(Id)





class OriginView(ObjectView):
    def __init__(self, size, tags=None, loc=None, color='#2f2f2f'):
        self.size = size
        self.color = color
        self.loc = loc if loc is not None else Vector2(0, 0)

        self.down = Vector2(0, self.size)
        self.right = Vector2(self.size, 0)

        super(OriginView, self).__init__(tags)




    def first_time_draw(self, canvas):
        sz = self.size

        vrt = [self.loc - self.down, self.loc + self.down, self.loc - self.right, self.loc + self.right]
        # vrt = [Vector2(0, -sz),Vector2(0, sz),Vector2(-sz, 0),Vector2(sz, 0),]
        # trans_vrt = list(self.apply_transform(camera_transform, vrt))
        crds = self.get_open_crds(vrt, (0, 1))
        id1 = canvas.create_line(crds, tags=self.tags, fill=self.color, width=1)
        crds = self.get_open_crds(vrt, (2, 3))
        id2 = canvas.create_line(crds, tags=self.tags, fill=self.color, width=1)
        self.element_ids.append(id1)
        self.element_ids.append(id2)




class NavMeshView(ObjectView):

    def __init__(self, navmesh, tags=None):
        rnd = lambda: random.randint(0,255)
        self.color = '#%02X%02X%02X' % (rnd(),rnd(),rnd())
        self.navmesh = navmesh
        self.obj_trans = Matrix.identity(3)

        super(NavMeshView, self).__init__(tags)



    def first_time_draw(self, canvas):

        trans_vertices = self.navmesh.vertices

        # draw pieces
        for piece in self.navmesh.pieces:
            piece_crds = self.get_open_crds(trans_vertices, piece)
            Id = canvas.create_polygon(piece_crds, tags=self.tags, fill='#1A1A1A', activefill='#111111')
            self.element_ids.append(Id)

        # draw portals
        for portal in self.navmesh.portals:
            portal_crds = self.get_open_crds(trans_vertices, portal)
            Id = canvas.create_line(portal_crds, tags=self.tags, fill='red', width=1)
            self.element_ids.append(Id)

        # draw outline
        outline_crds = self.get_closed_crds(trans_vertices, self.navmesh.indices)
        Id = canvas.create_line(outline_crds, tags=self.tags, fill='#FFFFFF', width=1)
        self.element_ids.append(Id)

                


class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

        self._new_vertices = []
        self._polygons = []
        self._dots_ids = []
        self.dot_size = 3


        # canvas object tags
        self.root_tag = 'root'
        self.camera_ui_tag = 'camera_ui'
        self.obj_creation_helper_tag = 'obj_creation_helper'



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

        # views that represent content objects
        self.object_views = []

        # views that represent helper objects
        self.helper_object_views = []

        # views that represent camera UI
        self.camera_pan_marker = None
        self.camera_rotate_marker = None
        self.camera_zoom_marker = None

        self.draw_objects = {}

        # info about camera position
        self.camera_pos = Vector2(0,0)
        self.camera_rot = 0.
        self.camera_target = Vector2(0,0)
        self.camera_zoom = 1.

        self.helper_object_views.append(OriginView(250, tags=self.root_tag))


        self.draw_all()



    def createWidgets(self):
        self.canvas = tk.Canvas(self, background='#000000', width=1000, height=900,
            scrollregion=(0, 0, 1000, 900))

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


        # mouse wheel
        self.canvas.bind('<Button-4>', self._mousewheel_up)
        self.canvas.bind('<Button-5>', self._mousewheel_down)


        self.hbar.pack(side = tk.BOTTOM, fill = tk.X)
        self.vbar.pack(side = tk.RIGHT, fill = tk.Y)
        self.canvas.pack(side = tk.LEFT, expand = True, fill = tk.BOTH)


        # butens
        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.quitButton.pack()


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



    def add_draw_object(self, name, draw_obj):
        self.draw_objects[name] = draw_obj



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
        self.camera_pan_marker = None
        self.canvas.delete('pan_marker')
        
        mods = self._get_event_modifiers(event)

        # do something only if no modifiers
        if len(mods) == 0:
            self.active_tool.left_click(event)


    def _right_up(self, event):
        self.rotate_mode = False
        self.camera_rotate_marker = None
        self.canvas.delete('rotate_marker')

        mods = self._get_event_modifiers(event)

        # do something only if no modifiers
        if len(mods) == 0:
            self.active_tool.right_click(event)



    def _ctrl_left_down(self, event):
        self.pan_mode = True
        # remember world crds of the screen center before panning
        self.camera_target = self.get_world_crds(
            self.canvas_center.x, self.canvas_center.y)

        pointer_world_crds = self.get_world_crds(
            event.x, event.y)

        self.camera_pan_marker = (
            OriginView(15,
                loc=pointer_world_crds,
                tags=(self.root_tag, self.camera_ui_tag, 'pan_marker'),
                color='yellow'
            )
        )



    def _ctrl_right_down(self, event):
        self.rotate_mode = True
        # remember world crds of the screen center before panning
        self.camera_target = self.get_world_crds(
            self.canvas_center.x, self.canvas_center.y)

        self.camera_rotate_marker = (
            OriginView(15,
                loc=self.camera_target,
                tags=(self.root_tag, self.camera_ui_tag, 'rotate_marker'),
                color='red'
            )
        )



    def _mouse_moved(self, event):
        
        # draw camera UI only when in camera control mode:
        if self.rotate_mode or self.pan_mode or self.zoom_mode:

            if self.pan_mode:
                delta_x = event.x - self.last_x
                delta_y = event.y - self.last_y

                # rotate delta to get correct pan direction:
                delta = Matrix.rotate2d((0,0), -self.camera_rot).multiply(
                    Vector2(delta_x, delta_y)).values

                self.camera_pos += Vector2(delta[0], delta[1])


            elif self.rotate_mode:
                delta_x = event.x - self.last_x

                angle = 0.008*delta_x

                # if event.y < self.canvas_center.y: angle*= -1

                self.camera_rot -= angle

                self.camera_target = self.get_world_crds(
                    self.canvas_center.x, self.canvas_center.y)

            self.draw_all()


        self.last_x = event.x
        self.last_y = event.y




    def _mousewheel_up(self, event):
        self.camera_zoom *= 1.04
        # self.draw_all()
        # pass

    def _mousewheel_down(self, event):
        self.camera_zoom /= 1.04
        # self.draw_all()
        # pass


    def _add_vertex(self, event):
        # reflect y to transform into right-hand coordinates
        x = event.x
        y = event.y
        new_vrt = self.get_world_crds(event.x, event.y)
        self._new_vertices.append(new_vrt)

        self.helper_object_views.append(PointHelperView(loc=new_vrt,
            tags=(self.root_tag, self.obj_creation_helper_tag)))

        # sz = self.dot_size
        
        if len(self._new_vertices) > 1:
            prev_v = self._new_vertices[-2]
            self.helper_object_views.append(
                SegmentHelperView(
                        prev_v, new_vrt,
                        tags=(self.root_tag, self.obj_creation_helper_tag)
                    )
                )

        self.draw_all()
       


    def get_world_crds(self, screen_x, screen_y):
        screen_crds = Vector2(screen_x, screen_y)

        # center becomes upper left corner
        screen_crds -= self.canvas_center

        # zoom TODO

        # then rotate
        rmtx = Matrix.rotate2d((0,0), -self.camera_rot)
        screen_crds = rmtx.multiply(screen_crds).values
        # then translate
        world_crds = Vector2(
            screen_crds[0] - self.camera_pos[0],
            screen_crds[1] - self.camera_pos[1]
        )
        
        return Vector2(world_crds[0], world_crds[1])



    def get_screen_crds(self, world_x, world_y):
        world_crds = Vector2(world_x, world_y)

        # translate then rotate
        camera_trans = (Matrix.rotate2d((0,0), self.camera_rot)
            .multiply(
                Matrix.translate2d((self.camera_pos))))

        # # then unzoom
        # camera_trans = (Matrix.scale2d((0, 0), (self.camera_zoom, self.camera_zoom))
        #     .multiply(camera_trans))

        # move the center of coordinates to the center of the canvas:
        camera_trans.loc[(0,2)] += self.canvas_center[0]
        camera_trans.loc[(1,2)] += self.canvas_center[1]

        screen_crds = camera_trans.multiply(world_crds).values
        return Vector2(screen_crds[0], screen_crds[1])



    def draw_all(self):
        # self.canvas.delete(self.root_tag)

        camera_trans = (Matrix.rotate2d((0,0), self.camera_rot)
            .multiply(
                Matrix.translate2d((self.camera_pos))))

        # camera_trans = (Matrix.scale2d((0, 0), (self.camera_zoom, self.camera_zoom))
        #     .multiply(camera_trans))

        # move the center of coordinates to the center of the canvas:
        camera_trans.loc[(0,2)] += self.canvas_center[0]
        camera_trans.loc[(1,2)] += self.canvas_center[1]

        for obj_name in self.draw_objects:
            draw_objects[obj_name].draw_self(camera_trans, self.canvas)


        # for v in self.object_views:
        #     v.draw_self(camera_trans, self.canvas)

        # for v in self.helper_object_views:
        #     v.draw_self(camera_trans, self.canvas)            

        # # for v in self.camera_ui_views:
        # if self.camera_pan_marker is not None:
        #     self.camera_pan_marker.draw_self(camera_trans, self.canvas)

        # if self.camera_rotate_marker is not None:
        #     self.camera_rotate_marker.draw_self(camera_trans, self.canvas)

        # if self.camera_zoom_marker is not None:
        #     self.camera_zoom_marker.draw_self(camera_trans, self.canvas)




    def _add_polygon(self, event):
        if len(self._new_vertices) < 3: return

        # del self.helper_object_views[:]
        self.canvas.delete(self.obj_creation_helper_tag)

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
        self.object_views.append(NavMeshView(new_poly, tags=self.root_tag))

        self.active_tool = self.select_tool

        for d_id in self._dots_ids:
            self.canvas.delete(d_id)

        self.draw_all()




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