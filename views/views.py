from mesh2d import vec, Matrix
try:
    import Tkinter as tk
    from tkFont import Font
    from itertools import izip as zip
except ImportError:
    import tkinter as tk
    from tkinter.font import Font
import random
import numpy as np
import uuid

class ObjectView(object):
    '''
    Object View aka 'Draw Object' handles drawing of some data on the canvas.
    This class is intended to be subclassed to draw concrete kinds of data such
    as points, lines, polygons, navmeshes, various helpers etc.
    '''
    def __init__(self):
        self.world_vertices_np = None # numpy array of vertex world coords
        self.element_ids = []
        self.coord_fences = []
        # generate unique tag for all canvas objects of this draw object
        self.tag = uuid.uuid4().hex


    def first_time_draw(self, canvas):
        '''
        First time draw (add elements to canvas and store their canvas ids in 'element_ids'.)
        '''
        raise NotImplementedError("Implement method 'first_time_draw' in your subclass")


    def modify(self, *args, **kwargs):
        '''
        Modify draw object. Concrete functionality depends on the concrete class.
        '''
        raise NotImplementedError("Implement method 'modify' in your subclass")


    
    def draw_self(self, camera_transform, canvas):
        if self.world_vertices_np is None:
            self.first_time_draw(canvas)

            np_buffer = []
            verts_added = 0
            # add this tag to all canvas objects of this draw object
            for eid in self.element_ids:
                canvas.addtag_withtag(self.tag, eid)

                # remember all coordinates of all canvas objects of this draw object
                curr_obj_crds = canvas.coords(eid)
                for idx in range(len(curr_obj_crds) // 2):
                    verts_added += 1
                    np_buffer.extend((
                        curr_obj_crds[2*idx],
                        curr_obj_crds[2*idx + 1],
                        1.0))

                self.coord_fences.append(verts_added)

            self.world_vertices_np = np.ndarray(
                shape = (len(np_buffer) // 3, 3),
                buffer=np.array(np_buffer))

        self.redraw(camera_transform, canvas)



    def set_vector(self, index, vector):
        self.world_vertices_np[index][0] = vector[0]
        self.world_vertices_np[index][1] = vector[1]
        self.world_vertices_np[index][2] = 1



    # def offset_by(self, offset):
    #     '''
    #     move all coordinates in world_vertices by vec
    #     '''
    #     for vnum in range(self.world_vertices_np.shape[0]):
    #         old_vec = vec(self.world_vertices_np[vnum][0], self.world_vertices_np[vnum][1])
    #         self.set_vector(vnum, old_vec + offset)


    def cleanup(self, canvas):
        canvas.delete(self.tag)



    def redraw(self, camera_transform, canvas):
        screen_coords = self.world_vertices_np.dot(camera_transform[:-1].T)
        notch = 0
        for (elem_id, fence) in zip(self.element_ids, self.coord_fences):
            canvas.coords(elem_id, *(screen_coords[notch:fence].flatten()))
            notch = fence


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


class TextView(ObjectView):
    # sizes = set(range(6, 78, 6))

    def __init__(self, loc, text, size=18, scale=False, color='#ffffff'):
        super(TextView, self).__init__()
        self.loc = loc
        self.color = color
        self.text = text
        self.size = self.current_size = size
        self.scale = scale

    def draw_self(self, camera_transform, canvas):
        if self.world_vertices_np is None:
            textId = canvas.create_text(0, 0, fill=self.color, text=self.text, font=Font(size=self.size), anchor=tk.NW)
            canvas.addtag_withtag(self.tag, textId)
            self.element_ids = (textId,)

            x, y = self.loc
            self.world_vertices_np = np.ndarray(
                shape = (3, 3),
                buffer=np.array((
                    x, y, 1.0,
                    0, 0, 1.0,
                    1, 0, 1.0)))

        self.redraw(camera_transform, canvas)

    def redraw(self, camera_transform, canvas):
        screen_coords = self.world_vertices_np.dot(camera_transform[:-1].T)

        if self.scale:
            scale = np.linalg.norm(screen_coords[1] - screen_coords[2])
            new_size = int(self.size * scale)
            if new_size != self.current_size and new_size % 5 == 0 and new_size <= 70:
                self.current_size = new_size
                self.cleanup(canvas)

                if new_size >= 10:
                    textId = canvas.create_text(0, 0, fill=self.color, text=self.text, font=Font(size=new_size), anchor=tk.NW)
                    canvas.addtag_withtag(self.tag, textId)
                    self.element_ids = (textId,)
                else:
                    new_size = 10


        elem_id, = self.element_ids
        x, y = screen_coords[0]
        canvas.coords(elem_id, x, y)


class PointHelperView(ObjectView):
    def __init__(self, loc, color='#ffffff'):
        self.loc = loc
        self.sz = 3
        self.color = color
        super(PointHelperView, self).__init__()


    def draw_self(self, camera_transform, canvas):
        if self.world_vertices_np is None:
            Id = canvas.create_polygon((0, 0, 0, 0, 0, 0, 0, 0), fill=self.color)
            canvas.addtag_withtag(self.tag, Id)
            self.element_ids.append(Id)

            x, y = self.loc
            self.world_vertices_np = np.ndarray(
                shape = (1, 3),
                buffer=np.array((x, y, 1.0)))

        self.redraw(camera_transform, canvas)

    def redraw(self, camera_transform, canvas):
        screen_coords = self.world_vertices_np[:1].dot(camera_transform[:-1].T)
        x, y = screen_coords[0]
        sz = self.sz
        for elem_id in self.element_ids:
            canvas.coords(elem_id,
                x - sz,
                y - sz,
                x + sz,
                y - sz,
                x + sz,
                y + sz,
                x - sz,
                y + sz,
            )

class WallHelperView(ObjectView):
    def __init__(self, vec1, vec2, width, color='#ffffff'):
        self.v1 = vec1
        self.v2 = vec2
        self.width = width
        self.color = color
        super(WallHelperView, self).__init__()


    def find_corners(self):
        wd = self.width
        axis = self.v2 - self.v1
        perp = vec(axis[1], -axis[0]).normalized()
        c1 = self.v1 + wd*perp
        c2 = self.v1 - wd*perp

        c3 = self.v2 - wd*perp
        c4 = self.v2 + wd*perp
        return c1, c2, c3, c4



    def first_time_draw(self, canvas):
        (c1, c2, c3, c4) = self.find_corners()

        id1 = canvas.create_line((c1[0], c1[1], c2[0], c2[1]),
            fill = self.color, width=1)

        id2 = canvas.create_line((c2[0], c2[1], c3[0], c3[1]),
            fill = self.color, width=1)

        id3 = canvas.create_line((c3[0], c3[1], c4[0], c4[1]),
            fill = self.color, width=1)

        id4 = canvas.create_line((c4[0], c4[1], c1[0], c1[1]),
            fill = self.color, width=1)

        self.element_ids.append(id1)
        self.element_ids.append(id2)
        self.element_ids.append(id3)
        self.element_ids.append(id4)



    def modify(self, vec1, vec2, width):
        self.v1 = vec1
        self.v2 = vec2
        self.width = width
        c1, c2, c3, c4 = self.find_corners()

        self.set_vector(0, c1)
        self.set_vector(1, c2)
        self.set_vector(2, c2)
        self.set_vector(3, c3)
        self.set_vector(4, c3)
        self.set_vector(5, c4)
        self.set_vector(6, c4)
        self.set_vector(7, c1)


class SegmentHelperView(ObjectView):
    def __init__(self, vert1, vert2, color='#ffffff', width=1):
        self.v1 = vert1
        self.v2 = vert2
        self.width = width

        self.color = color
        super(SegmentHelperView, self).__init__()


    def first_time_draw(self, canvas):
        Id = canvas.create_line(
                (
                    self.v1[0], self.v1[1],
                    self.v2[0], self.v2[1]
                ),
                fill = self.color,
                width=self.width
        )
        self.element_ids.append(Id)


    def modify(self, vec1, vec2):
        self.set_vector(0, vec1)
        self.set_vector(1, vec2)


class PlusView(ObjectView):
    def __init__(self, loc, size, color='#2f2f2f'):
        super(PlusView, self).__init__()
        self.size = size
        self.color = color
        x, y = loc
        self.world_vertices_np = np.ndarray(shape = (1, 3), buffer=np.array((x, y, 1.0)))


    def modify(self, new_loc):
        x, y = new_loc
        self.world_vertices_np[0] = (x, y, 1.)


    def draw_self(self, camera_transform, canvas):
        if len(self.element_ids) == 0:
            crds = (0.,)*4
            id1 = canvas.create_line(crds, fill=self.color, width=1)
            id2 = canvas.create_line(crds, fill=self.color, width=1)

            canvas.addtag_withtag(self.tag, id1)
            canvas.addtag_withtag(self.tag, id2)
            self.element_ids.append(id1)
            self.element_ids.append(id2)

        self.redraw(camera_transform, canvas)


    def redraw(self, camera_transform, canvas):
        screen_coords = self.world_vertices_np[:1].dot(camera_transform[:-1].T)
        x, y = screen_coords[0]
        sz = self.size

        el1, el2 = self.element_ids
        canvas.coords(el1,
            x, y-sz,
            x, y+sz,
        )
        canvas.coords(el2,
            x-sz, y,
            x+sz, y,
        )


class PolygonView(ObjectView):
    def __init__(self, poly, outline_color='#a0a0a0', hole_color='#a0a0ff'):
        self.outline = list(poly.graph.loop_iterator(poly.graph.loops[0]))
        self.holes = list(list(poly.graph.loop_iterator(hole)) for hole in poly.graph.loops[1:])
        self.vertices = poly.vertices
        self.outline_color = outline_color
        self.hole_color = hole_color
        super(PolygonView, self).__init__()


    def first_time_draw(self, canvas):
        outline_crds = self.get_closed_crds(self.vertices, self.outline)
        Id = canvas.create_line(outline_crds, fill=self.outline_color, width=1)
        self.element_ids.append(Id)

        for hole in self.holes:
            outline_crds = self.get_closed_crds(self.vertices, hole)
            Id = canvas.create_line(outline_crds, fill=self.hole_color, width=1)
            self.element_ids.append(Id)            




class NavMeshView(ObjectView):

    def __init__(self, navmesh):
        rnd = lambda: random.randint(0,255)
        self.color = '#%02X%02X%02X' % (rnd(),rnd(),rnd())
        self.navmesh = navmesh

        super(NavMeshView, self).__init__()



    def first_time_draw(self, canvas):

        trans_vertices = self.navmesh.vertices

        # draw rooms
        for room in self.navmesh.rooms:
            # print ("room = {}".format(room))
            room_crds = self.get_open_crds(trans_vertices, room)
            Id = canvas.create_polygon(room_crds, fill='#1E1E1E', activefill='#111111')
            self.element_ids.append(Id)

        # draw portals
        for portal in self.navmesh.portals:
            portal_crds = self.get_open_crds(trans_vertices, portal)
            Id = canvas.create_line(portal_crds, fill='#833330', width=1)
            self.element_ids.append(Id)

        # draw outline
        outline_crds = self.get_closed_crds(trans_vertices, self.navmesh.outline)
        Id = canvas.create_line(outline_crds, fill='#727272', width=1)
        self.element_ids.append(Id)

        # draw holes
        for hole in self.navmesh.holes:
            outline_crds = self.get_closed_crds(trans_vertices, hole)
            Id = canvas.create_line(outline_crds, fill='#727272', width=1)
            self.element_ids.append(Id)
