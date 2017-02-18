from mesh2d import Vector2, Matrix
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
        self.world_vertices = []
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
        if len(self.world_vertices) == 0:
            self.first_time_draw(canvas)

            crd_buf = []
            # add this tag to all canvas objects of this draw object
            for eid in self.element_ids:
                canvas.addtag_withtag(self.tag, eid)

                # remember all coordinates of all canvas objects of this draw object
                curr_obj_crds = canvas.coords(eid)
                self.coord_fences.append(len(crd_buf) + len(curr_obj_crds))
                crd_buf.extend(curr_obj_crds)

            # convert crd_buf to list of Vector2's
            for loc in range(len(crd_buf) / 2):
                # store coords as flat array including the 3rd coord which is 1.0
                self.world_vertices.extend([
                    crd_buf[2*loc],
                    crd_buf[2*loc+1],
                    1.0
                ])

            # print("new object with {} vertices created".format(len(self.world_vertices)))
            # print("fences: {}".format(self.coord_fences))

        self.redraw(camera_transform, canvas)



    def set_vector(self, index, vector):
        index *= 3
        self.world_vertices[index] = vector[0]
        self.world_vertices[index + 1] = vector[1]
        self.world_vertices[index + 2] = vector[2]



    def offset_by(self, vec):
        '''
        move all coordinates in world_vertices by vec
        '''
        for vnum in range(len(self.world_vertices) / 3):
            vhead = vnum*3
            old_vec = Vector2(self.world_vertices[vhead], self.world_vertices[vhead+1])
            self.set_vector(vnum, old_vec + vec)



    def cleanup(self, canvas):
        canvas.delete(self.tag)



    def redraw(self, camera_transform, canvas):
        obj_view = camera_transform

        # use numpy:
        screen_vertices = self.apply_transform_np(obj_view, self.world_vertices)

        # # do not use numpy:
        # screen_vertices = self.apply_transform(obj_view, self.world_vertices)

        screen_crd_buf = []
        for screen_v in screen_vertices:
            screen_crd_buf.append(screen_v[0])
            screen_crd_buf.append(screen_v[1])

        notch = 0

        for (loc, eid) in enumerate(self.element_ids):
            fence = self.coord_fences[loc]
            new_crds = screen_crd_buf[notch:fence]
            canvas.coords(eid, *new_crds)
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


    @staticmethod
    def apply_transform(transform, vertices):
        return list(transform.multiply(Vector2(vert[0], vert[1])).values[:-1] \
            for vert in vertices)


    @staticmethod
    def apply_transform_np(transform, vertices): 
        # np_verts = np.array(vertices).T
        np_verts = np.ndarray(
            shape = (len(vertices) / 3, 3),
            buffer=np.array(vertices)
        ).T

        transform = np.ndarray(
            shape = transform.shape,
            buffer = np.array(transform.values)
        )[:-1] # chop off the last row of the matrix (we don't need it)

        # return np.array(transform).dot(np_verts).T[:, :-1]
        return np.array(transform).dot(np_verts).T



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
        perp = Vector2(axis.y, -axis.x)
        perp /= perp.length()
        c1 = self.v1 + wd*perp
        c2 = self.v1 - wd*perp

        c3 = self.v2 - wd*perp
        c4 = self.v2 + wd*perp
        return c1, c2, c3, c4



    def first_time_draw(self, canvas):
        (c1, c2, c3, c4) = self.find_corners()

        id1 = canvas.create_line((c1.x, c1.y, c2.x, c2.y),
            fill = self.color, width=1)

        id2 = canvas.create_line((c2.x, c2.y, c3.x, c3.y),
            fill = self.color, width=1)

        id3 = canvas.create_line((c3.x, c3.y, c4.x, c4.y),
            fill = self.color, width=1)

        id4 = canvas.create_line((c4.x, c4.y, c1.x, c1.y),
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




class PointHelperView(ObjectView):
    def __init__(self, loc, color='#ffffff'):
        self.loc = loc
        self.color = color
        super(PointHelperView, self).__init__()


    def modify(self, vec):
        delta = vec - self.loc
        self.loc = vec
        self.offset_by(delta)


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

            fill = self.color
        )
        self.element_ids.append(Id)



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
        delta1 = vec1 - self.v1
        delta2 = vec2 - self.v2
        self.v1 = vec1
        self.v2 = vec2

        # self.world_vertices[0] = delta1.column(0)
        # self.world_vertices[1] = delta2.column(0)

        self.set_vector(0, delta1)
        self.set_vector(1, delta2)




class PlusView(ObjectView):
    def __init__(self, size, loc=None, color='#2f2f2f'):
        self.size = size
        self.color = color
        self.loc = loc if loc is not None else Vector2(0, 0)

        self.down = Vector2(0, self.size)
        self.right = Vector2(self.size, 0)

        super(PlusView, self).__init__()



    def modify(self, vec):
        delta = vec - self.loc
        self.loc = vec
        self.offset_by(delta)



    def first_time_draw(self, canvas):
        sz = self.size

        vrt = [self.loc - self.down, self.loc + self.down, self.loc - self.right, self.loc + self.right]
        # vrt = [Vector2(0, -sz),Vector2(0, sz),Vector2(-sz, 0),Vector2(sz, 0),]
        # trans_vrt = list(self.apply_transform(camera_transform, vrt))
        crds = self.get_open_crds(vrt, (0, 1))
        id1 = canvas.create_line(crds, fill=self.color, width=1)
        crds = self.get_open_crds(vrt, (2, 3))
        id2 = canvas.create_line(crds, fill=self.color, width=1)
        self.element_ids.append(id1)
        self.element_ids.append(id2)




class PolygonView(ObjectView):
    def __init__(self, poly):
        self.poly = poly
        self.color = '#4A4A4A'
        super(PolygonView, self).__init__()


    def first_time_draw(self, canvas):
        verts = self.poly.vertices

        poly_crds = self.get_open_crds(verts, self.poly.outline)
        Id = canvas.create_polygon(poly_crds, fill=self.color)
        self.element_ids.append(Id)

        outline_crds = self.get_closed_crds(verts, self.poly.outline)
        Id = canvas.create_line(outline_crds, fill='#ffffff', width=1)
        self.element_ids.append(Id)



class NavMeshView(ObjectView):

    def __init__(self, navmesh):
        rnd = lambda: random.randint(0,255)
        self.color = '#%02X%02X%02X' % (rnd(),rnd(),rnd())
        self.navmesh = navmesh
        self.obj_trans = Matrix.identity(3)

        super(NavMeshView, self).__init__()



    def first_time_draw(self, canvas):

        trans_vertices = self.navmesh.vertices

        # draw rooms
        for room in self.navmesh.rooms:
            room_crds = self.get_open_crds(trans_vertices, room)
            Id = canvas.create_polygon(room_crds, fill='#1A1A1A', activefill='#111111')
            self.element_ids.append(Id)

        # draw portals
        for portal in self.navmesh.portals:
            portal_crds = self.get_open_crds(trans_vertices, portal)
            Id = canvas.create_line(portal_crds, fill='red', width=1)
            self.element_ids.append(Id)

        # draw outline
        outline_crds = self.get_closed_crds(trans_vertices, self.navmesh.indices)
        Id = canvas.create_line(outline_crds, fill='#FFFFFF', width=1)
        self.element_ids.append(Id)
