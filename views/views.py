from mesh2d import Vector2, Matrix
import random

class ObjectView(object):
    '''
    Object View aka 'Draw Object' handles drawing of some data on the canvas.
    This class is intended to be subclassed to draw concrete kinds of data such
    as points, lines, polygons, navmeshes, various helpers etc.
    '''
    def __init__(self, tags=None):
        self.crd_buf = []
        self.element_ids = []

        if tags is not None:
            self.tags = tags
        else:
            self.tags = []



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
        if len(self.crd_buf) == 0:
            self.first_time_draw(canvas)

            # remember coordinate buffer
            for eid in self.element_ids:
                self.crd_buf.extend(canvas.coords(eid))

            print("new object with {} coordinates created".format(len(self.crd_buf)))

        self.redraw(camera_transform, canvas)



    def offset_by(self, vec):
        '''
        move all coordinates in crd_buf by vec
        '''
        new_crds = []
        switch = 0
        for crd in self.crd_buf:
            new_crds.append(crd + vec[switch])
            switch = (switch + 1) % 2
        self.crd_buf = new_crds




    def cleanup(self, canvas):
        for eid in self.element_ids:
            canvas.delete(eid)




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
        self.crd_buf[0] = c1.x
        self.crd_buf[1] = c1.y
        self.crd_buf[2] = c2.x
        self.crd_buf[3] = c2.y

        self.crd_buf[4] = c2.x
        self.crd_buf[5] = c2.y
        self.crd_buf[6] = c3.x
        self.crd_buf[7] = c3.y

        self.crd_buf[8] = c3.x
        self.crd_buf[9] = c3.y
        self.crd_buf[10] = c4.x
        self.crd_buf[11] = c4.y

        self.crd_buf[12] = c4.x
        self.crd_buf[13] = c4.y
        self.crd_buf[14] = c1.x
        self.crd_buf[15] = c1.y






class PointHelperView(ObjectView):
    def __init__(self, loc, color='#ffffff', tags=None):
        self.loc = loc
        self.color = color
        super(PointHelperView, self).__init__(tags)


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



    def modify(self, vec1, vec2):
        delta1 = vec1 - self.v1
        delta2 = vec2 - self.v2
        self.v1 = vec1
        self.v2 = vec2

        self.crd_buf[0] += delta1.x
        self.crd_buf[1] += delta1.y

        self.crd_buf[2] += delta2.x
        self.crd_buf[3] += delta2.y




class PlusView(ObjectView):
    def __init__(self, size, tags=None, loc=None, color='#2f2f2f'):
        self.size = size
        self.color = color
        self.loc = loc if loc is not None else Vector2(0, 0)

        self.down = Vector2(0, self.size)
        self.right = Vector2(self.size, 0)

        super(PlusView, self).__init__(tags)



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
        id1 = canvas.create_line(crds, tags=self.tags, fill=self.color, width=1)
        crds = self.get_open_crds(vrt, (2, 3))
        id2 = canvas.create_line(crds, tags=self.tags, fill=self.color, width=1)
        self.element_ids.append(id1)
        self.element_ids.append(id2)




class PolygonView(ObjectView):
    def __init__(self, poly):
        self.poly = poly
        self.color = '#4A4A4A'
        super(PolygonView, self).__init__()


    def first_time_draw(self, canvas):
        verts = self.poly.vertices
        outline_crds = self.get_closed_crds(verts, self.poly.outline)
        Id = canvas.create_line(outline_crds, fill=self.color, width=1)
        self.element_ids.append(Id)



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
