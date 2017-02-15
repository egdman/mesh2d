class Bbox:
    def __init__(self, vertices, indices):
        vrt = vertices
        ind = indices
        self.xmin = self.xmax = vrt[ind[0]].x
        self.ymin = self.ymax = vrt[ind[0]].y

        for i in ind:
            v = vrt[i]
            if v.x < self.xmin:
                self.xmin = v.x
            elif v.x > self.xmax:
                self.xmax = v.x

            if v.y < self.ymin:
                self.ymin = v.y
            elif v.y > self.ymax:
                self.ymax = v.y


        def point_inside(self, point):
            return point.x < self.xmax and point.x > self.xmin and \
                point.y < self.ymax and point.y > self.ymin
