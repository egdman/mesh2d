import Tkinter as tk
from tkFont import Font as font
from itertools import chain

def debug_draw_room(poly, loops, cv):
    outl = loops[0]
    holes = loops[1:]

    def get_crds(loop):

        crds = []
        for idx in loop:
            vrt = poly.vertices[idx]
            crds.append(vrt.x)
            crds.append(vrt.y)
        crds.append(poly.vertices[loop[0]].x)
        crds.append(poly.vertices[loop[0]].y)
        return crds

    cv.create_line(get_crds(outl), fill='cyan')
    for h in holes:
        cv.create_line(get_crds(h), fill='#ff6666')

    switch = 0
    anchors = (tk.SW, tk.NW)
    for idx in chain(*loops):
        vrt = poly.vertices[idx]
        cv.create_text(vrt.x, vrt.y, fill='white',
            # text = "{}: {}".format(idx, vrt), anchor=anchors[switch],
            # anchor=anchors[switch],
            text = "{}".format(idx),
            anchor=tk.NW,
            font=font(size=8))
        switch = (switch+1)%2