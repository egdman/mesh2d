import Tkinter as tk
from tkFont import Font as font
from itertools import chain, izip



def get_crds(poly, loop, closed=True):
    crds = []
    for idx in loop:
        vrt = poly.vertices[idx]
        crds.append(vrt.x)
        crds.append(vrt.y)
    if closed:
        crds.append(poly.vertices[loop[0]].x)
        crds.append(poly.vertices[loop[0]].y)
    return crds



def draw_poly(poly, cv, no_draw_these=None):
    if no_draw_these is None: no_draw_these = []
    outl = get_crds(poly, poly.outline)
    cv.create_line(outl, fill='cyan')
    for hole in poly.holes:
        cv.create_line(get_crds(poly, hole), fill='#ff6666')

    for idx in chain(poly.outline, *poly.holes):
        if idx in no_draw_these: continue

        vrt = poly.vertices[idx]
        cv.create_text(vrt.x, vrt.y, fill='white',
            text=" " + str(idx), anchor=tk.NW)




def debug_draw_bool(A, B, A_pieces, B_pieces, x_map, cv):

    for A_piece, B_piece in izip(A_pieces, B_pieces):
        A_piece_crds = get_crds(A, A_piece, False)
        B_piece_crds = get_crds(B, B_piece, False)


        if A_piece_crds: cv.create_line(A_piece_crds, fill='red', width=3)
        if B_piece_crds: cv.create_line(B_piece_crds, fill='green', width=3)


    for B_int, A_int in x_map.iteritems():
        crds = A.vertices[A_int]

        cv.create_text(crds.x, crds.y, fill='white',
            text="{} = {}".format(A_int, B_int), anchor=tk.NW)

    if len(x_map) > 0:
        B_inters, A_inters = zip(*x_map.iteritems())
    else:
        B_inters, A_inters = [], []

    draw_poly(A, cv, A_inters)
    draw_poly(B, cv, B_inters)





def debug_draw_room(poly, loops, cv):
    outl = loops[0]
    holes = loops[1:]

   

    cv.create_line(get_crds(poly, outl), fill='cyan')
    for h in holes:
        cv.create_line(get_crds(poly, h), fill='#ff6666')

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