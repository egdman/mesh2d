import Tkinter as tk
from tkFont import Font as font
from itertools import chain, tee

try:
    from itertools import izip as zip
except ImportError:
    pass

def pairs(iterable):
    a, b = tee(iterable, 2)
    next(b, None)
    return zip(a, b)

def triples(iterable):
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c)


def get_crds(poly, loop, closed=True):
    crds = []
    for idx in loop:
        vrt = poly.vertices[idx]
        crds.extend(vrt.comps)
    if closed:
        crds.extend(poly.vertices[loop[0]].comps)
    return crds



def draw_poly(poly, cv, no_draw_these=None):
    if no_draw_these is None: no_draw_these = []
    outline = list(poly.graph.loop_iterator(poly.graph.loops[0]))

    cv.create_line(get_crds(poly, outline), fill='cyan')

    holes = list(list(poly.graph.loop_iterator(h)) for h in poly.graph.loops[1:])
    for hole in holes:
        cv.create_line(get_crds(poly, hole), fill='orange')

    for idx in chain(outline, *holes):
        if idx in no_draw_these: continue

        vrt = poly.vertices[idx]
        cv.create_text(vrt[0], vrt[1], fill='white',
            text=" " + str(idx), anchor=tk.NW)



def debug_draw_bool(A, B, A_pieces, B_pieces, x_map, cv):
    for A_piece in A_pieces:
        A_piece_crds = get_crds(A, A_piece, False)
        if A_piece_crds: cv.create_line(A_piece_crds, fill='red', width=3)

    for B_piece in B_pieces:
        B_piece_crds = get_crds(B, B_piece, False)
        if B_piece_crds: cv.create_line(B_piece_crds, fill='green', width=3)

    for B_int, A_int in x_map.iteritems():
        crds = A.vertices[A_int]

        cv.create_text(crds[0], crds[1], fill='white',
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
        cv.create_text(vrt[0], vrt[1], fill='white',
            # text = "{}: {}".format(idx, vrt), anchor=anchors[switch],
            # anchor=anchors[switch],
            text = "{}".format(idx),
            anchor=tk.NW,
            font=font(size=8))
        switch = (switch+1)%2
