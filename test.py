def cut_to_pieces(array, cut_items):
    pieces = [[]]
    for el in array:
        pieces[-1].append(el)
        if el in cut_items:
            pieces.append([el])

    pieces[0] = pieces[-1] + pieces[0]
    del pieces[-1]
    return pieces



from collections import deque

ls = deque([0,1,2,3,4,7,8,9,10,11,12,5,6])
fd = [7,8,9,10,11,12]
fd = []

for _ in range(len(ls)):
    # print(ls)
    print(sorted(cut_to_pieces(ls, fd)))

    ls.rotate(1)

# def point_inside(idx):
#     return idx in [2]


# my_outline = [10,11,12,4,5,6,0,1,7,2,8,3,9]
# inters_ids = [7,8,9,10,11,12]

# my_outline_pieces = cut_to_pieces(my_outline, inters_ids)

# # find first piece that contains a non-intersection vertex
# first_piece, start_num = next(((piece, num) for num, piece \
#     in enumerate(my_outline_pieces) if len(piece) > 2))

# # put this piece at the start of list of pieces
# my_outline_pieces = deque(my_outline_pieces)
# my_outline_pieces.rotate(-start_num)

# inside = point_inside(first_piece[1])
# if inside: my_outline_pieces.rotate(-1)

# my_outline_pieces_outside = list(my_outline_pieces)[::2]

# print my_outline_pieces_outside





