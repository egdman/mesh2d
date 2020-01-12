from mesh2d import vec
from mesh2d.mesh2d import signed_area, Ray
import struct
import random

def test_clip(segment, sector):
    c0, c1 = segment_sector_clip(segment, sector)
    seg0, seg1 = segment
    # print("seg0 = {}, seg1 = {}".format(seg0, seg1))
    # print("  c0 = {},   c1 = {}".format(c0, c1))
    # print(" eq0 = {}, eq1 = {}".format(seg0==c0, seg1==c1))
    e0 = c0 is not None and seg0==c0
    e1 = c1 is not None and seg1==c1
    return e0 and e1
    # if not e0 or not e1:
    #     print("seg0 = {}, c0 = {}, seg1 = {}, c1 = {}".format(seg0, c0, seg1, c1))


def get_float():
    # n = random.getrandbits(64)
    # b = struct.pack('<Q', n)
    # f, = struct.unpack('<d', b)
    # return f
    return random.uniform(0, 1e24)


def ieee754_ser(floatNumber):
    i, = struct.unpack('<Q', struct.pack('<d', floatNumber))
    return format(i, '#x')


def ieee754_unser(hexString):
    return struct.unpack('<d', struct.pack('<Q', int(hexString, base=16)))[0]


DISP = vec(10., 10.)

def get_triangle():
    while True:
        A = vec(get_float(), get_float())
        B = vec(get_float(), get_float())
        # if A+DISP == B+DISP: continue
        C = vec(get_float(), get_float())
        # if A+DISP == C+DISP: continue
        # if B+DISP == C+DISP: continue

        # if vec.cross2(B-A, C-A) <= 0:
        #     return A, C, B
        return A, B, C



def report(A, B, C):
    sector = (B-A, A, C-A)
    segment = (B, C)
    c0, c1 = segment_sector_clip(segment, sector)
    seg0, seg1 = segment

    print("{}, {}".format(seg0, seg1))
    print("{}, {}".format(c0, c1))

    print(80*'-')

    # sector = (C-B, B, A-B)
    # segment = (C, A)
    # print(segment_sector_clip(segment, sector))
    # print(80*'-')

    # sector = (A-C, C, B-C)
    # segment = (A, B)
    # print(segment_sector_clip(segment, sector))
    # print(80*'-')


N = 1000
# for A, B, C in (get_triangle() for _ in xrange(N)):
#     sector1 = (B-A, A, C-A)
#     sector2 = (C-B, B, A-B)
#     sector3 = (A-C, C, B-C)

#     # sector1 = (B-A).normalized(), A, (C-A).normalized()
#     # sector2 = (C-B).normalized(), B, (A-B).normalized()
#     # sector3 = (A-C).normalized(), C, (B-C).normalized()

#     segment1 = (B, C)
#     segment2 = (C, A)
#     segment3 = (A, B)

#     ok = True
#     ok = ok and test_clip(segment1, sector1)
#     ok = ok and test_clip(segment2, sector2)
#     ok = ok and test_clip(segment3, sector3)
#     if not ok:
#         print("A=vec{} B=vec{} C=vec{}".format(A, B, C))


def printf(s, *args):
    args = ("{} : {}".format(arg, ieee754_ser(arg)) for arg in args)
    print(s.format(*args))


delta = 1e-12
# for _ in xrange(N):
#     a = get_float()
#     mul = get_float()

#     b = a + delta
#     if (b==a): continue

#     c1 = b > a
#     c2 = mul*b > mul*a

#     if c1 != c2:
#         printf("a = {}, b = {}, mul = {}", a, b, mul)


# a = ieee754_unser("0x40ce89091c3d3b0f")
# b = ieee754_unser("0x40ce89091c3d3b10")
# mul = ieee754_unser("0x40f34c7ab3d5474a")
# mul = 50.

# print("a == b         : {}".format(a == b))
# # print("mul*a  < mul*b : {}".format(mul*a  < mul*b))
# print("mul*a == mul*b : {}".format(mul*a == mul*b))
# a = 15634.0711743 : 0x40ce89091c3d3b0f
# b = 15634.0711743 : 0x40ce89091c3d3b10
# mul = 79047.6689046 : 0x40f34c7ab3d5474a


# A=vec(4.567962888318229e+168, 2.2988235281307506e+194)
# B=vec(3.0407281385945977e-81, -3.09301501062472e-271)
# C=vec(-1.5314714086414156e-117, 8.185455450410593e+91)

# A=vec(182475.43445202196, 438157.8702334319)
# B=vec(621490.000773662, 240402.29350511613)
# C=vec(474136.0717473703, 570765.4066757238)

# A=vec(4.016922426297742e+152, 8.934543999863397e+153)
# B=vec(4.963707885048733e+153, 2.3608676318494426e+153)
# C=vec(9.559377554916933e+153, 2.5190360074401564e+153)
# report(A, B, C)


def fail_fmt(*vecs):
    return "\n" + "\n".join(("({}, {})".format(ieee754_ser(v[0]), ieee754_ser(v[1])) for v in vecs))


def test_signed_area(a, b, c):
    a1 = signed_area(c, a, b)
    b1 = signed_area(a, b, c)
    c1 = signed_area(b, c, a)
    assert b1 == c1, fail_fmt(a, b, c)
    assert c1 == a1, fail_fmt(a, b, c)
    assert a1 == b1, fail_fmt(a, b, c)

    x1 = Ray(c, a).calc_area(b)
    y1 = Ray(a, b).calc_area(c)
    z1 = Ray(b, c).calc_area(a)
    assert x1 == a1, fail_fmt(a, b, c)
    assert y1 == b1, fail_fmt(a, b, c)
    assert z1 == c1, fail_fmt(a, b, c)

    a2 = signed_area(c, b, a)
    b2 = signed_area(a, c, b)
    c2 = signed_area(b, a, c)
    assert a2 == -a1, fail_fmt(a, b, c)
    assert b2 == -a1, fail_fmt(a, b, c)
    assert c2 == -a1, fail_fmt(a, b, c)

    x2 = Ray(c, b).calc_area(a)
    y2 = Ray(a, c).calc_area(b)
    z2 = Ray(b, a).calc_area(c)
    assert x2 == -a1, fail_fmt(a, b, c)
    assert y2 == -a1, fail_fmt(a, b, c)
    assert z2 == -a1, fail_fmt(a, b, c)

# random.seed(7378547834)
N = 50000
for A, B, C in (get_triangle() for _ in xrange(N)):
    test_signed_area(A, B, C)

print("DONE")

# counter example:
# A = vec(ieee754_unser("0x40f603ab6fc22ca1"), ieee754_unser("0x40f01866e674de69"))
# B = vec(ieee754_unser("0x40f3bd824b69d529"), ieee754_unser("0x40f3befd3db4dbf0"))
# C = vec(ieee754_unser("0x40b2a7d5db5308f1"), ieee754_unser("0x40c5b3c2d45dea18"))

# print("A={}\nB={}\nC={}".format(A, B, C))


# area_A = ((-1962434882.339233398437500000000000) - (-687080107.871335506439208984375000))# + (-512166083.074663877487182617187500)
# area_B = ((-512166083.074663877487182617187500) - (1962434882.339233398437500000000000)) # + (687080107.871335506439208984375000)
# area_C = ((687080107.871335506439208984375000) - (512166083.074663877487182617187500))   # + (-1962434882.339233398437500000000000)

# print("area_A={:.24f}".format(area_A))
# print("area_B={:.24f}".format(area_B))
# print("area_C={:.24f}".format(area_C))
