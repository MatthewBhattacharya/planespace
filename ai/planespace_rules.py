"""Faithful Python port of the Planespace v2.1 rules in ../index.html.

Kept as a 1:1 port (same function names/shapes as the JS) so it's easy to
verify against the browser version and easy to re-sync if the rules change.
"""

COLS = ROWS = 10
MAX_VERTS = 8
MIN_AREA = 1
MAX_AREA = 9
GEM_PTS = [(1, 1), (8, 1), (1, 8), (8, 8), (4, 4), (6, 3), (3, 6)]
GEM_SET = set(GEM_PTS)

BASE_TABLE = [4, 6, 6, 8, 8, 10, 10, 12, 12, 12, 14, 14, 14, 14, 16, 16, 16, 16]  # index = round(area*2)-1
# Flat bonus per shape type — no size bracket.
BONUS_TABLE = {
    'triangle': 12,    'isosceles': 18,      'right': 22,
    '45-45-90': 26,    'equilateral': 42,
    'quad': 8,         'trapezoid': 14,       'parallelogram': 22,
    'rect-rhomb': 28,  'square': 22,
    'pentagon': 12,    'convex-pent': 26,
    'hexagon': 18,     'convex-hex': 36,      'ngon': 10,
}
GEM_BONUS = 12


def vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1]


def crs(u, v):
    return u[0] * v[1] - u[1] * v[0]


def dsq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def neg(v):
    return (-v[0], -v[1])


def sign(x):
    return (x > 0) - (x < 0)


def shoelace(pts):
    n = len(pts)
    s = 0
    for i in range(n):
        j = (i + 1) % n
        s += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(s) / 2


def proper_x(a, b, c, d):
    d1 = crs(vec(c, d), vec(c, a))
    d2 = crs(vec(c, d), vec(c, b))
    d3 = crs(vec(a, b), vec(a, c))
    d4 = crs(vec(a, b), vec(a, d))
    return (sign(d1) * sign(d2) < 0) and (sign(d3) * sign(d4) < 0)


def is_simple(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if proper_x(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                return False
    return True


def pip(p, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > p[1]) != (yj > p[1]) and p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def centroid(pts):
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def overlaps(a, b):
    na, nb = len(a), len(b)
    for i in range(na):
        for j in range(nb):
            if proper_x(a[i], a[(i + 1) % na], b[j], b[(j + 1) % nb]):
                return True
    if pip(centroid(a), b):
        return True
    if pip(centroid(b), a):
        return True
    return False


def is_convex(pts):
    n = len(pts)
    s = 0
    for i in range(n):
        c = crs(vec(pts[i], pts[(i + 1) % n]), vec(pts[(i + 1) % n], pts[(i + 2) % n]))
        if c != 0:
            sg = 1 if c > 0 else -1
            if s == 0:
                s = sg
            elif sg != s:
                return False
    return True


def collinear3(a, b, c):
    return crs(vec(a, b), vec(a, c)) == 0


def edges_overlap(a, b, c, d):
    """True only if segments AB and CD overlap along more than a single point."""
    if not collinear3(a, b, c) or not collinear3(a, b, d):
        return False
    abx, aby = b[0] - a[0], b[1] - a[1]
    len2 = abx * abx + aby * aby
    if len2 == 0:
        return False

    def t(p):
        return ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / len2

    t2, t3 = t(c), t(d)
    lo2, hi2 = min(t2, t3), max(t2, t3)
    return min(1, hi2) - max(0, lo2) > 1e-9


def shares_opponent_edge(new_pts, shapes, player):
    n = len(new_pts)
    for sh in shapes:
        if sh[1] == player:
            continue
        verts = sh[0]
        m = len(verts)
        for i in range(n):
            for j in range(m):
                if edges_overlap(new_pts[i], new_pts[(i + 1) % n], verts[j], verts[(j + 1) % m]):
                    return True
    return False


def classify(pts):
    n = len(pts)
    conv = is_convex(pts)
    el2 = [dsq(pts[i], pts[(i + 1) % n]) for i in range(n)]

    def all_eq(a):
        return all(v == a[0] for v in a)

    def right_at(i):
        inc = vec(pts[(i - 1) % n], pts[i])
        out = vec(pts[i], pts[(i + 1) % n])
        return dot(neg(inc), out) == 0

    rights = [right_at(i) for i in range(n)]
    tier, names = '', []

    if n == 3:
        s2 = sorted(el2)
        is_right = s2[0] + s2[1] == s2[2]
        is_iso = el2[0] == el2[1] or el2[1] == el2[2] or el2[0] == el2[2]
        is_equil = all_eq(el2)
        names = ['Triangle']
        if is_equil:
            names += ['Equilateral']
            tier = 'equilateral'
        elif is_right and is_iso:
            names += ['Right', 'Isosceles', '45-45-90']
            tier = '45-45-90'
        elif is_right:
            names += ['Right']
            tier = 'right'
        elif is_iso:
            names += ['Isosceles']
            tier = 'isosceles'
        else:
            tier = 'triangle'
    elif n == 4:
        e = [vec(pts[i], pts[(i + 1) % n]) for i in range(n)]
        par_ac = crs(e[0], e[2]) == 0
        par_bd = crs(e[1], e[3]) == 0
        is_parall = par_ac and par_bd
        is_trap = par_ac != par_bd
        is_rhom = is_parall and all_eq(el2)
        all_right = all(rights)
        is_rect = is_parall and all_right
        is_sq = is_rhom and all_right
        names = ['Quadrilateral']
        if is_sq:
            names += ['Square']; tier = 'square'
        elif is_rect:
            names += ['Rectangle']; tier = 'rect-rhomb'
        elif is_rhom:
            names += ['Rhombus']; tier = 'rect-rhomb'
        elif is_parall:
            names += ['Parallelogram']; tier = 'parallelogram'
        elif is_trap:
            names += ['Trapezoid']; tier = 'trapezoid'
        else:
            tier = 'quad'
    elif n == 5:
        names = ['Convex Pentagon' if conv else 'Pentagon']
        tier = 'convex-pent' if conv else 'pentagon'
    elif n == 6:
        names = ['Convex Hexagon' if conv else 'Hexagon']
        tier = 'convex-hex' if conv else 'hexagon'
    else:
        names = [f'{n}-gon']
        tier = 'ngon'
        if conv:
            names.append('(Convex)')

    return {'n': n, 'tier': tier, 'names': names, 'conv': conv}


def score_shape(pts, cls, claimed_gems):
    area = shoelace(pts)
    base = BASE_TABLE[round(area * 2) - 1]
    bonus = BONUS_TABLE.get(cls['tier'], 0)
    new_gems = 0
    used_gems = 0
    for v in pts:
        if v in GEM_SET:
            used_gems += 1
            if v not in claimed_gems:
                new_gems += 1
    gem_bonus = new_gems * GEM_BONUS
    return {'area': area, 'base': base, 'bonus': bonus, 'gem_bonus': gem_bonus,
            'new_gems': new_gems, 'total': base + bonus + gem_bonus}


def validate(new_pts, shapes, claimed_gems, player):
    n = len(new_pts)
    if n < 3:
        return False, 'need more points'
    if len(set(new_pts)) != n:
        return False, 'duplicate vertices'
    if not is_simple(new_pts):
        return False, 'not simple'
    for i in range(n):
        if collinear3(new_pts[(i - 1) % n], new_pts[i], new_pts[(i + 1) % n]):
            return False, 'collinear consecutive vertices'
    area = shoelace(new_pts)
    if area < MIN_AREA:
        return False, 'area too small'
    if area > MAX_AREA:
        return False, 'area too big'
    for sh in shapes:
        if overlaps(new_pts, sh[0]):
            return False, 'overlaps existing shape'
    if shares_opponent_edge(new_pts, shapes, player):
        return False, "shares edge with opponent's shape"
    return True, None
