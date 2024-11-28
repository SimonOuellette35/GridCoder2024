# The following primitives are a subset from Michael's Hodel's DSL that consists of grid-to-grid transformations only.
# Michael Hodel's DSL: https://github.com/michaelhodel/arc-dsl

import inspect

ZERO = 0
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6
SEVEN = 7
EIGHT = 8
NINE = 9
TEN = 10
F = False
T = True

NEG_ONE = -1

ORIGIN = (0, 0)
UNITY = (1, 1)
DOWN = (1, 0)
RIGHT = (0, 1)
UP = (-1, 0)
LEFT = (0, -1)

NEG_TWO = -2
NEG_UNITY = (-1, -1)
UP_RIGHT = (-1, 1)
DOWN_LEFT = (1, -1)

ZERO_BY_TWO = (0, 2)
TWO_BY_ZERO = (2, 0)
TWO_BY_TWO = (2, 2)
THREE_BY_THREE = (3, 3)

def execute(func_str, grid):
    return eval(func_str)(grid)

def is_color_related(func_str):
    if func_str.startswith("set_fg_color") or func_str.startswith("color_swap"):
        return True
    else:
        return False

def is_rotation(func_str):
    if func_str.startswith("rot"):
        return True
    else:
        return False

def is_mirroring(func_str):
    if func_str.endswith("mirror"):
        return True
    else:
        return False

def is_rep(func_str):
    if func_str.startswith("rep"):
        return True
    else:
        return False

def get_prim_func_by_name(name):
    return semantics[name]

def get_shortcuts():
    shortcuts = {
        'TopHalf/LeftHalf': 'FirstQuadrant',
        'LeftHalf/TopHalf': 'FirstQuadrant',
        'TopHalf/RightHalf': 'SecondQuadrant',
        'RightHalf/TopHalf': 'SecondQuadrant',
        'BottomHalf/LeftHalf': 'ThirdQuadrant',
        'LeftHalf/BottomHalf': 'ThirdQuadrant',
        'BottomHalf/RightHalf': 'FourthQuadrant',
        'RightHalf/BottomHalf': 'FourthQuadrant',
        'VerticallyMirror/HorizontallyMirror': 'RotateHalf',
        'HorizontallyMirror/VerticallyMirror': 'RotateHalf'
    }

    return shortcuts

def get_index(name):
    return prim_indices[name]

def inverse_lookup(idx):
    for key, val in prim_indices.items():
        if val == idx:
            return key

def clear_single_colors(grid):
    func = fork(paint, identity, chain(lbind(recolor, ZERO), rbind(mfilter, matcher(size, ONE)), partition))
    return func(grid)

def clear_double_colors(grid):
    func = fork(paint, identity, chain(lbind(recolor, ZERO), rbind(mfilter, matcher(size, TWO)), partition))
    return func(grid)

def clear_triple_colors(grid):
    func = fork(paint, identity, chain(lbind(recolor, ZERO), rbind(mfilter, matcher(size, THREE)), partition))
    return func(grid)

def drag_down_underpaint(grid):
    func = fork(underpaint, identity, chain(rbind(shift, DOWN), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_down_paint(grid):
    func = fork(paint, identity, chain(rbind(shift, DOWN), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_left_underpaint(grid):
    func = fork(underpaint, identity, chain(rbind(shift, LEFT), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_left_paint(grid):
    func = fork(paint, identity, chain(rbind(shift, LEFT), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_up_underpaint(grid):
    func = fork(underpaint, identity, chain(rbind(shift, UP), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_up_paint(grid):
    func = fork(paint, identity, chain(rbind(shift, UP), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_right_underpaint(grid):
    func = fork(underpaint, identity, chain(rbind(shift, RIGHT), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_right_paint(grid):
    func = fork(paint, identity, chain(rbind(shift, RIGHT), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_diagonally_underpaint(grid):
    func = fork(underpaint, identity, chain(rbind(shift, UNITY), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_diagonally_paint(grid):
    func = fork(paint, identity, chain(rbind(shift, UNITY), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_counterdiagonally_underpaint(grid):
    func = fork(underpaint, identity, chain(rbind(shift, DOWN_LEFT), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def drag_counterdiagonally_paint(grid):
    func = fork(paint, identity, chain(rbind(shift, DOWN_LEFT), rbind(sfilter, chain(flip, rbind(equality, ZERO), first)), asobject))
    return func(grid)

def extend_by_one(grid):
    func = fork(paint, compose(lbind(canvas, ZERO), chain(increment, increment, shape)), compose(rbind(shift, UNITY), asobject))
    return func(grid)

def extend_by_two(grid):
    func = fork(paint, chain(lbind(canvas, ZERO), power(increment, FOUR), shape), compose(rbind(shift, TWO_BY_TWO), asobject))
    return func(grid)

def insert_top_row(grid):
    func = fork(vconcat, chain(lbind(canvas, ZERO), lbind(astuple, ONE), width), identity)
    return func(grid)

def insert_bottom_row(grid):
    func = fork(vconcat, identity, chain(lbind(canvas, ZERO), lbind(astuple, ONE), width))
    return func(grid)

def insert_left_col(grid):
    func = fork(hconcat, chain(lbind(canvas, ZERO), rbind(astuple, ONE), height), identity)
    return func(grid)

def insert_right_col(grid):
    func = fork(hconcat, identity, chain(lbind(canvas, ZERO), rbind(astuple, ONE), height))
    return func(grid)

def stack_rows_horizontally(grid):
    func = compose(rbind(repeat, ONE), merge)
    return func(grid)

def stack_rows_vertically(grid):
    func = chain(dmirror, compose(rbind(repeat, ONE), merge), dmirror)
    return func(grid)

def stack_rows_horizontally_compress(grid):
    func = chain(rbind(repeat, ONE), lbind(remove, ZERO), chain(first, rbind(repeat, ONE), merge))
    return func(grid)

def stack_columns_vertically_compress(grid):
    func = chain(dmirror, chain(rbind(repeat, ONE), lbind(remove, ZERO), chain(first, rbind(repeat, ONE), merge)), dmirror)
    return func(grid)

def symmetrize_left_around_vertical(grid):
    return hconcat(lefthalf(grid), vmirror(lefthalf(grid)))

def symmetrize_right_around_vertical(grid):
    return hconcat(vmirror(righthalf(grid)), righthalf(grid))

def symmetrize_top_around_horizontal(grid):
    return vconcat(tophalf(grid), hmirror(tophalf(grid)))

def symmetrize_bottom_around_horizontal(grid):
    return vconcat(hmirror(bottomhalf(grid)), bottomhalf(grid))

def symmetrize_quad(grid):
    func = fork(vconcat, fork(hconcat, compose(lefthalf, tophalf), chain(vmirror, lefthalf, tophalf)), fork(hconcat, chain(hmirror, lefthalf, tophalf), chain(compose(hmirror, vmirror), lefthalf, tophalf)))
    return func(grid)

def keep_only_diagonal(grid):
    func = fork(paint, identity, compose(lbind(recolor, ZERO), fork(difference, asobject, fork(toobject, fork(connect, compose(ulcorner, asindices), compose(lrcorner, asindices)), identity))))
    return func(grid)

def keep_only_counterdiagonal(grid):
    func = fork(paint, identity,
                compose(lbind(recolor, ZERO),
                        fork(difference,
                             asobject,
                             fork(toobject,
                                  fork(connect,
                                       compose(urcorner, asindices),
                                       compose(llcorner, asindices)), identity))))
    return func(grid)

def shear_rows_left(grid):
    func = compose(lbind(apply, fork(combine, compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, compose(increment, last)), lbind(rbind, greater), first))), compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, last), lbind(lbind, greater), first))))), fork(pair, compose(rbind(lbind(interval, ZERO), ONE), height), identity))
    return func(grid)

def shear_rows_right(grid):
    func = chain(vmirror, compose(lbind(apply, fork(combine, compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, compose(increment, last)), lbind(rbind, greater), first))), compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, last), lbind(lbind, greater), first))))), fork(pair, compose(rbind(lbind(interval, ZERO), ONE), height), identity)), vmirror)
    return func(grid)

def shear_cols_down(grid):
    func = chain(dmirror, compose(lbind(apply, fork(combine, compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, compose(increment, last)), lbind(rbind, greater), first))), compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, last), lbind(lbind, greater), first))))), fork(pair, compose(rbind(lbind(interval, ZERO), ONE), height), identity)), dmirror)
    return func(grid)

def shear_cols_up(grid):
    func = chain(dmirror, chain(vmirror, compose(lbind(apply, fork(combine, compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, compose(increment, last)), lbind(rbind, greater), first))), compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, last), lbind(lbind, greater), first))))), fork(pair, compose(rbind(lbind(interval, ZERO), ONE), height), identity)), vmirror), dmirror)
    return func(grid)

def upscale_horizontal_by_two(grid):
    return hupscale(grid, TWO)

def upscale_vertical_by_two(grid):
    return vupscale(grid, TWO)

def upscale_horizontal_by_three(grid):
    return hupscale(grid, THREE)

def upscale_vertical_by_three(grid):
    return vupscale(grid, THREE)

def upscale_by_two(grid):
    return upscale(grid, TWO)

def upscale_by_three(grid):
    return upscale(grid, THREE)

def clear_outline(grid):
    func = fork(paint, identity, chain(lbind(recolor, ZERO), box, asindices))
    return func(grid)

def clear_all_but_outline(grid):
    func = fork(paint, identity, compose(lbind(recolor, ZERO), fork(difference, asindices, compose(box, asindices))))
    return func(grid)

def clear_top_row(grid):
    func = fork(paint, identity, compose(lbind(recolor, ZERO), fork(connect, compose(ulcorner, asobject), compose(urcorner, asobject))))
    return func(grid)

def clear_bottom_row(grid):
    func = chain(hmirror, fork(paint, identity, compose(lbind(recolor, ZERO), fork(connect, compose(ulcorner, asobject), compose(urcorner, asobject)))), hmirror)
    return func(grid)

def clear_left_column(grid):
    func = chain(rot270, fork(paint, identity, compose(lbind(recolor, ZERO), fork(connect, compose(ulcorner, asobject), compose(urcorner, asobject)))), rot90)
    return func(grid)

def clear_right_column(grid):
    func = chain(rot90, fork(paint, identity, compose(lbind(recolor, ZERO), fork(connect, compose(ulcorner, asobject), compose(urcorner, asobject)))), rot270)
    return func(grid)

def clear_diagonal(grid):
    func = fork(paint, identity, compose(lbind(recolor, ZERO), fork(connect, compose(ulcorner, asindices), compose(lrcorner, asindices))))
    return func(grid)

def clear_counterdiagonal(grid):
    func = fork(paint, identity, compose(lbind(recolor, ZERO), fork(connect, compose(urcorner, asindices), compose(llcorner, asindices))))
    return func(grid)


def rep_first_row(grid):
    return repeat(first(grid), height(grid))

def rep_last_row(grid):
    return repeat(last(grid), height(grid))

def rep_first_col(grid):
    rot_grid = rot90(grid)
    return rot270(repeat(first(rot_grid), height(rot_grid)))

def rep_last_col(grid):
    rot_grid = rot270(grid)
    return rot90(repeat(first(rot_grid), height(rot_grid)))

def remove_top_row(grid):
    func = fork(subgrid, compose(lbind(insert, DOWN), chain(initset, decrement, shape)), identity)
    return func(grid)

def remove_bottom_row(grid):
    func = chain(hmirror, fork(subgrid, compose(lbind(insert, DOWN), chain(initset, decrement, shape)), identity), hmirror)
    return func(grid)

def remove_left_column(grid):
    func = chain(rot270, fork(subgrid, compose(lbind(insert, DOWN), chain(initset, decrement, shape)), identity), rot90)
    return func(grid)

def remove_right_column(grid):
    func = chain(rot90, fork(subgrid, compose(lbind(insert, DOWN), chain(initset, decrement, shape)), identity), rot270)
    return func(grid)

# TODO: test this
def inner_columns(grid):
    return subgrid(
                insert(add(shape(grid), multiply(LEFT, TWO)), initset(RIGHT)),
                grid)

# TODO: test this
def inner_rows(grid):
    return subgrid(
                insert(add(multiply(UP, TWO), shape(grid)), initset(DOWN)),
                grid)

def gravitate_right(grid):
    func = lbind(apply, fork(combine, compose(lbind(repeat, ZERO), compose(rbind(colorcount, ZERO), rbind(repeat, ONE))), lbind(remove, ZERO)))
    return func(grid)

def gravitate_left(grid):
    func = lbind(apply, fork(combine, lbind(remove, ZERO), chain(lbind(repeat, ZERO), rbind(colorcount, ZERO), rbind(repeat, ONE))))
    return func(grid)

def gravitate_up(grid):
    func = chain(rot270, lbind(apply, fork(combine, compose(lbind(repeat, ZERO), compose(rbind(colorcount, ZERO), rbind(repeat, ONE))), lbind(remove, ZERO))), rot90)
    return func(grid)

def gravitate_down(grid):
    func = chain(rot270, lbind(apply, fork(combine, lbind(remove, ZERO), chain(lbind(repeat, ZERO), rbind(colorcount, ZERO), rbind(repeat, ONE)))), rot90)
    return func(grid)

def gravitate_left_right(grid):
    func = fork(hconcat, compose(lbind(apply, fork(combine, lbind(remove, ZERO), chain(lbind(repeat, ZERO), rbind(colorcount, ZERO), rbind(repeat, ONE)))), lefthalf), compose(lbind(apply, fork(combine, compose(lbind(repeat, ZERO), compose(rbind(colorcount, ZERO), rbind(repeat, ONE))), lbind(remove, ZERO))), righthalf))
    return func(grid)

def gravitate_top_down(grid):
    func = fork(vconcat, compose(chain(rot270, lbind(apply, fork(combine, compose(lbind(repeat, ZERO), compose(rbind(colorcount, ZERO), rbind(repeat, ONE))), lbind(remove, ZERO))), rot90), tophalf), compose(chain(rot270, lbind(apply, fork(combine, lbind(remove, ZERO), chain(lbind(repeat, ZERO), rbind(colorcount, ZERO), rbind(repeat, ONE)))), rot90), bottomhalf))
    return func(grid)

def shift_left(grid):
    tmp = remove_left_column(grid)
    tmp = insert_right_col(tmp)
    return tmp

def shift_right(grid):
    tmp = remove_right_column(grid)
    tmp = insert_left_col(tmp)
    return tmp

def shift_up(grid):
    tmp = remove_top_row(grid)
    tmp = insert_bottom_row(tmp)
    return tmp

def shift_down(grid):
    tmp = remove_bottom_row(grid)
    tmp = insert_top_row(tmp)
    return tmp

def wrap_left(grid):
    func = fork(hconcat, fork(subgrid, compose(compose(lbind(insert, RIGHT), initset), fork(astuple, compose(decrement, height), compose(decrement, width))), identity), fork(subgrid, compose(compose(lbind(insert, ORIGIN), initset), compose(toivec, compose(decrement, height))), identity))
    return func(grid)

def wrap_right(grid):
    func = chain(vmirror, fork(hconcat, fork(subgrid, compose(compose(lbind(insert, RIGHT), initset), fork(astuple, compose(decrement, height), compose(decrement, width))), identity), fork(subgrid, compose(compose(lbind(insert, ORIGIN), initset), compose(toivec, compose(decrement, height))), identity)), vmirror)
    return func(grid)

def wrap_up(grid):
    func = chain(rot90, fork(hconcat, fork(subgrid, compose(compose(lbind(insert, RIGHT), initset), fork(astuple, compose(decrement, height), compose(decrement, width))), identity), fork(subgrid, compose(compose(lbind(insert, ORIGIN), initset), compose(toivec, compose(decrement, height))), identity)), rot270)
    return func(grid)

def wrap_down(grid):
    func = chain(rot270, fork(hconcat, fork(subgrid, compose(compose(lbind(insert, RIGHT), initset), fork(astuple, compose(decrement, height), compose(decrement, width))), identity), fork(subgrid, compose(compose(lbind(insert, ORIGIN), initset), compose(toivec, compose(decrement, height))), identity)), rot90)
    return func(grid)

# TODO: test this
def outer_columns(grid):
    return hconcat(first(hsplit(grid, width(grid))), last(hsplit(grid, width(grid))))

# TODO: test this
def outer_rows(grid):
    return vconcat(first(vsplit(grid, height(grid))), last(vsplit(grid, height(grid))))

# TODO: test this
def left_column(grid):
    return first(hsplit(grid, width(grid)))

# TODO: test this
def right_column(grid):
    return last(hsplit(grid, width(grid)))

# TODO: test this
def top_row(grid):
    return first(vsplit(grid, height(grid)))

# TODO: test this
def bottom_row(grid):
    return last(vsplit(grid, height(grid)))

def first_quadrant(grid):
    return tophalf(lefthalf(grid))

def second_quadrant(grid):
    return tophalf(righthalf(grid))

def third_quadrant(grid):
    return bottomhalf(lefthalf(grid))

def fourth_quadrant(grid):
    return bottomhalf(righthalf(grid))

def identity(x):
    """ identity function """
    return x

def add(a, b):
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)

def subtract(a, b):
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)


def multiply(a, b):
    """ multiplication """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)

def divide(a, b):
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)

def invert(n):
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])

def even(n):
    """ evenness """
    return n % 2 == 0

def double(n):
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)

def halve(n):
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)

def flip(b):
    """ logical not """
    return not b

def equality(a, b):
    """ equality """
    return a == b

def contained(value, container):
    """ element of """
    return value in container

def combine(a, b):
    """ union """
    return type(a)((*a, *b))

def intersection(a, b):
    """ returns the intersection of two containers """
    return a & b

def difference(a, b):
    """ difference """
    return type(a)(e for e in a if e not in b)

def dedupe(iterable):
    """ remove duplicates """
    return tuple(e for i, e in enumerate(iterable) if iterable.index(e) == i)

def order(container, compfunc):
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))

def repeat(item, num):
    """ repetition of item within vector """
    return tuple(item for i in range(num))

def greater(a, b):
    """ greater """
    return a > b

def size(container):
    """ cardinality """
    return len(container)

def merge(containers):
    """ merging """
    return type(containers)(e for c in containers for e in c)

def maximum(container):
    """ maximum """
    return max(container, default=0)

def minimum(container):
    """ minimum """
    return min(container, default=0)

def valmax(container, compfunc):
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))

def valmin(container, compfunc):
    """ minimum by custom function """
    return compfunc(min(container, key=compfunc, default=0))

def argmax(container, compfunc):
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def argmin(container, compfunc):
    """ smallest item by custom order """
    return min(container, key=compfunc, default=None)

def mostcommon(container):
    """ most common item """
    return max(set(container), key=container.count)

def leastcommon(container):
    """ least common item """
    return min(set(container), key=container.count)

def initset(value):
    """ initialize container """
    return frozenset({value})

def both(a, b):
    """ logical and """
    return a and b

def either(a, b):
    """ logical or """
    return a or b

def increment(x):
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def decrement(x):
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)

def crement(x):
    """ incrementing positive and decrementing negative """
    if isinstance(x, int):
        return 0 if x == 0 else (x + 1 if x > 0 else x - 1)
    return (
        0 if x[0] == 0 else (x[0] + 1 if x[0] > 0 else x[0] - 1),
        0 if x[1] == 0 else (x[1] + 1 if x[1] > 0 else x[1] - 1)
    )

def sign(x):
    """ sign """
    if isinstance(x, int):
        return 0 if x == 0 else (1 if x > 0 else -1)
    return (
        0 if x[0] == 0 else (1 if x[0] > 0 else -1),
        0 if x[1] == 0 else (1 if x[1] > 0 else -1)
    )

def positive(x):
    """ positive """
    return x > 0

def toivec(i):
    """ vector pointing vertically """
    return (i, 0)

def tojvec(j):
    """ vector pointing horizontally """
    return (0, j)

def sfilter(container, condition):
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def mfilter(container, function):
    """ filter and merge """
    return merge(sfilter(container, function))

def extract(container, condition):
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))

def totuple(container):
    """ conversion to tuple """
    return tuple(container)

def first(container):
    """ first item of container """
    return next(iter(container))

def last(container):
    """ last item of container """
    return max(enumerate(container))[1]

def insert(value, container):
    """ insert item into container """
    return container.union(frozenset({value}))

def remove(value, container):
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def other(container, value):
    """ other value in the container """
    return first(remove(value, container))

def interval(start, stop, step):
    """ range """
    return tuple(range(start, stop, step))

def astuple(a, b):
    """ constructs a tuple """
    return (a, b)

def product(a, b):
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)

def pair(a, b):
    """ zipping of two tuples """
    return tuple(zip(a, b))

def branch(condition, if_value, else_value):
    """ if else branching """
    return if_value if condition else else_value

def compose(outer, inner):
    """ function composition """
    return lambda x: outer(inner(x))

def chain(h, g, f):
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

def matcher(function, target):
    """ construction of equality function """
    return lambda x: function(x) == target

def rbind(function, fixed):
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)

def lbind(function, fixed):
    """ fix the leftmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)

def power(function, n):
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))

def fork(outer, a, b):
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def apply(function, container):
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

def rapply(functions, value):
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

def mapply(function, container):
    """ apply and merge """
    return merge(apply(function, container))

def papply(function, a, b):
    """ apply function on two vectors """
    return tuple(function(i, j) for i, j in zip(a, b))

def mpapply(function, a, b):
    """ apply function on two vectors and merge """
    return merge(papply(function, a, b))

def prapply(function, a, b):
    """ apply function on cartesian product """
    return frozenset(function(i, j) for j in b for i in a)

def mostcolor(element):
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def leastcolor(element):
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

def height(piece):
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1

def width(piece):
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

def shape(piece):
    """ height and width of grid or patch """
    return (height(piece), width(piece))

def portrait(piece):
    """ whether height is greater than width """
    return height(piece) > width(piece)

def colorcount(element, value):
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

def colorfilter(objs, value):
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

def sizefilter(container, n):
    """ filter items by size """
    return frozenset(item for item in container if len(item) == n)

def asindices(grid):
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

def ofcolor(grid, value):
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def ulcorner(patch):
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def urcorner(patch):
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def llcorner(patch):
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))

def lrcorner(patch):
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))

def crop(grid, start, dims):
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def toindices(patch):
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

def recolor(value, patch):
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

def shift(patch, directions):
    """ shift patch """
    if len(patch) == 0:
        return patch
    di, dj = directions
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)
    return frozenset((i + di, j + dj) for i, j in patch)

def normalize(patch):
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

def dneighbors(loc):
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})

def ineighbors(loc):
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})

def neighbors(loc):
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)

def objects(grid, univalued, diagonal, without_bg):
    """ objects occurring on the grid """
    bg = mostcolor(grid) if without_bg else None
    objs = set()
    occupied = set()
    h, w = len(grid), len(grid[0])
    unvisited = asindices(grid)
    diagfun = neighbors if diagonal else dneighbors
    for loc in unvisited:
        if loc in occupied:
            continue
        val = grid[loc[0]][loc[1]]
        if val == bg:
            continue
        obj = {(val, loc)}
        cands = {loc}
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    neighborhood |= {
                        (i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w
                    }
            cands = neighborhood - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)

def partition(grid):
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )

def fgpartition(grid):
    """ each cell with the same value part of the same object without background """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid) - {mostcolor(grid)}
    )

def uppermost(patch):
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))

def lowermost(patch):
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindices(patch))

def leftmost(patch):
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))

def rightmost(patch):
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindices(patch))

def square(piece):
    """ whether the piece forms a square """
    return len(piece) == len(piece[0]) if isinstance(piece, tuple) else height(piece) * width(piece) == len(piece) and height(piece) == width(piece)

def vline(patch):
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1

def hline(patch):
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1

def hmatching(a, b):
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindices(a)) & set(i for i, j in toindices(b))) > 0

def vmatching(a, b):
    """ whether there exists a column for which both patches have cells """
    return len(set(j for i, j in toindices(a)) & set(j for i, j in toindices(b))) > 0

def manhattan(a, b):
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindices(a) for bi, bj in toindices(b))

def adjacent(a, b):
    """ whether two patches are adjacent """
    return manhattan(a, b) == 1

def bordering(patch, grid):
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1

def centerofmass(patch):
    """ center of mass """
    return tuple(map(lambda x: sum(x) // len(patch), zip(*toindices(patch))))

def palette(element):
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def numcolors(element):
    """ number of colors occurring in object or grid """
    return len(palette(element))

def color(obj):
    """ color of object """
    return next(iter(obj))[0]

def toobject(patch, grid):
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)

def asobject(grid):
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def rot90(grid):
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def rot180(grid):
    """ half rotation """
    return tuple(tuple(row[::-1]) for row in grid[::-1])


def rot270(grid):
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]


def hmirror(piece):
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)

def vmirror(piece):
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)

def dmirror(piece):
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)

def cmirror(piece):
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))

def fill(grid, value, patch):
    """ fill value at indices """
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)

def paint(grid, obj):
    """ paint object to grid """
    h, w = len(grid), len(grid[0])
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)

def underfill(grid, value, patch):
    """ fill value at indices that are background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if grid_filled[i][j] == bg:
                grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)

def underpaint(grid, obj):
    """ paint object to grid where there is background """
    h, w = len(grid), len(grid[0])
    bg = 0
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            if grid_painted[i][j] == bg:
                grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)

def hupscale(grid, factor):
    """ upscale grid horizontally """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_row = tuple()
        for value in row:
            upscaled_row = upscaled_row + tuple(value for num in range(factor))
        upscaled_grid = upscaled_grid + (upscaled_row,)
    return upscaled_grid

def vupscale(grid, factor):
    """ upscale grid vertically """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_grid = upscaled_grid + tuple(row for num in range(factor))
    return upscaled_grid

def upscale(element, factor):
    """ upscale object or grid """
    if isinstance(element, tuple):
        upscaled_grid = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            upscaled_grid = upscaled_grid + tuple(upscaled_row for num in range(factor))
        return upscaled_grid
    else:
        if len(element) == 0:
            return frozenset()
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        upscaled_obj = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    upscaled_obj.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(upscaled_obj), (di_inv, dj_inv))

def downscale(grid, factor):
    """ downscale grid """
    h, w = len(grid), len(grid[0])
    downscaled_grid = tuple()
    for i in range(h):
        downscaled_row = tuple()
        for j in range(w):
            if j % factor == 0:
                downscaled_row = downscaled_row + (grid[i][j],)
        downscaled_grid = downscaled_grid + (downscaled_row, )
    h = len(downscaled_grid)
    downscaled_grid2 = tuple()
    for i in range(h):
        if i % factor == 0:
            downscaled_grid2 = downscaled_grid2 + (downscaled_grid[i],)
    return downscaled_grid2

def hconcat(a, b):
    """ concatenate two grids horizontally """
    if len(a) != len(b):
        raise Exception("Grids must have the same number of rows for horizontal concatenation")

    return tuple(i + j for i, j in zip(a, b))

def vconcat(a, b):
    """ concatenate two grids vertically """
    if len(a[0]) != len(b[0]):
        raise Exception("Grids must have the same number of columns for vertical concatenation")
        
    return a + b

def subgrid(patch, grid):
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

def hsplit(grid, n):
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))

def vsplit(grid, n):
    """ split grid vertically """
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))

# TODO: is this redundant with the cellwise AND/OR/XOR/Difference primitives?
def cellwise(a, b, fallback):
    """ cellwise match of two grids """
    h, w = len(a), len(a[0])
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            value = a_value if a_value == b[i][j] else fallback
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return resulting_grid

def replace(grid, replacee, replacer):
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

def switch(grid, a, b):
    """ color switching """
    return tuple(tuple(v if (v != a and v != b) else {a: b, b: a}[v] for v in r) for r in grid)

def center(patch):
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)

def position(a, b):
    """ relative position between two patches """
    ia, ja = center(toindices(a))
    ib, jb = center(toindices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)

def index(grid, loc):
    """ color at location """
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]]

def canvas(value, dimensions):
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def corners(patch):
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})

def connect(a, b):
    """ line between two points """
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej)))
    elif bi - ai == aj - bj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1)))
    return frozenset()

def cover(grid, patch):
    """ remove object from grid """
    return fill(grid, mostcolor(grid), toindices(patch))

def trim(grid):
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])

def move(grid, obj, offset):
    """ move object on grid """
    return paint(cover(grid, obj), shift(obj, offset))

def topthird(grid):
    return grid[:len(grid) // 3]

def bottomthird(grid):
    return grid[2 * (len(grid) // 3) + len(grid) % 3:]

def vcenterthird(grid):
    sub_grid_dim = len(grid) // 3
    offset = 1
    if 3 * sub_grid_dim == len(grid):
        offset = 0
    start_row = len(grid) // 3 + offset
    end_row = 2 * start_row

    tmp = grid[start_row:end_row]
    return tmp

def hcenterthird(grid):
    return rot270(vcenterthird(rot90(grid)))

def leftthird(grid):
    return rot270(topthird(rot90(grid)))

def rightthird(grid):
    return rot270(bottomthird(rot90(grid)))

def tophalf(grid):
    """ upper half of grid """
    return grid[:len(grid) // 2]

def bottomhalf(grid):
    """ lower half of grid """
    return grid[len(grid) // 2 + len(grid) % 2:]

def lefthalf(grid):
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))

def righthalf(grid):
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))

def vfrontier(location):
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

def hfrontier(location):
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

def backdrop(patch):
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))

def delta(patch):
    """ indices in bounding box but not part of patch """
    if len(patch) == 0:
        return frozenset({})
    return backdrop(patch) - toindices(patch)

def gravitate(source, destination):
    """ direction to move source until adjacent to destination """
    source_i, source_j = center(source)
    destination_i, destination_j = center(destination)
    i, j = 0, 0
    if vmatching(source, destination):
        i = 1 if source_i < destination_i else -1
    else:
        j = 1 if source_j < destination_j else -1
    direction = (i, j)
    gravitation_i, gravitation_j = i, j
    maxcount = 42
    c = 0
    while not adjacent(source, destination) and c < maxcount:
        c += 1
        gravitation_i += i
        gravitation_j += j
        source = shift(source, direction)
    return (gravitation_i - i, gravitation_j - j)

def inbox(patch):
    """ inbox for patch """
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

def outbox(patch):
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

def box(patch):
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

def shoot(start, direction):
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))

def occurrences(grid, obj):
    """ locations of occurrences of object in grid """
    occurrences = set()
    normed = normalize(obj)
    h, w = len(grid), len(grid[0])
    for i in range(h):
        for j in range(w):
            occurs = True
            for v, (a, b) in shift(normed, (i, j)):
                if 0 <= a < h and 0 <= b < w:
                    if grid[a][b] != v:
                        occurs = False
                        break
                else:
                    occurs = False
                    break
            if occurs:
                occurrences.add((i, j))
    return frozenset(occurrences)

def frontiers(grid):
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers

def compress(grid):
    """ removes frontiers from grid """
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri)

def hperiod(obj):
    """ horizontal periodicity """
    normalized = normalize(obj)
    w = width(normalized)
    for p in range(1, w):
        offsetted = shift(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            return p
    return w

def vperiod(obj):
    """ vertical periodicity """
    normalized = normalize(obj)
    h = height(normalized)
    for p in range(1, h):
        offsetted = shift(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            return p
    return h

def duplicate_top_row(grid):
    """ Duplicates the top row of the grid """
    top_row = grid[0]
    print("top_row = ", top_row)
    return (top_row,) + grid

def duplicate_bottom_row(grid):
    """ Duplicates the bottom row of the grid """
    bottom_row = grid[-1]
    return grid + (bottom_row,)

def duplicate_left_column(grid):
    """ Duplicates the left column of the grid """
    return tuple(row[:1] + row for row in grid)

def duplicate_right_column(grid):
    """ Duplicates the right column of the grid """
    return tuple(row + row[-1:] for row in grid)

def set_fg_color(grid, color):
    px_indices = difference(asindices(grid), ofcolor(grid, 0))

    return fill(grid, color, px_indices)

def color_change(grid, from_color, to_color):
    px_indices = ofcolor(grid, from_color)

    return fill(grid, to_color, px_indices)

def color_swap(grid, color1, color2):
    px_indices1 = ofcolor(grid, color1)
    px_indices2 = ofcolor(grid, color2)

    tmp = fill(grid, color2, px_indices1)
    return fill(tmp, color1, px_indices2)

def invert_colors(grid):
    bg_indices = ofcolor(grid, 0)
    fg_indices = difference(asindices(grid), bg_indices)
    fg_color = dominant_color(grid)

    tmp = fill(grid, fg_color, bg_indices)
    return fill(tmp, 0, fg_indices)

def cellwiseAND(a, b):
    """ cellwise match of two grids """
    h, w = min(len(a), len(b)), min(len(a[0]), len(b[0]))
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            b_value = b[i][j]
            value = 0
            if a_value != 0 and b_value != 0:
                value = a_value
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return resulting_grid

def cellwiseXOR(a, b):
    """ cellwise match of two grids """
    h, w = min(len(a), len(b)), min(len(a[0]), len(b[0]))
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            b_value = b[i][j]
            value = 0
            if a_value != 0 and b_value == 0:
                value = a_value
            elif a_value == 0 and b_value != 0:
                value = b_value
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return resulting_grid

def cellwiseOR(a, b):
    """ cellwise match of two grids """
    """ logic: draw a pixel if either grid has a foreground pixel there (prioritize color a if both have one) """
    h, w = min(len(a), len(b)), min(len(a[0]), len(b[0]))
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            b_value = b[i][j]
            if a_value != 0:
                value = a_value
            else:
                value = b_value
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return resulting_grid

def dominant_color(a):
    # most_color:
    values = [v for r in a for v in r] if isinstance(a, tuple) else [v for v, _ in a]
    filtered_lst = [num for num in values if num != 0]
    if len(filtered_lst) == 0:
        return 0
    else:
        return max(set(filtered_lst), key=values.count)

def cellwiseNOR(a, b):
    """ cellwise match of two grids """
    """ logic: draw a pixel if neither grid has a foreground pixel there. color: most dominant non-zero color of a """
    h, w = min(len(a), len(b)), min(len(a[0]), len(b[0]))
    resulting_grid = tuple()
    draw_color = dominant_color(a)
    if draw_color == 0:
        draw_color = 1
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            b_value = b[i][j]
            value = 0
            if a_value == 0 and b_value == 0:
                value = draw_color
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return resulting_grid

def cellwiseDifference(a, b):
    """ cellwise match of two grids """
    """ logic: draw a pixel if a has one there, but not b """
    h, w = min(len(a), len(b)), min(len(a[0]), len(b[0]))
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            b_value = b[i][j]
            value = 0
            if a_value != 0 and b_value == 0:
                value = a_value
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return resulting_grid

def get_primitives(n):
    output = {}
    for name, func in semantics.items():
        if n == 1 and is_1arg(name):
            output[name] = func
        elif n == 2 and not is_1arg(name):
            output[name] = func

    return output

def is_1arg(prim_name):
    idx = prim_indices[prim_name]
    if idx < 81 or idx > 87:
        return True
    else:
        return False

def get_num_args(prim_name):
    if prim_name.startswith('color'):
        return 1
    
    lambda_func = semantics[prim_name]
    count = 0
    while callable(lambda_func):
        sig = inspect.signature(lambda_func)
        params = sig.parameters
        count += len(params)
        # Get the next nested function by invoking the lambda
        if count == 0:
            break
        try:
            lambda_func = lambda_func(*(None for _ in params))
        except TypeError:
            break
    return count

def is_diagonal_primitive(prim_name):
    if 'keep_only_diagonal' in prim_name or 'keep_only_counterdiagonal' in prim_name or \
       'clear_diagonal' in prim_name or 'clear_counterdiagonal' in prim_name or 'dmirror' in prim_name or \
       'cmirror' in prim_name:
        return True
    else:
        return False

def is_color_primitive(prim_name):
    if prim_name.startswith('color') or prim_name.endswith('+color_swap') or prim_name.endswith('+color_change'):
        return True
    else:
        return False

prim_indices = {
    # Atomic primitives
    'set_fg_color1': 0,
    'set_fg_color2': 1,
    'set_fg_color3': 2,
    'set_fg_color4': 3,
    'set_fg_color5': 4,
    'set_fg_color6': 5,
    'set_fg_color7': 6,
    'set_fg_color8': 7,
    'set_fg_color9': 8,
    'shear_rows_left': 9,
    'shear_rows_right': 10,
    'shear_cols_down': 11,
    'shear_cols_up': 12,
    'wrap_left': 13,
    'wrap_right': 14,
    'wrap_up': 15,
    'wrap_down': 16,
    'keep_only_diagonal': 17,
    'keep_only_counterdiagonal': 18,
    'vmirror': 19,
    'hmirror': 20,
    'dmirror': 21,
    'cmirror': 22,
    'rot90': 23,
    'tophalf': 24,
    'bottomhalf': 25,
    'lefthalf': 26,
    'righthalf': 27,
    'drag_down_underpaint': 28,
    'drag_down_paint': 29,
    'drag_left_underpaint': 30,
    'drag_left_paint': 31,
    'drag_up_underpaint': 32,
    'drag_up_paint': 33,
    'drag_right_underpaint': 34,
    'drag_right_paint': 35,
    'drag_diagonally_underpaint': 36,
    'drag_diagonally_paint': 37,
    'drag_counterdiagonally_underpaint': 38,
    'drag_counterdiagonally_paint': 39,
    'extend_by_one': 40,
    'insert_top_row': 41,
    'insert_bottom_row': 42,
    'insert_left_col': 43,
    'insert_right_col': 44,
    'stack_rows_horizontally': 45,
    'stack_rows_vertically': 46,
    'symmetrize_left_around_vertical': 47,
    'symmetrize_right_around_vertical': 48,
    'symmetrize_top_around_horizontal': 49,
    'symmetrize_bottom_around_horizontal': 50,
    'symmetrize_quad': 51,
    'upscale_horizontal_by_two': 52,
    'upscale_vertical_by_two': 53,
    'clear_all_but_outline': 54,
    'clear_top_row': 55,
    'clear_bottom_row': 56,
    'clear_left_column': 57,
    'clear_right_column': 58,
    'clear_diagonal': 59,
    'clear_counterdiagonal': 60,
    'rep_first_row': 61,
    'rep_last_row': 62,
    'rep_first_col': 63,
    'rep_last_col': 64,
    'remove_top_row': 65,
    'remove_bottom_row': 66,
    'remove_left_column': 67,
    'remove_right_column': 68,
    'gravitate_right': 69,
    'gravitate_left': 70,
    'gravitate_up': 71,
    'gravitate_down': 72,
    'gravitate_left_right': 73,
    'gravitate_top_down': 74,
    'topthird': 75,
    'vcenterthird': 76,
    'bottomthird': 77,
    'leftthird': 78,
    'hcenterthird': 79,
    'rightthird': 80,
    'cellwiseOR': 81,
    'cellwiseAND': 82,
    'cellwiseXOR': 83,
    'cellwiseDifference': 84,
    'cellwiseNOR': 85,
    'vconcat': 86,
    'hconcat': 87,

    # 1-deep, but manually managed, because too complicated to use generate_2deep_DSL to produce 2-deep combos
    'color_change': 88,
    'invert_colors': 89,

    'upscale_horizontal_by_three': 90,
    'upscale_vertical_by_three': 91,
    'upscale_by_three': 92,
    'first_quadrant': 93,
    'second_quadrant': 94,
    'third_quadrant': 95,
    'fourth_quadrant': 96,
    'hfirstfourth': 97,
    'hsecondfourth': 98,
    'hthirdfourth': 99,
    'hlastfourth': 100,
    'vfirstfourth': 101,
    'vsecondfourth': 102,
    'vthirdfourth': 103,
    'vlastfourth': 104,

    'rot180': 105,
    'rot270': 106,
    'clear_outline': 107,    # this is equal to clear both rows and both columns
    'trim': 108,
    'duplicate_top_row': 109,
    'duplicate_bottom_row': 110,
    'duplicate_left_column':111,
    'duplicate_right_column':112
}

semantics = {
    # Atomic 1-arg primitives
    'set_fg_color1': lambda g: set_fg_color(g, 1),
    'set_fg_color2': lambda g: set_fg_color(g, 2),
    'set_fg_color3': lambda g: set_fg_color(g, 3),
    'set_fg_color4': lambda g: set_fg_color(g, 4),
    'set_fg_color5': lambda g: set_fg_color(g, 5),
    'set_fg_color6': lambda g: set_fg_color(g, 6),
    'set_fg_color7': lambda g: set_fg_color(g, 7),
    'set_fg_color8': lambda g: set_fg_color(g, 8),
    'set_fg_color9': lambda g: set_fg_color(g, 9),
    'shear_rows_left': lambda g: shear_rows_left(g),
    'shear_rows_right': lambda g: shear_rows_right(g),
    'shear_cols_down': lambda g: shear_cols_down(g),
    'shear_cols_up': lambda g: shear_cols_up(g),
    'wrap_left': lambda g: wrap_left(g),
    'wrap_right': lambda g: wrap_right(g),
    'wrap_up': lambda g: wrap_up(g),
    'wrap_down': lambda g: wrap_down(g),
    'keep_only_diagonal': lambda g: keep_only_diagonal(g),                  # If the diagonal is clear, this results in empty grid. If only a diagonal, results in identity.
    'keep_only_counterdiagonal': lambda g: keep_only_counterdiagonal(g),
    'vmirror': lambda g: vmirror(g),
    'hmirror': lambda g: hmirror(g),
    'dmirror': lambda g: dmirror(g),
    'cmirror': lambda g: cmirror(g),
    'rot90': lambda g: rot90(g),
    'tophalf': lambda g: tophalf(g),
    'bottomhalf': lambda g: bottomhalf(g),
    'lefthalf': lambda g: lefthalf(g),
    'righthalf': lambda g: righthalf(g),
    'drag_down_underpaint': lambda g: drag_down_underpaint(g),
    'drag_down_paint': lambda g: drag_down_paint(g),
    'drag_left_underpaint': lambda g: drag_left_underpaint(g),
    'drag_left_paint': lambda g: drag_left_paint(g),
    'drag_up_underpaint': lambda g: drag_up_underpaint(g),
    'drag_up_paint': lambda g: drag_up_paint(g),
    'drag_right_underpaint': lambda g: drag_right_underpaint(g),
    'drag_right_paint': lambda g: drag_right_paint(g),
    'drag_diagonally_underpaint': lambda g: drag_diagonally_underpaint(g),
    'drag_diagonally_paint': lambda g: drag_diagonally_paint(g),
    'drag_counterdiagonally_underpaint': lambda g: drag_counterdiagonally_underpaint(g),
    'drag_counterdiagonally_paint': lambda g: drag_counterdiagonally_paint(g),
    'extend_by_one': lambda g: extend_by_one(g),
    'insert_top_row': lambda g: insert_top_row(g),
    'insert_bottom_row': lambda g: insert_bottom_row(g),
    'insert_left_col': lambda g: insert_left_col(g),
    'insert_right_col': lambda g: insert_right_col(g),
    'stack_rows_horizontally': lambda g: stack_rows_horizontally(g),
    'stack_rows_vertically': lambda g: stack_rows_vertically(g),
    'symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(g),
    'symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(g),
    'symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(g),
    'symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(g),
    'symmetrize_quad': lambda g: symmetrize_quad(g),
    'upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(g),
    'upscale_vertical_by_two': lambda g: upscale_vertical_by_two(g),
    'clear_all_but_outline': lambda g: clear_all_but_outline(g),
    'clear_top_row': lambda g: clear_top_row(g),
    'clear_bottom_row': lambda g: clear_bottom_row(g),
    'clear_left_column': lambda g: clear_left_column(g),
    'clear_right_column': lambda g: clear_right_column(g),
    'clear_diagonal': lambda g: clear_diagonal(g),
    'clear_counterdiagonal': lambda g: clear_counterdiagonal(g),
    'rep_first_row': lambda g: rep_first_row(g),
    'rep_last_row': lambda g: rep_last_row(g),
    'rep_first_col': lambda g: rep_first_col(g),
    'rep_last_col': lambda g: rep_last_col(g),
    'remove_top_row': lambda g: remove_top_row(g),
    'remove_bottom_row': lambda g: remove_bottom_row(g),
    'remove_left_column': lambda g: remove_left_column(g),
    'remove_right_column': lambda g: remove_right_column(g),
    'gravitate_right': lambda g: gravitate_right(g),
    'gravitate_left': lambda g: gravitate_left(g),
    'gravitate_up': lambda g: gravitate_up(g),
    'gravitate_down': lambda g: gravitate_down(g),
    'gravitate_left_right': lambda g: gravitate_left_right(g),
    'gravitate_top_down': lambda g: gravitate_top_down(g),
    'topthird': lambda g: topthird(g),
    'vcenterthird': lambda g: vcenterthird(g),
    'bottomthird': lambda g: bottomthird(g),
    'leftthird': lambda g: leftthird(g),
    'hcenterthird': lambda g: hcenterthird(g),
    'rightthird': lambda g: rightthird(g),

    # 2-arg primitives
    'cellwiseOR': lambda g1: lambda g2: cellwiseOR(g1, g2),
    'cellwiseAND': lambda g1: lambda g2: cellwiseAND(g1, g2),
    'cellwiseXOR': lambda g1: lambda g2: cellwiseXOR(g1, g2),
    'cellwiseDifference': lambda g1: lambda g2: cellwiseDifference(g1, g2),
    'cellwiseNOR': lambda g1: lambda g2: cellwiseNOR(g1, g2),
    'vconcat': lambda g1: lambda g2: vconcat(g1, g2),
    'hconcat': lambda g1: lambda g2: hconcat(g1, g2),

    # separate 1-arg primitives so they're not included automatically in atomic primitives for generate_2deep_DSL
    'color_change': lambda g: lambda c1: lambda c2: color_change(g, c1, c2),
    'invert_colors': lambda g: invert_colors(g),

    # these are not 2-deep compositions of others, but I don't want to compose them either
    'upscale_horizontal_by_three': lambda g: upscale_horizontal_by_three(g),
    'upscale_vertical_by_three': lambda g: upscale_vertical_by_three(g),
    'upscale_by_three': lambda g: upscale_by_three(g),
    'first_quadrant': lambda g: first_quadrant(g),
    'second_quadrant': lambda g: second_quadrant(g),
    'third_quadrant': lambda g: third_quadrant(g),
    'fourth_quadrant': lambda g: fourth_quadrant(g),
    'hfirstfourth': lambda g: lefthalf(lefthalf(g)),
    'hsecondfourth': lambda g: righthalf(lefthalf(g)),
    'hthirdfourth': lambda g: lefthalf(righthalf(g)),
    'hlastfourth': lambda g: righthalf(righthalf(g)),
    'vfirstfourth': lambda g: tophalf(tophalf(g)),
    'vsecondfourth': lambda g: bottomhalf(tophalf(g)),
    'vthirdfourth': lambda g: tophalf(bottomhalf(g)),
    'vlastfourth': lambda g: bottomhalf(bottomhalf(g)),

    'rot180': lambda g: rot180(g),
    'rot270': lambda g: rot270(g),
    'clear_outline': lambda g: clear_outline(g),
    'trim': lambda g: trim(g),
    'duplicate_top_row': lambda g: duplicate_top_row(g),
    'duplicate_bottom_row': lambda g: duplicate_bottom_row(g),
    'duplicate_left_column': lambda g: duplicate_left_column(g),
    'duplicate_right_column': lambda g: duplicate_right_column(g)
}
