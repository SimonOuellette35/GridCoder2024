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
    if idx < 81 or (idx > 87 and idx < 156) or (idx > 176 and idx < 243) or (idx > 263 and get_num_args(prim_name) == 1):
        return True
    else:
        return False

def get_num_args(prim_name):
    if prim_name.startswith('color_swap'):
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

def is_atomic(prim_name):
    if prim_indices[prim_name] <= 87:
        return True
    else:
        return False

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
'set_fg_color1': 0,
'set_fg_color2': 1,
'set_fg_color3': 2,
'set_fg_color4': 3,
'set_fg_color5': 4,
'set_fg_color6': 5,
'set_fg_color7': 6,
'set_fg_color8': 7,
'set_fg_color9': 8,
'shift_left': 9,
'shift_right': 10,
'shift_up': 11,
'shift_down': 12,
'vmirror': 13,
'hmirror': 14,
'rot90': 15,
'tophalf': 16,
'bottomhalf': 17,
'lefthalf': 18,
'righthalf': 19,
'symmetrize_left_around_vertical': 20,
'symmetrize_right_around_vertical': 21,
'symmetrize_top_around_horizontal': 22,
'symmetrize_bottom_around_horizontal': 23,
'upscale_horizontal_by_two': 24,
'upscale_vertical_by_two': 25,
'upscale_by_two': 26,
'gravitate_right': 27,
'gravitate_left': 28,
'gravitate_up': 29,
'gravitate_down': 30,
'gravitate_left_right': 31,
'gravitate_top_down': 32,
'topthird': 33,
'vcenterthird': 34,
'bottomthird': 35,
'leftthird': 36,
'hcenterthird': 37,
'rightthird': 38,
'color_change': 39,
'invert_colors': 40,
'shift_left+color_change': 41,
'shift_right+color_change': 42,
'shift_up+color_change': 43,
'shift_down+color_change': 44,
'vmirror+color_change': 45,
'hmirror+color_change': 46,
'rot90+color_change': 47,
'tophalf+color_change': 48,
'bottomhalf+color_change': 49,
'lefthalf+color_change': 50,
'righthalf+color_change': 51,
'symmetrize_left_around_vertical+color_change': 52,
'symmetrize_right_around_vertical+color_change': 53,
'symmetrize_top_around_horizontal+color_change': 54,
'symmetrize_bottom_around_horizontal+color_change': 55,
'upscale_horizontal_by_two+color_change': 56,
'upscale_vertical_by_two+color_change': 57,
'gravitate_right+color_change': 58,
'gravitate_left+color_change': 59,
'gravitate_up+color_change': 60,
'gravitate_down+color_change': 61,
'gravitate_left_right+color_change': 62,
'gravitate_top_down+color_change': 63,
'shift_left+invert_colors': 64,
'shift_right+invert_colors': 65,
'shift_up+invert_colors': 66,
'shift_down+invert_colors': 67,
'vmirror+invert_colors': 68,
'hmirror+invert_colors': 69,
'rot90+invert_colors': 70,
'tophalf+invert_colors': 71,
'bottomhalf+invert_colors': 72,
'lefthalf+invert_colors': 73,
'righthalf+invert_colors': 74,
'symmetrize_left_around_vertical+invert_colors': 75,
'symmetrize_right_around_vertical+invert_colors': 76,
'symmetrize_top_around_horizontal+invert_colors': 77,
'symmetrize_bottom_around_horizontal+invert_colors': 78,
'upscale_horizontal_by_two+invert_colors': 79,
'upscale_vertical_by_two+invert_colors': 80,
'gravitate_right+invert_colors': 81,
'gravitate_left+invert_colors': 82,
'gravitate_up+invert_colors': 83,
'gravitate_down+invert_colors': 84,
'gravitate_left_right+invert_colors': 85,
'gravitate_top_down+invert_colors': 86,
'set_fg_color1+shift_left': 87,
'set_fg_color1+shift_right': 88,
'set_fg_color1+shift_up': 89,
'set_fg_color1+shift_down': 90,
'set_fg_color1+vmirror': 91,
'set_fg_color1+hmirror': 92,
'set_fg_color1+rot90': 93,
'set_fg_color1+tophalf': 94,
'set_fg_color1+bottomhalf': 95,
'set_fg_color1+lefthalf': 96,
'set_fg_color1+righthalf': 97,
'set_fg_color1+symmetrize_left_around_vertical': 98,
'set_fg_color1+symmetrize_right_around_vertical': 99,
'set_fg_color1+symmetrize_top_around_horizontal': 100,
'set_fg_color1+symmetrize_bottom_around_horizontal': 101,
'set_fg_color1+upscale_horizontal_by_two': 102,
'set_fg_color1+upscale_vertical_by_two': 103,
'set_fg_color1+gravitate_right': 104,
'set_fg_color1+gravitate_left': 105,
'set_fg_color1+gravitate_up': 106,
'set_fg_color1+gravitate_down': 107,
'set_fg_color1+gravitate_left_right': 108,
'set_fg_color1+gravitate_top_down': 109,
'set_fg_color1+topthird': 110,
'set_fg_color1+vcenterthird': 111,
'set_fg_color1+bottomthird': 112,
'set_fg_color1+leftthird': 113,
'set_fg_color1+hcenterthird': 114,
'set_fg_color1+rightthird': 115,
'set_fg_color2+shift_left': 116,
'set_fg_color2+shift_right': 117,
'set_fg_color2+shift_up': 118,
'set_fg_color2+shift_down': 119,
'set_fg_color2+vmirror': 120,
'set_fg_color2+hmirror': 121,
'set_fg_color2+rot90': 122,
'set_fg_color2+tophalf': 123,
'set_fg_color2+bottomhalf': 124,
'set_fg_color2+lefthalf': 125,
'set_fg_color2+righthalf': 126,
'set_fg_color2+symmetrize_left_around_vertical': 127,
'set_fg_color2+symmetrize_right_around_vertical': 128,
'set_fg_color2+symmetrize_top_around_horizontal': 129,
'set_fg_color2+symmetrize_bottom_around_horizontal': 130,
'set_fg_color2+upscale_horizontal_by_two': 131,
'set_fg_color2+upscale_vertical_by_two': 132,
'set_fg_color2+gravitate_right': 133,
'set_fg_color2+gravitate_left': 134,
'set_fg_color2+gravitate_up': 135,
'set_fg_color2+gravitate_down': 136,
'set_fg_color2+gravitate_left_right': 137,
'set_fg_color2+gravitate_top_down': 138,
'set_fg_color2+topthird': 139,
'set_fg_color2+vcenterthird': 140,
'set_fg_color2+bottomthird': 141,
'set_fg_color2+leftthird': 142,
'set_fg_color2+hcenterthird': 143,
'set_fg_color2+rightthird': 144,
'set_fg_color3+shift_left': 145,
'set_fg_color3+shift_right': 146,
'set_fg_color3+shift_up': 147,
'set_fg_color3+shift_down': 148,
'set_fg_color3+vmirror': 149,
'set_fg_color3+hmirror': 150,
'set_fg_color3+rot90': 151,
'set_fg_color3+tophalf': 152,
'set_fg_color3+bottomhalf': 153,
'set_fg_color3+lefthalf': 154,
'set_fg_color3+righthalf': 155,
'set_fg_color3+symmetrize_left_around_vertical': 156,
'set_fg_color3+symmetrize_right_around_vertical': 157,
'set_fg_color3+symmetrize_top_around_horizontal': 158,
'set_fg_color3+symmetrize_bottom_around_horizontal': 159,
'set_fg_color3+upscale_horizontal_by_two': 160,
'set_fg_color3+upscale_vertical_by_two': 161,
'set_fg_color3+gravitate_right': 162,
'set_fg_color3+gravitate_left': 163,
'set_fg_color3+gravitate_up': 164,
'set_fg_color3+gravitate_down': 165,
'set_fg_color3+gravitate_left_right': 166,
'set_fg_color3+gravitate_top_down': 167,
'set_fg_color3+topthird': 168,
'set_fg_color3+vcenterthird': 169,
'set_fg_color3+bottomthird': 170,
'set_fg_color3+leftthird': 171,
'set_fg_color3+hcenterthird': 172,
'set_fg_color3+rightthird': 173,
'set_fg_color4+shift_left': 174,
'set_fg_color4+shift_right': 175,
'set_fg_color4+shift_up': 176,
'set_fg_color4+shift_down': 177,
'set_fg_color4+vmirror': 178,
'set_fg_color4+hmirror': 179,
'set_fg_color4+rot90': 180,
'set_fg_color4+tophalf': 181,
'set_fg_color4+bottomhalf': 182,
'set_fg_color4+lefthalf': 183,
'set_fg_color4+righthalf': 184,
'set_fg_color4+symmetrize_left_around_vertical': 185,
'set_fg_color4+symmetrize_right_around_vertical': 186,
'set_fg_color4+symmetrize_top_around_horizontal': 187,
'set_fg_color4+symmetrize_bottom_around_horizontal': 188,
'set_fg_color4+upscale_horizontal_by_two': 189,
'set_fg_color4+upscale_vertical_by_two': 190,
'set_fg_color4+gravitate_right': 191,
'set_fg_color4+gravitate_left': 192,
'set_fg_color4+gravitate_up': 193,
'set_fg_color4+gravitate_down': 194,
'set_fg_color4+gravitate_left_right': 195,
'set_fg_color4+gravitate_top_down': 196,
'set_fg_color4+topthird': 197,
'set_fg_color4+vcenterthird': 198,
'set_fg_color4+bottomthird': 199,
'set_fg_color4+leftthird': 200,
'set_fg_color4+hcenterthird': 201,
'set_fg_color4+rightthird': 202,
'set_fg_color5+shift_left': 203,
'set_fg_color5+shift_right': 204,
'set_fg_color5+shift_up': 205,
'set_fg_color5+shift_down': 206,
'set_fg_color5+vmirror': 207,
'set_fg_color5+hmirror': 208,
'set_fg_color5+rot90': 209,
'set_fg_color5+tophalf': 210,
'set_fg_color5+bottomhalf': 211,
'set_fg_color5+lefthalf': 212,
'set_fg_color5+righthalf': 213,
'set_fg_color5+symmetrize_left_around_vertical': 214,
'set_fg_color5+symmetrize_right_around_vertical': 215,
'set_fg_color5+symmetrize_top_around_horizontal': 216,
'set_fg_color5+symmetrize_bottom_around_horizontal': 217,
'set_fg_color5+upscale_horizontal_by_two': 218,
'set_fg_color5+upscale_vertical_by_two': 219,
'set_fg_color5+gravitate_right': 220,
'set_fg_color5+gravitate_left': 221,
'set_fg_color5+gravitate_up': 222,
'set_fg_color5+gravitate_down': 223,
'set_fg_color5+gravitate_left_right': 224,
'set_fg_color5+gravitate_top_down': 225,
'set_fg_color5+topthird': 226,
'set_fg_color5+vcenterthird': 227,
'set_fg_color5+bottomthird': 228,
'set_fg_color5+leftthird': 229,
'set_fg_color5+hcenterthird': 230,
'set_fg_color5+rightthird': 231,
'set_fg_color6+shift_left': 232,
'set_fg_color6+shift_right': 233,
'set_fg_color6+shift_up': 234,
'set_fg_color6+shift_down': 235,
'set_fg_color6+vmirror': 236,
'set_fg_color6+hmirror': 237,
'set_fg_color6+rot90': 238,
'set_fg_color6+tophalf': 239,
'set_fg_color6+bottomhalf': 240,
'set_fg_color6+lefthalf': 241,
'set_fg_color6+righthalf': 242,
'set_fg_color6+symmetrize_left_around_vertical': 243,
'set_fg_color6+symmetrize_right_around_vertical': 244,
'set_fg_color6+symmetrize_top_around_horizontal': 245,
'set_fg_color6+symmetrize_bottom_around_horizontal': 246,
'set_fg_color6+upscale_horizontal_by_two': 247,
'set_fg_color6+upscale_vertical_by_two': 248,
'set_fg_color6+gravitate_right': 249,
'set_fg_color6+gravitate_left': 250,
'set_fg_color6+gravitate_up': 251,
'set_fg_color6+gravitate_down': 252,
'set_fg_color6+gravitate_left_right': 253,
'set_fg_color6+gravitate_top_down': 254,
'set_fg_color6+topthird': 255,
'set_fg_color6+vcenterthird': 256,
'set_fg_color6+bottomthird': 257,
'set_fg_color6+leftthird': 258,
'set_fg_color6+hcenterthird': 259,
'set_fg_color6+rightthird': 260,
'set_fg_color7+shift_left': 261,
'set_fg_color7+shift_right': 262,
'set_fg_color7+shift_up': 263,
'set_fg_color7+shift_down': 264,
'set_fg_color7+vmirror': 265,
'set_fg_color7+hmirror': 266,
'set_fg_color7+rot90': 267,
'set_fg_color7+tophalf': 268,
'set_fg_color7+bottomhalf': 269,
'set_fg_color7+lefthalf': 270,
'set_fg_color7+righthalf': 271,
'set_fg_color7+symmetrize_left_around_vertical': 272,
'set_fg_color7+symmetrize_right_around_vertical': 273,
'set_fg_color7+symmetrize_top_around_horizontal': 274,
'set_fg_color7+symmetrize_bottom_around_horizontal': 275,
'set_fg_color7+upscale_horizontal_by_two': 276,
'set_fg_color7+upscale_vertical_by_two': 277,
'set_fg_color7+gravitate_right': 278,
'set_fg_color7+gravitate_left': 279,
'set_fg_color7+gravitate_up': 280,
'set_fg_color7+gravitate_down': 281,
'set_fg_color7+gravitate_left_right': 282,
'set_fg_color7+gravitate_top_down': 283,
'set_fg_color7+topthird': 284,
'set_fg_color7+vcenterthird': 285,
'set_fg_color7+bottomthird': 286,
'set_fg_color7+leftthird': 287,
'set_fg_color7+hcenterthird': 288,
'set_fg_color7+rightthird': 289,
'set_fg_color8+shift_left': 290,
'set_fg_color8+shift_right': 291,
'set_fg_color8+shift_up': 292,
'set_fg_color8+shift_down': 293,
'set_fg_color8+vmirror': 294,
'set_fg_color8+hmirror': 295,
'set_fg_color8+rot90': 296,
'set_fg_color8+tophalf': 297,
'set_fg_color8+bottomhalf': 298,
'set_fg_color8+lefthalf': 299,
'set_fg_color8+righthalf': 300,
'set_fg_color8+symmetrize_left_around_vertical': 301,
'set_fg_color8+symmetrize_right_around_vertical': 302,
'set_fg_color8+symmetrize_top_around_horizontal': 303,
'set_fg_color8+symmetrize_bottom_around_horizontal': 304,
'set_fg_color8+upscale_horizontal_by_two': 305,
'set_fg_color8+upscale_vertical_by_two': 306,
'set_fg_color8+gravitate_right': 307,
'set_fg_color8+gravitate_left': 308,
'set_fg_color8+gravitate_up': 309,
'set_fg_color8+gravitate_down': 310,
'set_fg_color8+gravitate_left_right': 311,
'set_fg_color8+gravitate_top_down': 312,
'set_fg_color8+topthird': 313,
'set_fg_color8+vcenterthird': 314,
'set_fg_color8+bottomthird': 315,
'set_fg_color8+leftthird': 316,
'set_fg_color8+hcenterthird': 317,
'set_fg_color8+rightthird': 318,
'set_fg_color9+shift_left': 319,
'set_fg_color9+shift_right': 320,
'set_fg_color9+shift_up': 321,
'set_fg_color9+shift_down': 322,
'set_fg_color9+vmirror': 323,
'set_fg_color9+hmirror': 324,
'set_fg_color9+rot90': 325,
'set_fg_color9+tophalf': 326,
'set_fg_color9+bottomhalf': 327,
'set_fg_color9+lefthalf': 328,
'set_fg_color9+righthalf': 329,
'set_fg_color9+symmetrize_left_around_vertical': 330,
'set_fg_color9+symmetrize_right_around_vertical': 331,
'set_fg_color9+symmetrize_top_around_horizontal': 332,
'set_fg_color9+symmetrize_bottom_around_horizontal': 333,
'set_fg_color9+upscale_horizontal_by_two': 334,
'set_fg_color9+upscale_vertical_by_two': 335,
'set_fg_color9+gravitate_right': 336,
'set_fg_color9+gravitate_left': 337,
'set_fg_color9+gravitate_up': 338,
'set_fg_color9+gravitate_down': 339,
'set_fg_color9+gravitate_left_right': 340,
'set_fg_color9+gravitate_top_down': 341,
'set_fg_color9+topthird': 342,
'set_fg_color9+vcenterthird': 343,
'set_fg_color9+bottomthird': 344,
'set_fg_color9+leftthird': 345,
'set_fg_color9+hcenterthird': 346,
'set_fg_color9+rightthird': 347,
'shift_left+shift_left': 348,
'shift_left+shift_up': 349,
'shift_left+shift_down': 350,
'shift_left+vmirror': 351,
'shift_left+hmirror': 352,
'shift_left+rot90': 353,
'shift_left+tophalf': 354,
'shift_left+bottomhalf': 355,
'shift_left+lefthalf': 356,
'shift_left+righthalf': 357,
'shift_left+symmetrize_left_around_vertical': 358,
'shift_left+symmetrize_right_around_vertical': 359,
'shift_left+symmetrize_top_around_horizontal': 360,
'shift_left+symmetrize_bottom_around_horizontal': 361,
'shift_left+upscale_horizontal_by_two': 362,
'shift_left+upscale_vertical_by_two': 363,
'shift_left+gravitate_right': 364,
'shift_left+gravitate_left': 365,
'shift_left+gravitate_up': 366,
'shift_left+gravitate_down': 367,
'shift_left+gravitate_left_right': 368,
'shift_left+gravitate_top_down': 369,
'shift_left+topthird': 370,
'shift_left+vcenterthird': 371,
'shift_left+bottomthird': 372,
'shift_left+leftthird': 373,
'shift_left+hcenterthird': 374,
'shift_left+rightthird': 375,
'shift_right+shift_right': 376,
'shift_right+shift_up': 377,
'shift_right+shift_down': 378,
'shift_right+vmirror': 379,
'shift_right+hmirror': 380,
'shift_right+rot90': 381,
'shift_right+tophalf': 382,
'shift_right+bottomhalf': 383,
'shift_right+lefthalf': 384,
'shift_right+righthalf': 385,
'shift_right+symmetrize_left_around_vertical': 386,
'shift_right+symmetrize_right_around_vertical': 387,
'shift_right+symmetrize_top_around_horizontal': 388,
'shift_right+symmetrize_bottom_around_horizontal': 389,
'shift_right+upscale_horizontal_by_two': 390,
'shift_right+upscale_vertical_by_two': 391,
'shift_right+gravitate_right': 392,
'shift_right+gravitate_left': 393,
'shift_right+gravitate_up': 394,
'shift_right+gravitate_down': 395,
'shift_right+gravitate_left_right': 396,
'shift_right+gravitate_top_down': 397,
'shift_right+topthird': 398,
'shift_right+vcenterthird': 399,
'shift_right+bottomthird': 400,
'shift_right+leftthird': 401,
'shift_right+hcenterthird': 402,
'shift_right+rightthird': 403,
'shift_up+shift_up': 404,
'shift_up+vmirror': 405,
'shift_up+hmirror': 406,
'shift_up+rot90': 407,
'shift_up+tophalf': 408,
'shift_up+bottomhalf': 409,
'shift_up+lefthalf': 410,
'shift_up+righthalf': 411,
'shift_up+symmetrize_left_around_vertical': 412,
'shift_up+symmetrize_right_around_vertical': 413,
'shift_up+symmetrize_top_around_horizontal': 414,
'shift_up+symmetrize_bottom_around_horizontal': 415,
'shift_up+upscale_horizontal_by_two': 416,
'shift_up+upscale_vertical_by_two': 417,
'shift_up+gravitate_right': 418,
'shift_up+gravitate_left': 419,
'shift_up+gravitate_up': 420,
'shift_up+gravitate_down': 421,
'shift_up+gravitate_left_right': 422,
'shift_up+gravitate_top_down': 423,
'shift_up+topthird': 424,
'shift_up+vcenterthird': 425,
'shift_up+bottomthird': 426,
'shift_up+leftthird': 427,
'shift_up+hcenterthird': 428,
'shift_up+rightthird': 429,
'shift_down+shift_down': 430,
'shift_down+vmirror': 431,
'shift_down+hmirror': 432,
'shift_down+rot90': 433,
'shift_down+tophalf': 434,
'shift_down+bottomhalf': 435,
'shift_down+lefthalf': 436,
'shift_down+righthalf': 437,
'shift_down+symmetrize_left_around_vertical': 438,
'shift_down+symmetrize_right_around_vertical': 439,
'shift_down+symmetrize_top_around_horizontal': 440,
'shift_down+symmetrize_bottom_around_horizontal': 441,
'shift_down+upscale_horizontal_by_two': 442,
'shift_down+upscale_vertical_by_two': 443,
'shift_down+gravitate_right': 444,
'shift_down+gravitate_left': 445,
'shift_down+gravitate_up': 446,
'shift_down+gravitate_down': 447,
'shift_down+gravitate_left_right': 448,
'shift_down+gravitate_top_down': 449,
'shift_down+topthird': 450,
'shift_down+vcenterthird': 451,
'shift_down+bottomthird': 452,
'shift_down+leftthird': 453,
'shift_down+hcenterthird': 454,
'shift_down+rightthird': 455,
'vmirror+hmirror': 456,
'vmirror+tophalf': 457,
'vmirror+bottomhalf': 458,
'vmirror+lefthalf': 459,
'vmirror+righthalf': 460,
'vmirror+symmetrize_top_around_horizontal': 461,
'vmirror+symmetrize_bottom_around_horizontal': 462,
'vmirror+upscale_horizontal_by_two': 463,
'vmirror+upscale_vertical_by_two': 464,
'vmirror+gravitate_right': 465,
'vmirror+gravitate_left': 466,
'vmirror+gravitate_up': 467,
'vmirror+gravitate_down': 468,
'vmirror+gravitate_left_right': 469,
'vmirror+gravitate_top_down': 470,
'vmirror+topthird': 471,
'vmirror+vcenterthird': 472,
'vmirror+bottomthird': 473,
'vmirror+leftthird': 474,
'vmirror+hcenterthird': 475,
'vmirror+rightthird': 476,
'hmirror+tophalf': 477,
'hmirror+bottomhalf': 478,
'hmirror+lefthalf': 479,
'hmirror+righthalf': 480,
'hmirror+symmetrize_left_around_vertical': 481,
'hmirror+symmetrize_right_around_vertical': 482,
'hmirror+upscale_horizontal_by_two': 483,
'hmirror+upscale_vertical_by_two': 484,
'hmirror+gravitate_right': 485,
'hmirror+gravitate_left': 486,
'hmirror+gravitate_up': 487,
'hmirror+gravitate_down': 488,
'hmirror+gravitate_left_right': 489,
'hmirror+gravitate_top_down': 490,
'hmirror+topthird': 491,
'hmirror+vcenterthird': 492,
'hmirror+bottomthird': 493,
'hmirror+leftthird': 494,
'hmirror+hcenterthird': 495,
'hmirror+rightthird': 496,
'rot90+tophalf': 497,
'rot90+bottomhalf': 498,
'rot90+lefthalf': 499,
'rot90+righthalf': 500,
'rot90+upscale_horizontal_by_two': 501,
'rot90+upscale_vertical_by_two': 502,
'rot90+gravitate_right': 503,
'rot90+gravitate_left': 504,
'rot90+gravitate_up': 505,
'rot90+gravitate_down': 506,
'rot90+gravitate_left_right': 507,
'rot90+gravitate_top_down': 508,
'rot90+topthird': 509,
'rot90+vcenterthird': 510,
'rot90+bottomthird': 511,
'rot90+leftthird': 512,
'rot90+hcenterthird': 513,
'rot90+rightthird': 514,
'tophalf+shift_up': 515,
'tophalf+shift_down': 516,
'tophalf+tophalf': 517,
'tophalf+bottomhalf': 518,
'tophalf+lefthalf': 519,
'tophalf+righthalf': 520,
'tophalf+symmetrize_left_around_vertical': 521,
'tophalf+symmetrize_right_around_vertical': 522,
'tophalf+symmetrize_top_around_horizontal': 523,
'tophalf+symmetrize_bottom_around_horizontal': 524,
'tophalf+upscale_horizontal_by_two': 525,
'tophalf+upscale_vertical_by_two': 526,
'tophalf+gravitate_right': 527,
'tophalf+gravitate_left': 528,
'tophalf+gravitate_up': 529,
'tophalf+gravitate_down': 530,
'tophalf+gravitate_left_right': 531,
'tophalf+gravitate_top_down': 532,
'tophalf+topthird': 533,
'tophalf+vcenterthird': 534,
'tophalf+bottomthird': 535,
'tophalf+leftthird': 536,
'tophalf+hcenterthird': 537,
'tophalf+rightthird': 538,
'bottomhalf+shift_up': 539,
'bottomhalf+shift_down': 540,
'bottomhalf+tophalf': 541,
'bottomhalf+bottomhalf': 542,
'bottomhalf+lefthalf': 543,
'bottomhalf+righthalf': 544,
'bottomhalf+symmetrize_left_around_vertical': 545,
'bottomhalf+symmetrize_right_around_vertical': 546,
'bottomhalf+symmetrize_top_around_horizontal': 547,
'bottomhalf+symmetrize_bottom_around_horizontal': 548,
'bottomhalf+upscale_horizontal_by_two': 549,
'bottomhalf+upscale_vertical_by_two': 550,
'bottomhalf+gravitate_right': 551,
'bottomhalf+gravitate_left': 552,
'bottomhalf+gravitate_up': 553,
'bottomhalf+gravitate_down': 554,
'bottomhalf+gravitate_left_right': 555,
'bottomhalf+gravitate_top_down': 556,
'bottomhalf+topthird': 557,
'bottomhalf+vcenterthird': 558,
'bottomhalf+bottomthird': 559,
'bottomhalf+leftthird': 560,
'bottomhalf+hcenterthird': 561,
'bottomhalf+rightthird': 562,
'lefthalf+shift_left': 563,
'lefthalf+shift_right': 564,
'lefthalf+lefthalf': 565,
'lefthalf+righthalf': 566,
'lefthalf+symmetrize_left_around_vertical': 567,
'lefthalf+symmetrize_right_around_vertical': 568,
'lefthalf+symmetrize_top_around_horizontal': 569,
'lefthalf+symmetrize_bottom_around_horizontal': 570,
'lefthalf+upscale_horizontal_by_two': 571,
'lefthalf+upscale_vertical_by_two': 572,
'lefthalf+gravitate_right': 573,
'lefthalf+gravitate_left': 574,
'lefthalf+gravitate_up': 575,
'lefthalf+gravitate_down': 576,
'lefthalf+gravitate_left_right': 577,
'lefthalf+gravitate_top_down': 578,
'lefthalf+topthird': 579,
'lefthalf+vcenterthird': 580,
'lefthalf+bottomthird': 581,
'lefthalf+leftthird': 582,
'lefthalf+hcenterthird': 583,
'lefthalf+rightthird': 584,
'righthalf+shift_left': 585,
'righthalf+shift_right': 586,
'righthalf+lefthalf': 587,
'righthalf+righthalf': 588,
'righthalf+symmetrize_left_around_vertical': 589,
'righthalf+symmetrize_right_around_vertical': 590,
'righthalf+symmetrize_top_around_horizontal': 591,
'righthalf+symmetrize_bottom_around_horizontal': 592,
'righthalf+upscale_horizontal_by_two': 593,
'righthalf+upscale_vertical_by_two': 594,
'righthalf+gravitate_right': 595,
'righthalf+gravitate_left': 596,
'righthalf+gravitate_up': 597,
'righthalf+gravitate_down': 598,
'righthalf+gravitate_left_right': 599,
'righthalf+gravitate_top_down': 600,
'righthalf+topthird': 601,
'righthalf+vcenterthird': 602,
'righthalf+bottomthird': 603,
'righthalf+leftthird': 604,
'righthalf+hcenterthird': 605,
'righthalf+rightthird': 606,
'symmetrize_left_around_vertical+shift_left': 607,
'symmetrize_left_around_vertical+shift_right': 608,
'symmetrize_left_around_vertical+upscale_horizontal_by_two': 609,
'symmetrize_left_around_vertical+upscale_vertical_by_two': 610,
'symmetrize_left_around_vertical+topthird': 611,
'symmetrize_left_around_vertical+vcenterthird': 612,
'symmetrize_left_around_vertical+bottomthird': 613,
'symmetrize_left_around_vertical+leftthird': 614,
'symmetrize_left_around_vertical+hcenterthird': 615,
'symmetrize_left_around_vertical+rightthird': 616,
'symmetrize_right_around_vertical+shift_left': 617,
'symmetrize_right_around_vertical+shift_right': 618,
'symmetrize_right_around_vertical+symmetrize_bottom_around_horizontal': 619,
'symmetrize_right_around_vertical+upscale_horizontal_by_two': 620,
'symmetrize_right_around_vertical+upscale_vertical_by_two': 621,
'symmetrize_right_around_vertical+topthird': 622,
'symmetrize_right_around_vertical+vcenterthird': 623,
'symmetrize_right_around_vertical+bottomthird': 624,
'symmetrize_right_around_vertical+leftthird': 625,
'symmetrize_right_around_vertical+hcenterthird': 626,
'symmetrize_right_around_vertical+rightthird': 627,
'symmetrize_top_around_horizontal+shift_up': 628,
'symmetrize_top_around_horizontal+shift_down': 629,
'symmetrize_top_around_horizontal+upscale_horizontal_by_two': 630,
'symmetrize_top_around_horizontal+upscale_vertical_by_two': 631,
'symmetrize_top_around_horizontal+topthird': 632,
'symmetrize_top_around_horizontal+vcenterthird': 633,
'symmetrize_top_around_horizontal+bottomthird': 634,
'symmetrize_top_around_horizontal+leftthird': 635,
'symmetrize_top_around_horizontal+hcenterthird': 636,
'symmetrize_top_around_horizontal+rightthird': 637,
'symmetrize_bottom_around_horizontal+shift_up': 638,
'symmetrize_bottom_around_horizontal+shift_down': 639,
'symmetrize_bottom_around_horizontal+upscale_horizontal_by_two': 640,
'symmetrize_bottom_around_horizontal+upscale_vertical_by_two': 641,
'symmetrize_bottom_around_horizontal+topthird': 642,
'symmetrize_bottom_around_horizontal+vcenterthird': 643,
'symmetrize_bottom_around_horizontal+bottomthird': 644,
'symmetrize_bottom_around_horizontal+leftthird': 645,
'symmetrize_bottom_around_horizontal+hcenterthird': 646,
'symmetrize_bottom_around_horizontal+rightthird': 647,
'upscale_horizontal_by_two+shift_left': 648,
'upscale_horizontal_by_two+shift_right': 649,
'upscale_horizontal_by_two+lefthalf': 650,
'upscale_horizontal_by_two+righthalf': 651,
'upscale_horizontal_by_two+symmetrize_left_around_vertical': 652,
'upscale_horizontal_by_two+symmetrize_right_around_vertical': 653,
'upscale_horizontal_by_two+upscale_horizontal_by_two': 654,
'upscale_horizontal_by_two+upscale_vertical_by_two': 655,
'upscale_horizontal_by_two+gravitate_right': 656,
'upscale_horizontal_by_two+gravitate_left': 657,
'upscale_horizontal_by_two+gravitate_up': 658,
'upscale_horizontal_by_two+gravitate_down': 659,
'upscale_horizontal_by_two+gravitate_left_right': 660,
'upscale_horizontal_by_two+gravitate_top_down': 661,
'upscale_horizontal_by_two+topthird': 662,
'upscale_horizontal_by_two+vcenterthird': 663,
'upscale_horizontal_by_two+bottomthird': 664,
'upscale_horizontal_by_two+leftthird': 665,
'upscale_horizontal_by_two+hcenterthird': 666,
'upscale_horizontal_by_two+rightthird': 667,
'upscale_vertical_by_two+shift_up': 668,
'upscale_vertical_by_two+shift_down': 669,
'upscale_vertical_by_two+tophalf': 670,
'upscale_vertical_by_two+bottomhalf': 671,
'upscale_vertical_by_two+symmetrize_top_around_horizontal': 672,
'upscale_vertical_by_two+symmetrize_bottom_around_horizontal': 673,
'upscale_vertical_by_two+upscale_vertical_by_two': 674,
'upscale_vertical_by_two+gravitate_right': 675,
'upscale_vertical_by_two+gravitate_left': 676,
'upscale_vertical_by_two+gravitate_up': 677,
'upscale_vertical_by_two+gravitate_down': 678,
'upscale_vertical_by_two+gravitate_left_right': 679,
'upscale_vertical_by_two+gravitate_top_down': 680,
'upscale_vertical_by_two+topthird': 681,
'upscale_vertical_by_two+vcenterthird': 682,
'upscale_vertical_by_two+bottomthird': 683,
'upscale_vertical_by_two+leftthird': 684,
'upscale_vertical_by_two+hcenterthird': 685,
'upscale_vertical_by_two+rightthird': 686,
'gravitate_right+shift_left': 687,
'gravitate_right+shift_right': 688,
'gravitate_right+lefthalf': 689,
'gravitate_right+righthalf': 690,
'gravitate_left+shift_left': 691,
'gravitate_left+shift_right': 692,
'gravitate_left+lefthalf': 693,
'gravitate_left+righthalf': 694,
'gravitate_up+shift_up': 695,
'gravitate_up+shift_down': 696,
'gravitate_up+tophalf': 697,
'gravitate_up+bottomhalf': 698,
'gravitate_down+shift_up': 699,
'gravitate_down+shift_down': 700,
'gravitate_down+tophalf': 701,
'gravitate_down+bottomhalf': 702,
'gravitate_left_right+shift_left': 703,
'gravitate_left_right+shift_right': 704,
'gravitate_left_right+upscale_horizontal_by_two': 705,
'gravitate_left_right+gravitate_top_down': 706,
'gravitate_top_down+shift_up': 707,
'gravitate_top_down+shift_down': 708,
'gravitate_top_down+upscale_vertical_by_two': 709,
'topthird+shift_up': 710,
'topthird+shift_down': 711,
'topthird+bottomhalf': 712,
'topthird+symmetrize_top_around_horizontal': 713,
'topthird+symmetrize_bottom_around_horizontal': 714,
'topthird+upscale_vertical_by_two': 715,
'topthird+topthird': 716,
'topthird+vcenterthird': 717,
'topthird+bottomthird': 718,
'topthird+leftthird': 719,
'topthird+hcenterthird': 720,
'topthird+rightthird': 721,
'vcenterthird+shift_up': 722,
'vcenterthird+shift_down': 723,
'vcenterthird+hmirror': 724,
'vcenterthird+rot90': 725,
'vcenterthird+tophalf': 726,
'vcenterthird+bottomhalf': 727,
'vcenterthird+symmetrize_top_around_horizontal': 728,
'vcenterthird+symmetrize_bottom_around_horizontal': 729,
'vcenterthird+topthird': 730,
'vcenterthird+vcenterthird': 731,
'vcenterthird+bottomthird': 732,
'vcenterthird+leftthird': 733,
'vcenterthird+hcenterthird': 734,
'vcenterthird+rightthird': 735,
'bottomthird+shift_up': 736,
'bottomthird+shift_down': 737,
'bottomthird+tophalf': 738,
'bottomthird+symmetrize_top_around_horizontal': 739,
'bottomthird+symmetrize_bottom_around_horizontal': 740,
'bottomthird+upscale_vertical_by_two': 741,
'bottomthird+topthird': 742,
'bottomthird+vcenterthird': 743,
'bottomthird+bottomthird': 744,
'bottomthird+leftthird': 745,
'bottomthird+hcenterthird': 746,
'bottomthird+rightthird': 747,
'leftthird+shift_left': 748,
'leftthird+shift_right': 749,
'leftthird+righthalf': 750,
'leftthird+symmetrize_left_around_vertical': 751,
'leftthird+symmetrize_right_around_vertical': 752,
'leftthird+upscale_horizontal_by_two': 753,
'leftthird+leftthird': 754,
'leftthird+hcenterthird': 755,
'leftthird+rightthird': 756,
'hcenterthird+shift_left': 757,
'hcenterthird+shift_right': 758,
'hcenterthird+vmirror': 759,
'hcenterthird+lefthalf': 760,
'hcenterthird+righthalf': 761,
'hcenterthird+symmetrize_left_around_vertical': 762,
'hcenterthird+symmetrize_right_around_vertical': 763,
'hcenterthird+upscale_horizontal_by_two': 764,
'hcenterthird+leftthird': 765,
'hcenterthird+hcenterthird': 766,
'hcenterthird+rightthird': 767,
'rightthird+shift_left': 768,
'rightthird+shift_right': 769,
'rightthird+lefthalf': 770,
'rightthird+symmetrize_left_around_vertical': 771,
'rightthird+symmetrize_right_around_vertical': 772,
'rightthird+upscale_horizontal_by_two': 773,
'rightthird+leftthird': 774,
'rightthird+hcenterthird': 775,
'rightthird+rightthird': 776,
'bottomhalf+rot270': 777,
'tophalf+rot270': 778,
'lefthalf+rot270': 779,
'righthalf+rot270': 780,
'bottomhalf+rot180': 781,
'tophalf+rot180': 782,
'lefthalf+rot180': 783,
'righthalf+rot180': 784,
'upscale_by_three': 785,
'rot180': 786,
'rot270': 787,
'duplicate_top_row': 788,
'duplicate_bottom_row': 789,
'duplicate_left_column': 790,
'duplicate_right_column': 791,
'duplicate_top_row+duplicate_bottom_row': 792,
'duplicate_left_column+duplicate_right_column': 793,
'duplicate_top_row+duplicate_bottom_row+duplicate_left_column+duplicate_right_column': 794,
'compress': 795
}

semantics = {
'set_fg_color1': lambda g: set_fg_color(g, 1),
'set_fg_color2': lambda g: set_fg_color(g, 2),
'set_fg_color3': lambda g: set_fg_color(g, 3),
'set_fg_color4': lambda g: set_fg_color(g, 4),
'set_fg_color5': lambda g: set_fg_color(g, 5),
'set_fg_color6': lambda g: set_fg_color(g, 6),
'set_fg_color7': lambda g: set_fg_color(g, 7),
'set_fg_color8': lambda g: set_fg_color(g, 8),
'set_fg_color9': lambda g: set_fg_color(g, 9),
'shift_left': lambda g: shift_left(g),
'shift_right': lambda g: shift_right(g),
'shift_up': lambda g: shift_up(g),
'shift_down': lambda g: shift_down(g),
'vmirror': lambda g: vmirror(g),
'hmirror': lambda g: hmirror(g),
'rot90': lambda g: rot90(g),
'tophalf': lambda g: tophalf(g),
'bottomhalf': lambda g: bottomhalf(g),
'lefthalf': lambda g: lefthalf(g),
'righthalf': lambda g: righthalf(g),
'symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(g),
'symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(g),
'symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(g),
'symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(g),
'upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(g),
'upscale_vertical_by_two': lambda g: upscale_vertical_by_two(g),
'upscale_by_two': lambda g: upscale_by_two(g),
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
'color_change': lambda g: lambda c1: lambda c2: color_change(g, c1, c2),
'invert_colors': lambda g: invert_colors(g),
'shift_left+color_change': lambda g: lambda c1: lambda c2: color_change(shift_left(g), c1, c2),
'shift_right+color_change': lambda g: lambda c1: lambda c2: color_change(shift_right(g), c1, c2),
'shift_up+color_change': lambda g: lambda c1: lambda c2: color_change(shift_up(g), c1, c2),
'shift_down+color_change': lambda g: lambda c1: lambda c2: color_change(shift_down(g), c1, c2),
'vmirror+color_change': lambda g: lambda c1: lambda c2: color_change(vmirror(g), c1, c2),
'hmirror+color_change': lambda g: lambda c1: lambda c2: color_change(hmirror(g), c1, c2),
'rot90+color_change': lambda g: lambda c1: lambda c2: color_change(rot90(g), c1, c2),
'tophalf+color_change': lambda g: lambda c1: lambda c2: color_change(tophalf(g), c1, c2),
'bottomhalf+color_change': lambda g: lambda c1: lambda c2: color_change(bottomhalf(g), c1, c2),
'lefthalf+color_change': lambda g: lambda c1: lambda c2: color_change(lefthalf(g), c1, c2),
'righthalf+color_change': lambda g: lambda c1: lambda c2: color_change(righthalf(g), c1, c2),
'symmetrize_left_around_vertical+color_change': lambda g: lambda c1: lambda c2: color_change(symmetrize_left_around_vertical(g), c1, c2),
'symmetrize_right_around_vertical+color_change': lambda g: lambda c1: lambda c2: color_change(symmetrize_right_around_vertical(g), c1, c2),
'symmetrize_top_around_horizontal+color_change': lambda g: lambda c1: lambda c2: color_change(symmetrize_top_around_horizontal(g), c1, c2),
'symmetrize_bottom_around_horizontal+color_change': lambda g: lambda c1: lambda c2: color_change(symmetrize_bottom_around_horizontal(g), c1, c2),
'upscale_horizontal_by_two+color_change': lambda g: lambda c1: lambda c2: color_change(upscale_horizontal_by_two(g), c1, c2),
'upscale_vertical_by_two+color_change': lambda g: lambda c1: lambda c2: color_change(upscale_vertical_by_two(g), c1, c2),
'gravitate_right+color_change': lambda g: lambda c1: lambda c2: color_change(gravitate_right(g), c1, c2),
'gravitate_left+color_change': lambda g: lambda c1: lambda c2: color_change(gravitate_left(g), c1, c2),
'gravitate_up+color_change': lambda g: lambda c1: lambda c2: color_change(gravitate_up(g), c1, c2),
'gravitate_down+color_change': lambda g: lambda c1: lambda c2: color_change(gravitate_down(g), c1, c2),
'gravitate_left_right+color_change': lambda g: lambda c1: lambda c2: color_change(gravitate_left_right(g), c1, c2),
'gravitate_top_down+color_change': lambda g: lambda c1: lambda c2: color_change(gravitate_top_down(g), c1, c2),
'shift_left+invert_colors': lambda g: invert_colors(shift_left(g)),
'shift_right+invert_colors': lambda g: invert_colors(shift_right(g)),
'shift_up+invert_colors': lambda g: invert_colors(shift_up(g)),
'shift_down+invert_colors': lambda g: invert_colors(shift_down(g)),
'vmirror+invert_colors': lambda g: invert_colors(vmirror(g)),
'hmirror+invert_colors': lambda g: invert_colors(hmirror(g)),
'rot90+invert_colors': lambda g: invert_colors(rot90(g)),
'tophalf+invert_colors': lambda g: invert_colors(tophalf(g)),
'bottomhalf+invert_colors': lambda g: invert_colors(bottomhalf(g)),
'lefthalf+invert_colors': lambda g: invert_colors(lefthalf(g)),
'righthalf+invert_colors': lambda g: invert_colors(righthalf(g)),
'symmetrize_left_around_vertical+invert_colors': lambda g: invert_colors(symmetrize_left_around_vertical(g)),
'symmetrize_right_around_vertical+invert_colors': lambda g: invert_colors(symmetrize_right_around_vertical(g)),
'symmetrize_top_around_horizontal+invert_colors': lambda g: invert_colors(symmetrize_top_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+invert_colors': lambda g: invert_colors(symmetrize_bottom_around_horizontal(g)),
'upscale_horizontal_by_two+invert_colors': lambda g: invert_colors(upscale_horizontal_by_two(g)),
'upscale_vertical_by_two+invert_colors': lambda g: invert_colors(upscale_vertical_by_two(g)),
'gravitate_right+invert_colors': lambda g: invert_colors(gravitate_right(g)),
'gravitate_left+invert_colors': lambda g: invert_colors(gravitate_left(g)),
'gravitate_up+invert_colors': lambda g: invert_colors(gravitate_up(g)),
'gravitate_down+invert_colors': lambda g: invert_colors(gravitate_down(g)),
'gravitate_left_right+invert_colors': lambda g: invert_colors(gravitate_left_right(g)),
'gravitate_top_down+invert_colors': lambda g: invert_colors(gravitate_top_down(g)),
'set_fg_color1+shift_left': lambda g: shift_left(set_fg_color(g, 1)),
'set_fg_color1+shift_right': lambda g: shift_right(set_fg_color(g, 1)),
'set_fg_color1+shift_up': lambda g: shift_up(set_fg_color(g, 1)),
'set_fg_color1+shift_down': lambda g: shift_down(set_fg_color(g, 1)),
'set_fg_color1+vmirror': lambda g: vmirror(set_fg_color(g, 1)),
'set_fg_color1+hmirror': lambda g: hmirror(set_fg_color(g, 1)),
'set_fg_color1+rot90': lambda g: rot90(set_fg_color(g, 1)),
'set_fg_color1+tophalf': lambda g: tophalf(set_fg_color(g, 1)),
'set_fg_color1+bottomhalf': lambda g: bottomhalf(set_fg_color(g, 1)),
'set_fg_color1+lefthalf': lambda g: lefthalf(set_fg_color(g, 1)),
'set_fg_color1+righthalf': lambda g: righthalf(set_fg_color(g, 1)),
'set_fg_color1+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(set_fg_color(g, 1)),
'set_fg_color1+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(set_fg_color(g, 1)),
'set_fg_color1+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(set_fg_color(g, 1)),
'set_fg_color1+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(set_fg_color(g, 1)),
'set_fg_color1+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(set_fg_color(g, 1)),
'set_fg_color1+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(set_fg_color(g, 1)),
'set_fg_color1+gravitate_right': lambda g: gravitate_right(set_fg_color(g, 1)),
'set_fg_color1+gravitate_left': lambda g: gravitate_left(set_fg_color(g, 1)),
'set_fg_color1+gravitate_up': lambda g: gravitate_up(set_fg_color(g, 1)),
'set_fg_color1+gravitate_down': lambda g: gravitate_down(set_fg_color(g, 1)),
'set_fg_color1+gravitate_left_right': lambda g: gravitate_left_right(set_fg_color(g, 1)),
'set_fg_color1+gravitate_top_down': lambda g: gravitate_top_down(set_fg_color(g, 1)),
'set_fg_color1+topthird': lambda g: topthird(set_fg_color(g, 1)),
'set_fg_color1+vcenterthird': lambda g: vcenterthird(set_fg_color(g, 1)),
'set_fg_color1+bottomthird': lambda g: bottomthird(set_fg_color(g, 1)),
'set_fg_color1+leftthird': lambda g: leftthird(set_fg_color(g, 1)),
'set_fg_color1+hcenterthird': lambda g: hcenterthird(set_fg_color(g, 1)),
'set_fg_color1+rightthird': lambda g: rightthird(set_fg_color(g, 1)),
'set_fg_color2+shift_left': lambda g: shift_left(set_fg_color(g, 2)),
'set_fg_color2+shift_right': lambda g: shift_right(set_fg_color(g, 2)),
'set_fg_color2+shift_up': lambda g: shift_up(set_fg_color(g, 2)),
'set_fg_color2+shift_down': lambda g: shift_down(set_fg_color(g, 2)),
'set_fg_color2+vmirror': lambda g: vmirror(set_fg_color(g, 2)),
'set_fg_color2+hmirror': lambda g: hmirror(set_fg_color(g, 2)),
'set_fg_color2+rot90': lambda g: rot90(set_fg_color(g, 2)),
'set_fg_color2+tophalf': lambda g: tophalf(set_fg_color(g, 2)),
'set_fg_color2+bottomhalf': lambda g: bottomhalf(set_fg_color(g, 2)),
'set_fg_color2+lefthalf': lambda g: lefthalf(set_fg_color(g, 2)),
'set_fg_color2+righthalf': lambda g: righthalf(set_fg_color(g, 2)),
'set_fg_color2+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(set_fg_color(g, 2)),
'set_fg_color2+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(set_fg_color(g, 2)),
'set_fg_color2+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(set_fg_color(g, 2)),
'set_fg_color2+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(set_fg_color(g, 2)),
'set_fg_color2+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(set_fg_color(g, 2)),
'set_fg_color2+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(set_fg_color(g, 2)),
'set_fg_color2+gravitate_right': lambda g: gravitate_right(set_fg_color(g, 2)),
'set_fg_color2+gravitate_left': lambda g: gravitate_left(set_fg_color(g, 2)),
'set_fg_color2+gravitate_up': lambda g: gravitate_up(set_fg_color(g, 2)),
'set_fg_color2+gravitate_down': lambda g: gravitate_down(set_fg_color(g, 2)),
'set_fg_color2+gravitate_left_right': lambda g: gravitate_left_right(set_fg_color(g, 2)),
'set_fg_color2+gravitate_top_down': lambda g: gravitate_top_down(set_fg_color(g, 2)),
'set_fg_color2+topthird': lambda g: topthird(set_fg_color(g, 2)),
'set_fg_color2+vcenterthird': lambda g: vcenterthird(set_fg_color(g, 2)),
'set_fg_color2+bottomthird': lambda g: bottomthird(set_fg_color(g, 2)),
'set_fg_color2+leftthird': lambda g: leftthird(set_fg_color(g, 2)),
'set_fg_color2+hcenterthird': lambda g: hcenterthird(set_fg_color(g, 2)),
'set_fg_color2+rightthird': lambda g: rightthird(set_fg_color(g, 2)),
'set_fg_color3+shift_left': lambda g: shift_left(set_fg_color(g, 3)),
'set_fg_color3+shift_right': lambda g: shift_right(set_fg_color(g, 3)),
'set_fg_color3+shift_up': lambda g: shift_up(set_fg_color(g, 3)),
'set_fg_color3+shift_down': lambda g: shift_down(set_fg_color(g, 3)),
'set_fg_color3+vmirror': lambda g: vmirror(set_fg_color(g, 3)),
'set_fg_color3+hmirror': lambda g: hmirror(set_fg_color(g, 3)),
'set_fg_color3+rot90': lambda g: rot90(set_fg_color(g, 3)),
'set_fg_color3+tophalf': lambda g: tophalf(set_fg_color(g, 3)),
'set_fg_color3+bottomhalf': lambda g: bottomhalf(set_fg_color(g, 3)),
'set_fg_color3+lefthalf': lambda g: lefthalf(set_fg_color(g, 3)),
'set_fg_color3+righthalf': lambda g: righthalf(set_fg_color(g, 3)),
'set_fg_color3+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(set_fg_color(g, 3)),
'set_fg_color3+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(set_fg_color(g, 3)),
'set_fg_color3+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(set_fg_color(g, 3)),
'set_fg_color3+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(set_fg_color(g, 3)),
'set_fg_color3+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(set_fg_color(g, 3)),
'set_fg_color3+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(set_fg_color(g, 3)),
'set_fg_color3+gravitate_right': lambda g: gravitate_right(set_fg_color(g, 3)),
'set_fg_color3+gravitate_left': lambda g: gravitate_left(set_fg_color(g, 3)),
'set_fg_color3+gravitate_up': lambda g: gravitate_up(set_fg_color(g, 3)),
'set_fg_color3+gravitate_down': lambda g: gravitate_down(set_fg_color(g, 3)),
'set_fg_color3+gravitate_left_right': lambda g: gravitate_left_right(set_fg_color(g, 3)),
'set_fg_color3+gravitate_top_down': lambda g: gravitate_top_down(set_fg_color(g, 3)),
'set_fg_color3+topthird': lambda g: topthird(set_fg_color(g, 3)),
'set_fg_color3+vcenterthird': lambda g: vcenterthird(set_fg_color(g, 3)),
'set_fg_color3+bottomthird': lambda g: bottomthird(set_fg_color(g, 3)),
'set_fg_color3+leftthird': lambda g: leftthird(set_fg_color(g, 3)),
'set_fg_color3+hcenterthird': lambda g: hcenterthird(set_fg_color(g, 3)),
'set_fg_color3+rightthird': lambda g: rightthird(set_fg_color(g, 3)),
'set_fg_color4+shift_left': lambda g: shift_left(set_fg_color(g, 4)),
'set_fg_color4+shift_right': lambda g: shift_right(set_fg_color(g, 4)),
'set_fg_color4+shift_up': lambda g: shift_up(set_fg_color(g, 4)),
'set_fg_color4+shift_down': lambda g: shift_down(set_fg_color(g, 4)),
'set_fg_color4+vmirror': lambda g: vmirror(set_fg_color(g, 4)),
'set_fg_color4+hmirror': lambda g: hmirror(set_fg_color(g, 4)),
'set_fg_color4+rot90': lambda g: rot90(set_fg_color(g, 4)),
'set_fg_color4+tophalf': lambda g: tophalf(set_fg_color(g, 4)),
'set_fg_color4+bottomhalf': lambda g: bottomhalf(set_fg_color(g, 4)),
'set_fg_color4+lefthalf': lambda g: lefthalf(set_fg_color(g, 4)),
'set_fg_color4+righthalf': lambda g: righthalf(set_fg_color(g, 4)),
'set_fg_color4+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(set_fg_color(g, 4)),
'set_fg_color4+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(set_fg_color(g, 4)),
'set_fg_color4+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(set_fg_color(g, 4)),
'set_fg_color4+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(set_fg_color(g, 4)),
'set_fg_color4+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(set_fg_color(g, 4)),
'set_fg_color4+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(set_fg_color(g, 4)),
'set_fg_color4+gravitate_right': lambda g: gravitate_right(set_fg_color(g, 4)),
'set_fg_color4+gravitate_left': lambda g: gravitate_left(set_fg_color(g, 4)),
'set_fg_color4+gravitate_up': lambda g: gravitate_up(set_fg_color(g, 4)),
'set_fg_color4+gravitate_down': lambda g: gravitate_down(set_fg_color(g, 4)),
'set_fg_color4+gravitate_left_right': lambda g: gravitate_left_right(set_fg_color(g, 4)),
'set_fg_color4+gravitate_top_down': lambda g: gravitate_top_down(set_fg_color(g, 4)),
'set_fg_color4+topthird': lambda g: topthird(set_fg_color(g, 4)),
'set_fg_color4+vcenterthird': lambda g: vcenterthird(set_fg_color(g, 4)),
'set_fg_color4+bottomthird': lambda g: bottomthird(set_fg_color(g, 4)),
'set_fg_color4+leftthird': lambda g: leftthird(set_fg_color(g, 4)),
'set_fg_color4+hcenterthird': lambda g: hcenterthird(set_fg_color(g, 4)),
'set_fg_color4+rightthird': lambda g: rightthird(set_fg_color(g, 4)),
'set_fg_color5+shift_left': lambda g: shift_left(set_fg_color(g, 5)),
'set_fg_color5+shift_right': lambda g: shift_right(set_fg_color(g, 5)),
'set_fg_color5+shift_up': lambda g: shift_up(set_fg_color(g, 5)),
'set_fg_color5+shift_down': lambda g: shift_down(set_fg_color(g, 5)),
'set_fg_color5+vmirror': lambda g: vmirror(set_fg_color(g, 5)),
'set_fg_color5+hmirror': lambda g: hmirror(set_fg_color(g, 5)),
'set_fg_color5+rot90': lambda g: rot90(set_fg_color(g, 5)),
'set_fg_color5+tophalf': lambda g: tophalf(set_fg_color(g, 5)),
'set_fg_color5+bottomhalf': lambda g: bottomhalf(set_fg_color(g, 5)),
'set_fg_color5+lefthalf': lambda g: lefthalf(set_fg_color(g, 5)),
'set_fg_color5+righthalf': lambda g: righthalf(set_fg_color(g, 5)),
'set_fg_color5+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(set_fg_color(g, 5)),
'set_fg_color5+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(set_fg_color(g, 5)),
'set_fg_color5+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(set_fg_color(g, 5)),
'set_fg_color5+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(set_fg_color(g, 5)),
'set_fg_color5+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(set_fg_color(g, 5)),
'set_fg_color5+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(set_fg_color(g, 5)),
'set_fg_color5+gravitate_right': lambda g: gravitate_right(set_fg_color(g, 5)),
'set_fg_color5+gravitate_left': lambda g: gravitate_left(set_fg_color(g, 5)),
'set_fg_color5+gravitate_up': lambda g: gravitate_up(set_fg_color(g, 5)),
'set_fg_color5+gravitate_down': lambda g: gravitate_down(set_fg_color(g, 5)),
'set_fg_color5+gravitate_left_right': lambda g: gravitate_left_right(set_fg_color(g, 5)),
'set_fg_color5+gravitate_top_down': lambda g: gravitate_top_down(set_fg_color(g, 5)),
'set_fg_color5+topthird': lambda g: topthird(set_fg_color(g, 5)),
'set_fg_color5+vcenterthird': lambda g: vcenterthird(set_fg_color(g, 5)),
'set_fg_color5+bottomthird': lambda g: bottomthird(set_fg_color(g, 5)),
'set_fg_color5+leftthird': lambda g: leftthird(set_fg_color(g, 5)),
'set_fg_color5+hcenterthird': lambda g: hcenterthird(set_fg_color(g, 5)),
'set_fg_color5+rightthird': lambda g: rightthird(set_fg_color(g, 5)),
'set_fg_color6+shift_left': lambda g: shift_left(set_fg_color(g, 6)),
'set_fg_color6+shift_right': lambda g: shift_right(set_fg_color(g, 6)),
'set_fg_color6+shift_up': lambda g: shift_up(set_fg_color(g, 6)),
'set_fg_color6+shift_down': lambda g: shift_down(set_fg_color(g, 6)),
'set_fg_color6+vmirror': lambda g: vmirror(set_fg_color(g, 6)),
'set_fg_color6+hmirror': lambda g: hmirror(set_fg_color(g, 6)),
'set_fg_color6+rot90': lambda g: rot90(set_fg_color(g, 6)),
'set_fg_color6+tophalf': lambda g: tophalf(set_fg_color(g, 6)),
'set_fg_color6+bottomhalf': lambda g: bottomhalf(set_fg_color(g, 6)),
'set_fg_color6+lefthalf': lambda g: lefthalf(set_fg_color(g, 6)),
'set_fg_color6+righthalf': lambda g: righthalf(set_fg_color(g, 6)),
'set_fg_color6+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(set_fg_color(g, 6)),
'set_fg_color6+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(set_fg_color(g, 6)),
'set_fg_color6+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(set_fg_color(g, 6)),
'set_fg_color6+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(set_fg_color(g, 6)),
'set_fg_color6+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(set_fg_color(g, 6)),
'set_fg_color6+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(set_fg_color(g, 6)),
'set_fg_color6+gravitate_right': lambda g: gravitate_right(set_fg_color(g, 6)),
'set_fg_color6+gravitate_left': lambda g: gravitate_left(set_fg_color(g, 6)),
'set_fg_color6+gravitate_up': lambda g: gravitate_up(set_fg_color(g, 6)),
'set_fg_color6+gravitate_down': lambda g: gravitate_down(set_fg_color(g, 6)),
'set_fg_color6+gravitate_left_right': lambda g: gravitate_left_right(set_fg_color(g, 6)),
'set_fg_color6+gravitate_top_down': lambda g: gravitate_top_down(set_fg_color(g, 6)),
'set_fg_color6+topthird': lambda g: topthird(set_fg_color(g, 6)),
'set_fg_color6+vcenterthird': lambda g: vcenterthird(set_fg_color(g, 6)),
'set_fg_color6+bottomthird': lambda g: bottomthird(set_fg_color(g, 6)),
'set_fg_color6+leftthird': lambda g: leftthird(set_fg_color(g, 6)),
'set_fg_color6+hcenterthird': lambda g: hcenterthird(set_fg_color(g, 6)),
'set_fg_color6+rightthird': lambda g: rightthird(set_fg_color(g, 6)),
'set_fg_color7+shift_left': lambda g: shift_left(set_fg_color(g, 7)),
'set_fg_color7+shift_right': lambda g: shift_right(set_fg_color(g, 7)),
'set_fg_color7+shift_up': lambda g: shift_up(set_fg_color(g, 7)),
'set_fg_color7+shift_down': lambda g: shift_down(set_fg_color(g, 7)),
'set_fg_color7+vmirror': lambda g: vmirror(set_fg_color(g, 7)),
'set_fg_color7+hmirror': lambda g: hmirror(set_fg_color(g, 7)),
'set_fg_color7+rot90': lambda g: rot90(set_fg_color(g, 7)),
'set_fg_color7+tophalf': lambda g: tophalf(set_fg_color(g, 7)),
'set_fg_color7+bottomhalf': lambda g: bottomhalf(set_fg_color(g, 7)),
'set_fg_color7+lefthalf': lambda g: lefthalf(set_fg_color(g, 7)),
'set_fg_color7+righthalf': lambda g: righthalf(set_fg_color(g, 7)),
'set_fg_color7+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(set_fg_color(g, 7)),
'set_fg_color7+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(set_fg_color(g, 7)),
'set_fg_color7+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(set_fg_color(g, 7)),
'set_fg_color7+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(set_fg_color(g, 7)),
'set_fg_color7+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(set_fg_color(g, 7)),
'set_fg_color7+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(set_fg_color(g, 7)),
'set_fg_color7+gravitate_right': lambda g: gravitate_right(set_fg_color(g, 7)),
'set_fg_color7+gravitate_left': lambda g: gravitate_left(set_fg_color(g, 7)),
'set_fg_color7+gravitate_up': lambda g: gravitate_up(set_fg_color(g, 7)),
'set_fg_color7+gravitate_down': lambda g: gravitate_down(set_fg_color(g, 7)),
'set_fg_color7+gravitate_left_right': lambda g: gravitate_left_right(set_fg_color(g, 7)),
'set_fg_color7+gravitate_top_down': lambda g: gravitate_top_down(set_fg_color(g, 7)),
'set_fg_color7+topthird': lambda g: topthird(set_fg_color(g, 7)),
'set_fg_color7+vcenterthird': lambda g: vcenterthird(set_fg_color(g, 7)),
'set_fg_color7+bottomthird': lambda g: bottomthird(set_fg_color(g, 7)),
'set_fg_color7+leftthird': lambda g: leftthird(set_fg_color(g, 7)),
'set_fg_color7+hcenterthird': lambda g: hcenterthird(set_fg_color(g, 7)),
'set_fg_color7+rightthird': lambda g: rightthird(set_fg_color(g, 7)),
'set_fg_color8+shift_left': lambda g: shift_left(set_fg_color(g, 8)),
'set_fg_color8+shift_right': lambda g: shift_right(set_fg_color(g, 8)),
'set_fg_color8+shift_up': lambda g: shift_up(set_fg_color(g, 8)),
'set_fg_color8+shift_down': lambda g: shift_down(set_fg_color(g, 8)),
'set_fg_color8+vmirror': lambda g: vmirror(set_fg_color(g, 8)),
'set_fg_color8+hmirror': lambda g: hmirror(set_fg_color(g, 8)),
'set_fg_color8+rot90': lambda g: rot90(set_fg_color(g, 8)),
'set_fg_color8+tophalf': lambda g: tophalf(set_fg_color(g, 8)),
'set_fg_color8+bottomhalf': lambda g: bottomhalf(set_fg_color(g, 8)),
'set_fg_color8+lefthalf': lambda g: lefthalf(set_fg_color(g, 8)),
'set_fg_color8+righthalf': lambda g: righthalf(set_fg_color(g, 8)),
'set_fg_color8+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(set_fg_color(g, 8)),
'set_fg_color8+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(set_fg_color(g, 8)),
'set_fg_color8+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(set_fg_color(g, 8)),
'set_fg_color8+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(set_fg_color(g, 8)),
'set_fg_color8+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(set_fg_color(g, 8)),
'set_fg_color8+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(set_fg_color(g, 8)),
'set_fg_color8+gravitate_right': lambda g: gravitate_right(set_fg_color(g, 8)),
'set_fg_color8+gravitate_left': lambda g: gravitate_left(set_fg_color(g, 8)),
'set_fg_color8+gravitate_up': lambda g: gravitate_up(set_fg_color(g, 8)),
'set_fg_color8+gravitate_down': lambda g: gravitate_down(set_fg_color(g, 8)),
'set_fg_color8+gravitate_left_right': lambda g: gravitate_left_right(set_fg_color(g, 8)),
'set_fg_color8+gravitate_top_down': lambda g: gravitate_top_down(set_fg_color(g, 8)),
'set_fg_color8+topthird': lambda g: topthird(set_fg_color(g, 8)),
'set_fg_color8+vcenterthird': lambda g: vcenterthird(set_fg_color(g, 8)),
'set_fg_color8+bottomthird': lambda g: bottomthird(set_fg_color(g, 8)),
'set_fg_color8+leftthird': lambda g: leftthird(set_fg_color(g, 8)),
'set_fg_color8+hcenterthird': lambda g: hcenterthird(set_fg_color(g, 8)),
'set_fg_color8+rightthird': lambda g: rightthird(set_fg_color(g, 8)),
'set_fg_color9+shift_left': lambda g: shift_left(set_fg_color(g, 9)),
'set_fg_color9+shift_right': lambda g: shift_right(set_fg_color(g, 9)),
'set_fg_color9+shift_up': lambda g: shift_up(set_fg_color(g, 9)),
'set_fg_color9+shift_down': lambda g: shift_down(set_fg_color(g, 9)),
'set_fg_color9+vmirror': lambda g: vmirror(set_fg_color(g, 9)),
'set_fg_color9+hmirror': lambda g: hmirror(set_fg_color(g, 9)),
'set_fg_color9+rot90': lambda g: rot90(set_fg_color(g, 9)),
'set_fg_color9+tophalf': lambda g: tophalf(set_fg_color(g, 9)),
'set_fg_color9+bottomhalf': lambda g: bottomhalf(set_fg_color(g, 9)),
'set_fg_color9+lefthalf': lambda g: lefthalf(set_fg_color(g, 9)),
'set_fg_color9+righthalf': lambda g: righthalf(set_fg_color(g, 9)),
'set_fg_color9+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(set_fg_color(g, 9)),
'set_fg_color9+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(set_fg_color(g, 9)),
'set_fg_color9+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(set_fg_color(g, 9)),
'set_fg_color9+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(set_fg_color(g, 9)),
'set_fg_color9+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(set_fg_color(g, 9)),
'set_fg_color9+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(set_fg_color(g, 9)),
'set_fg_color9+gravitate_right': lambda g: gravitate_right(set_fg_color(g, 9)),
'set_fg_color9+gravitate_left': lambda g: gravitate_left(set_fg_color(g, 9)),
'set_fg_color9+gravitate_up': lambda g: gravitate_up(set_fg_color(g, 9)),
'set_fg_color9+gravitate_down': lambda g: gravitate_down(set_fg_color(g, 9)),
'set_fg_color9+gravitate_left_right': lambda g: gravitate_left_right(set_fg_color(g, 9)),
'set_fg_color9+gravitate_top_down': lambda g: gravitate_top_down(set_fg_color(g, 9)),
'set_fg_color9+topthird': lambda g: topthird(set_fg_color(g, 9)),
'set_fg_color9+vcenterthird': lambda g: vcenterthird(set_fg_color(g, 9)),
'set_fg_color9+bottomthird': lambda g: bottomthird(set_fg_color(g, 9)),
'set_fg_color9+leftthird': lambda g: leftthird(set_fg_color(g, 9)),
'set_fg_color9+hcenterthird': lambda g: hcenterthird(set_fg_color(g, 9)),
'set_fg_color9+rightthird': lambda g: rightthird(set_fg_color(g, 9)),
'shift_left+shift_left': lambda g: shift_left(shift_left(g)),
'shift_left+shift_up': lambda g: shift_up(shift_left(g)),
'shift_left+shift_down': lambda g: shift_down(shift_left(g)),
'shift_left+vmirror': lambda g: vmirror(shift_left(g)),
'shift_left+hmirror': lambda g: hmirror(shift_left(g)),
'shift_left+rot90': lambda g: rot90(shift_left(g)),
'shift_left+tophalf': lambda g: tophalf(shift_left(g)),
'shift_left+bottomhalf': lambda g: bottomhalf(shift_left(g)),
'shift_left+lefthalf': lambda g: lefthalf(shift_left(g)),
'shift_left+righthalf': lambda g: righthalf(shift_left(g)),
'shift_left+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(shift_left(g)),
'shift_left+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(shift_left(g)),
'shift_left+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(shift_left(g)),
'shift_left+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(shift_left(g)),
'shift_left+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(shift_left(g)),
'shift_left+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(shift_left(g)),
'shift_left+gravitate_right': lambda g: gravitate_right(shift_left(g)),
'shift_left+gravitate_left': lambda g: gravitate_left(shift_left(g)),
'shift_left+gravitate_up': lambda g: gravitate_up(shift_left(g)),
'shift_left+gravitate_down': lambda g: gravitate_down(shift_left(g)),
'shift_left+gravitate_left_right': lambda g: gravitate_left_right(shift_left(g)),
'shift_left+gravitate_top_down': lambda g: gravitate_top_down(shift_left(g)),
'shift_left+topthird': lambda g: topthird(shift_left(g)),
'shift_left+vcenterthird': lambda g: vcenterthird(shift_left(g)),
'shift_left+bottomthird': lambda g: bottomthird(shift_left(g)),
'shift_left+leftthird': lambda g: leftthird(shift_left(g)),
'shift_left+hcenterthird': lambda g: hcenterthird(shift_left(g)),
'shift_left+rightthird': lambda g: rightthird(shift_left(g)),
'shift_right+shift_right': lambda g: shift_right(shift_right(g)),
'shift_right+shift_up': lambda g: shift_up(shift_right(g)),
'shift_right+shift_down': lambda g: shift_down(shift_right(g)),
'shift_right+vmirror': lambda g: vmirror(shift_right(g)),
'shift_right+hmirror': lambda g: hmirror(shift_right(g)),
'shift_right+rot90': lambda g: rot90(shift_right(g)),
'shift_right+tophalf': lambda g: tophalf(shift_right(g)),
'shift_right+bottomhalf': lambda g: bottomhalf(shift_right(g)),
'shift_right+lefthalf': lambda g: lefthalf(shift_right(g)),
'shift_right+righthalf': lambda g: righthalf(shift_right(g)),
'shift_right+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(shift_right(g)),
'shift_right+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(shift_right(g)),
'shift_right+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(shift_right(g)),
'shift_right+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(shift_right(g)),
'shift_right+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(shift_right(g)),
'shift_right+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(shift_right(g)),
'shift_right+gravitate_right': lambda g: gravitate_right(shift_right(g)),
'shift_right+gravitate_left': lambda g: gravitate_left(shift_right(g)),
'shift_right+gravitate_up': lambda g: gravitate_up(shift_right(g)),
'shift_right+gravitate_down': lambda g: gravitate_down(shift_right(g)),
'shift_right+gravitate_left_right': lambda g: gravitate_left_right(shift_right(g)),
'shift_right+gravitate_top_down': lambda g: gravitate_top_down(shift_right(g)),
'shift_right+topthird': lambda g: topthird(shift_right(g)),
'shift_right+vcenterthird': lambda g: vcenterthird(shift_right(g)),
'shift_right+bottomthird': lambda g: bottomthird(shift_right(g)),
'shift_right+leftthird': lambda g: leftthird(shift_right(g)),
'shift_right+hcenterthird': lambda g: hcenterthird(shift_right(g)),
'shift_right+rightthird': lambda g: rightthird(shift_right(g)),
'shift_up+shift_up': lambda g: shift_up(shift_up(g)),
'shift_up+vmirror': lambda g: vmirror(shift_up(g)),
'shift_up+hmirror': lambda g: hmirror(shift_up(g)),
'shift_up+rot90': lambda g: rot90(shift_up(g)),
'shift_up+tophalf': lambda g: tophalf(shift_up(g)),
'shift_up+bottomhalf': lambda g: bottomhalf(shift_up(g)),
'shift_up+lefthalf': lambda g: lefthalf(shift_up(g)),
'shift_up+righthalf': lambda g: righthalf(shift_up(g)),
'shift_up+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(shift_up(g)),
'shift_up+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(shift_up(g)),
'shift_up+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(shift_up(g)),
'shift_up+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(shift_up(g)),
'shift_up+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(shift_up(g)),
'shift_up+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(shift_up(g)),
'shift_up+gravitate_right': lambda g: gravitate_right(shift_up(g)),
'shift_up+gravitate_left': lambda g: gravitate_left(shift_up(g)),
'shift_up+gravitate_up': lambda g: gravitate_up(shift_up(g)),
'shift_up+gravitate_down': lambda g: gravitate_down(shift_up(g)),
'shift_up+gravitate_left_right': lambda g: gravitate_left_right(shift_up(g)),
'shift_up+gravitate_top_down': lambda g: gravitate_top_down(shift_up(g)),
'shift_up+topthird': lambda g: topthird(shift_up(g)),
'shift_up+vcenterthird': lambda g: vcenterthird(shift_up(g)),
'shift_up+bottomthird': lambda g: bottomthird(shift_up(g)),
'shift_up+leftthird': lambda g: leftthird(shift_up(g)),
'shift_up+hcenterthird': lambda g: hcenterthird(shift_up(g)),
'shift_up+rightthird': lambda g: rightthird(shift_up(g)),
'shift_down+shift_down': lambda g: shift_down(shift_down(g)),
'shift_down+vmirror': lambda g: vmirror(shift_down(g)),
'shift_down+hmirror': lambda g: hmirror(shift_down(g)),
'shift_down+rot90': lambda g: rot90(shift_down(g)),
'shift_down+tophalf': lambda g: tophalf(shift_down(g)),
'shift_down+bottomhalf': lambda g: bottomhalf(shift_down(g)),
'shift_down+lefthalf': lambda g: lefthalf(shift_down(g)),
'shift_down+righthalf': lambda g: righthalf(shift_down(g)),
'shift_down+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(shift_down(g)),
'shift_down+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(shift_down(g)),
'shift_down+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(shift_down(g)),
'shift_down+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(shift_down(g)),
'shift_down+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(shift_down(g)),
'shift_down+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(shift_down(g)),
'shift_down+gravitate_right': lambda g: gravitate_right(shift_down(g)),
'shift_down+gravitate_left': lambda g: gravitate_left(shift_down(g)),
'shift_down+gravitate_up': lambda g: gravitate_up(shift_down(g)),
'shift_down+gravitate_down': lambda g: gravitate_down(shift_down(g)),
'shift_down+gravitate_left_right': lambda g: gravitate_left_right(shift_down(g)),
'shift_down+gravitate_top_down': lambda g: gravitate_top_down(shift_down(g)),
'shift_down+topthird': lambda g: topthird(shift_down(g)),
'shift_down+vcenterthird': lambda g: vcenterthird(shift_down(g)),
'shift_down+bottomthird': lambda g: bottomthird(shift_down(g)),
'shift_down+leftthird': lambda g: leftthird(shift_down(g)),
'shift_down+hcenterthird': lambda g: hcenterthird(shift_down(g)),
'shift_down+rightthird': lambda g: rightthird(shift_down(g)),
'vmirror+hmirror': lambda g: hmirror(vmirror(g)),
'vmirror+tophalf': lambda g: tophalf(vmirror(g)),
'vmirror+bottomhalf': lambda g: bottomhalf(vmirror(g)),
'vmirror+lefthalf': lambda g: lefthalf(vmirror(g)),
'vmirror+righthalf': lambda g: righthalf(vmirror(g)),
'vmirror+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(vmirror(g)),
'vmirror+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(vmirror(g)),
'vmirror+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(vmirror(g)),
'vmirror+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(vmirror(g)),
'vmirror+gravitate_right': lambda g: gravitate_right(vmirror(g)),
'vmirror+gravitate_left': lambda g: gravitate_left(vmirror(g)),
'vmirror+gravitate_up': lambda g: gravitate_up(vmirror(g)),
'vmirror+gravitate_down': lambda g: gravitate_down(vmirror(g)),
'vmirror+gravitate_left_right': lambda g: gravitate_left_right(vmirror(g)),
'vmirror+gravitate_top_down': lambda g: gravitate_top_down(vmirror(g)),
'vmirror+topthird': lambda g: topthird(vmirror(g)),
'vmirror+vcenterthird': lambda g: vcenterthird(vmirror(g)),
'vmirror+bottomthird': lambda g: bottomthird(vmirror(g)),
'vmirror+leftthird': lambda g: leftthird(vmirror(g)),
'vmirror+hcenterthird': lambda g: hcenterthird(vmirror(g)),
'vmirror+rightthird': lambda g: rightthird(vmirror(g)),
'hmirror+tophalf': lambda g: tophalf(hmirror(g)),
'hmirror+bottomhalf': lambda g: bottomhalf(hmirror(g)),
'hmirror+lefthalf': lambda g: lefthalf(hmirror(g)),
'hmirror+righthalf': lambda g: righthalf(hmirror(g)),
'hmirror+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(hmirror(g)),
'hmirror+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(hmirror(g)),
'hmirror+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(hmirror(g)),
'hmirror+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(hmirror(g)),
'hmirror+gravitate_right': lambda g: gravitate_right(hmirror(g)),
'hmirror+gravitate_left': lambda g: gravitate_left(hmirror(g)),
'hmirror+gravitate_up': lambda g: gravitate_up(hmirror(g)),
'hmirror+gravitate_down': lambda g: gravitate_down(hmirror(g)),
'hmirror+gravitate_left_right': lambda g: gravitate_left_right(hmirror(g)),
'hmirror+gravitate_top_down': lambda g: gravitate_top_down(hmirror(g)),
'hmirror+topthird': lambda g: topthird(hmirror(g)),
'hmirror+vcenterthird': lambda g: vcenterthird(hmirror(g)),
'hmirror+bottomthird': lambda g: bottomthird(hmirror(g)),
'hmirror+leftthird': lambda g: leftthird(hmirror(g)),
'hmirror+hcenterthird': lambda g: hcenterthird(hmirror(g)),
'hmirror+rightthird': lambda g: rightthird(hmirror(g)),
'rot90+tophalf': lambda g: tophalf(rot90(g)),
'rot90+bottomhalf': lambda g: bottomhalf(rot90(g)),
'rot90+lefthalf': lambda g: lefthalf(rot90(g)),
'rot90+righthalf': lambda g: righthalf(rot90(g)),
'rot90+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(rot90(g)),
'rot90+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(rot90(g)),
'rot90+gravitate_right': lambda g: gravitate_right(rot90(g)),
'rot90+gravitate_left': lambda g: gravitate_left(rot90(g)),
'rot90+gravitate_up': lambda g: gravitate_up(rot90(g)),
'rot90+gravitate_down': lambda g: gravitate_down(rot90(g)),
'rot90+gravitate_left_right': lambda g: gravitate_left_right(rot90(g)),
'rot90+gravitate_top_down': lambda g: gravitate_top_down(rot90(g)),
'rot90+topthird': lambda g: topthird(rot90(g)),
'rot90+vcenterthird': lambda g: vcenterthird(rot90(g)),
'rot90+bottomthird': lambda g: bottomthird(rot90(g)),
'rot90+leftthird': lambda g: leftthird(rot90(g)),
'rot90+hcenterthird': lambda g: hcenterthird(rot90(g)),
'rot90+rightthird': lambda g: rightthird(rot90(g)),
'tophalf+shift_up': lambda g: shift_up(tophalf(g)),
'tophalf+shift_down': lambda g: shift_down(tophalf(g)),
'tophalf+tophalf': lambda g: tophalf(tophalf(g)),
'tophalf+bottomhalf': lambda g: bottomhalf(tophalf(g)),
'tophalf+lefthalf': lambda g: lefthalf(tophalf(g)),
'tophalf+righthalf': lambda g: righthalf(tophalf(g)),
'tophalf+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(tophalf(g)),
'tophalf+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(tophalf(g)),
'tophalf+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(tophalf(g)),
'tophalf+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(tophalf(g)),
'tophalf+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(tophalf(g)),
'tophalf+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(tophalf(g)),
'tophalf+gravitate_right': lambda g: gravitate_right(tophalf(g)),
'tophalf+gravitate_left': lambda g: gravitate_left(tophalf(g)),
'tophalf+gravitate_up': lambda g: gravitate_up(tophalf(g)),
'tophalf+gravitate_down': lambda g: gravitate_down(tophalf(g)),
'tophalf+gravitate_left_right': lambda g: gravitate_left_right(tophalf(g)),
'tophalf+gravitate_top_down': lambda g: gravitate_top_down(tophalf(g)),
'tophalf+topthird': lambda g: topthird(tophalf(g)),
'tophalf+vcenterthird': lambda g: vcenterthird(tophalf(g)),
'tophalf+bottomthird': lambda g: bottomthird(tophalf(g)),
'tophalf+leftthird': lambda g: leftthird(tophalf(g)),
'tophalf+hcenterthird': lambda g: hcenterthird(tophalf(g)),
'tophalf+rightthird': lambda g: rightthird(tophalf(g)),
'bottomhalf+shift_up': lambda g: shift_up(bottomhalf(g)),
'bottomhalf+shift_down': lambda g: shift_down(bottomhalf(g)),
'bottomhalf+tophalf': lambda g: tophalf(bottomhalf(g)),
'bottomhalf+bottomhalf': lambda g: bottomhalf(bottomhalf(g)),
'bottomhalf+lefthalf': lambda g: lefthalf(bottomhalf(g)),
'bottomhalf+righthalf': lambda g: righthalf(bottomhalf(g)),
'bottomhalf+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(bottomhalf(g)),
'bottomhalf+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(bottomhalf(g)),
'bottomhalf+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(bottomhalf(g)),
'bottomhalf+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(bottomhalf(g)),
'bottomhalf+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(bottomhalf(g)),
'bottomhalf+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(bottomhalf(g)),
'bottomhalf+gravitate_right': lambda g: gravitate_right(bottomhalf(g)),
'bottomhalf+gravitate_left': lambda g: gravitate_left(bottomhalf(g)),
'bottomhalf+gravitate_up': lambda g: gravitate_up(bottomhalf(g)),
'bottomhalf+gravitate_down': lambda g: gravitate_down(bottomhalf(g)),
'bottomhalf+gravitate_left_right': lambda g: gravitate_left_right(bottomhalf(g)),
'bottomhalf+gravitate_top_down': lambda g: gravitate_top_down(bottomhalf(g)),
'bottomhalf+topthird': lambda g: topthird(bottomhalf(g)),
'bottomhalf+vcenterthird': lambda g: vcenterthird(bottomhalf(g)),
'bottomhalf+bottomthird': lambda g: bottomthird(bottomhalf(g)),
'bottomhalf+leftthird': lambda g: leftthird(bottomhalf(g)),
'bottomhalf+hcenterthird': lambda g: hcenterthird(bottomhalf(g)),
'bottomhalf+rightthird': lambda g: rightthird(bottomhalf(g)),
'lefthalf+shift_left': lambda g: shift_left(lefthalf(g)),
'lefthalf+shift_right': lambda g: shift_right(lefthalf(g)),
'lefthalf+lefthalf': lambda g: lefthalf(lefthalf(g)),
'lefthalf+righthalf': lambda g: righthalf(lefthalf(g)),
'lefthalf+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(lefthalf(g)),
'lefthalf+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(lefthalf(g)),
'lefthalf+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(lefthalf(g)),
'lefthalf+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(lefthalf(g)),
'lefthalf+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(lefthalf(g)),
'lefthalf+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(lefthalf(g)),
'lefthalf+gravitate_right': lambda g: gravitate_right(lefthalf(g)),
'lefthalf+gravitate_left': lambda g: gravitate_left(lefthalf(g)),
'lefthalf+gravitate_up': lambda g: gravitate_up(lefthalf(g)),
'lefthalf+gravitate_down': lambda g: gravitate_down(lefthalf(g)),
'lefthalf+gravitate_left_right': lambda g: gravitate_left_right(lefthalf(g)),
'lefthalf+gravitate_top_down': lambda g: gravitate_top_down(lefthalf(g)),
'lefthalf+topthird': lambda g: topthird(lefthalf(g)),
'lefthalf+vcenterthird': lambda g: vcenterthird(lefthalf(g)),
'lefthalf+bottomthird': lambda g: bottomthird(lefthalf(g)),
'lefthalf+leftthird': lambda g: leftthird(lefthalf(g)),
'lefthalf+hcenterthird': lambda g: hcenterthird(lefthalf(g)),
'lefthalf+rightthird': lambda g: rightthird(lefthalf(g)),
'righthalf+shift_left': lambda g: shift_left(righthalf(g)),
'righthalf+shift_right': lambda g: shift_right(righthalf(g)),
'righthalf+lefthalf': lambda g: lefthalf(righthalf(g)),
'righthalf+righthalf': lambda g: righthalf(righthalf(g)),
'righthalf+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(righthalf(g)),
'righthalf+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(righthalf(g)),
'righthalf+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(righthalf(g)),
'righthalf+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(righthalf(g)),
'righthalf+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(righthalf(g)),
'righthalf+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(righthalf(g)),
'righthalf+gravitate_right': lambda g: gravitate_right(righthalf(g)),
'righthalf+gravitate_left': lambda g: gravitate_left(righthalf(g)),
'righthalf+gravitate_up': lambda g: gravitate_up(righthalf(g)),
'righthalf+gravitate_down': lambda g: gravitate_down(righthalf(g)),
'righthalf+gravitate_left_right': lambda g: gravitate_left_right(righthalf(g)),
'righthalf+gravitate_top_down': lambda g: gravitate_top_down(righthalf(g)),
'righthalf+topthird': lambda g: topthird(righthalf(g)),
'righthalf+vcenterthird': lambda g: vcenterthird(righthalf(g)),
'righthalf+bottomthird': lambda g: bottomthird(righthalf(g)),
'righthalf+leftthird': lambda g: leftthird(righthalf(g)),
'righthalf+hcenterthird': lambda g: hcenterthird(righthalf(g)),
'righthalf+rightthird': lambda g: rightthird(righthalf(g)),
'symmetrize_left_around_vertical+shift_left': lambda g: shift_left(symmetrize_left_around_vertical(g)),
'symmetrize_left_around_vertical+shift_right': lambda g: shift_right(symmetrize_left_around_vertical(g)),
'symmetrize_left_around_vertical+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(symmetrize_left_around_vertical(g)),
'symmetrize_left_around_vertical+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(symmetrize_left_around_vertical(g)),
'symmetrize_left_around_vertical+topthird': lambda g: topthird(symmetrize_left_around_vertical(g)),
'symmetrize_left_around_vertical+vcenterthird': lambda g: vcenterthird(symmetrize_left_around_vertical(g)),
'symmetrize_left_around_vertical+bottomthird': lambda g: bottomthird(symmetrize_left_around_vertical(g)),
'symmetrize_left_around_vertical+leftthird': lambda g: leftthird(symmetrize_left_around_vertical(g)),
'symmetrize_left_around_vertical+hcenterthird': lambda g: hcenterthird(symmetrize_left_around_vertical(g)),
'symmetrize_left_around_vertical+rightthird': lambda g: rightthird(symmetrize_left_around_vertical(g)),
'symmetrize_right_around_vertical+shift_left': lambda g: shift_left(symmetrize_right_around_vertical(g)),
'symmetrize_right_around_vertical+shift_right': lambda g: shift_right(symmetrize_right_around_vertical(g)),
'symmetrize_right_around_vertical+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(symmetrize_right_around_vertical(g)),
'symmetrize_right_around_vertical+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(symmetrize_right_around_vertical(g)),
'symmetrize_right_around_vertical+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(symmetrize_right_around_vertical(g)),
'symmetrize_right_around_vertical+topthird': lambda g: topthird(symmetrize_right_around_vertical(g)),
'symmetrize_right_around_vertical+vcenterthird': lambda g: vcenterthird(symmetrize_right_around_vertical(g)),
'symmetrize_right_around_vertical+bottomthird': lambda g: bottomthird(symmetrize_right_around_vertical(g)),
'symmetrize_right_around_vertical+leftthird': lambda g: leftthird(symmetrize_right_around_vertical(g)),
'symmetrize_right_around_vertical+hcenterthird': lambda g: hcenterthird(symmetrize_right_around_vertical(g)),
'symmetrize_right_around_vertical+rightthird': lambda g: rightthird(symmetrize_right_around_vertical(g)),
'symmetrize_top_around_horizontal+shift_up': lambda g: shift_up(symmetrize_top_around_horizontal(g)),
'symmetrize_top_around_horizontal+shift_down': lambda g: shift_down(symmetrize_top_around_horizontal(g)),
'symmetrize_top_around_horizontal+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(symmetrize_top_around_horizontal(g)),
'symmetrize_top_around_horizontal+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(symmetrize_top_around_horizontal(g)),
'symmetrize_top_around_horizontal+topthird': lambda g: topthird(symmetrize_top_around_horizontal(g)),
'symmetrize_top_around_horizontal+vcenterthird': lambda g: vcenterthird(symmetrize_top_around_horizontal(g)),
'symmetrize_top_around_horizontal+bottomthird': lambda g: bottomthird(symmetrize_top_around_horizontal(g)),
'symmetrize_top_around_horizontal+leftthird': lambda g: leftthird(symmetrize_top_around_horizontal(g)),
'symmetrize_top_around_horizontal+hcenterthird': lambda g: hcenterthird(symmetrize_top_around_horizontal(g)),
'symmetrize_top_around_horizontal+rightthird': lambda g: rightthird(symmetrize_top_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+shift_up': lambda g: shift_up(symmetrize_bottom_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+shift_down': lambda g: shift_down(symmetrize_bottom_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(symmetrize_bottom_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(symmetrize_bottom_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+topthird': lambda g: topthird(symmetrize_bottom_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+vcenterthird': lambda g: vcenterthird(symmetrize_bottom_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+bottomthird': lambda g: bottomthird(symmetrize_bottom_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+leftthird': lambda g: leftthird(symmetrize_bottom_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+hcenterthird': lambda g: hcenterthird(symmetrize_bottom_around_horizontal(g)),
'symmetrize_bottom_around_horizontal+rightthird': lambda g: rightthird(symmetrize_bottom_around_horizontal(g)),
'upscale_horizontal_by_two+shift_left': lambda g: shift_left(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+shift_right': lambda g: shift_right(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+lefthalf': lambda g: lefthalf(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+righthalf': lambda g: righthalf(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+gravitate_right': lambda g: gravitate_right(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+gravitate_left': lambda g: gravitate_left(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+gravitate_up': lambda g: gravitate_up(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+gravitate_down': lambda g: gravitate_down(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+gravitate_left_right': lambda g: gravitate_left_right(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+gravitate_top_down': lambda g: gravitate_top_down(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+topthird': lambda g: topthird(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+vcenterthird': lambda g: vcenterthird(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+bottomthird': lambda g: bottomthird(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+leftthird': lambda g: leftthird(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+hcenterthird': lambda g: hcenterthird(upscale_horizontal_by_two(g)),
'upscale_horizontal_by_two+rightthird': lambda g: rightthird(upscale_horizontal_by_two(g)),
'upscale_vertical_by_two+shift_up': lambda g: shift_up(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+shift_down': lambda g: shift_down(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+tophalf': lambda g: tophalf(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+bottomhalf': lambda g: bottomhalf(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+gravitate_right': lambda g: gravitate_right(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+gravitate_left': lambda g: gravitate_left(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+gravitate_up': lambda g: gravitate_up(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+gravitate_down': lambda g: gravitate_down(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+gravitate_left_right': lambda g: gravitate_left_right(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+gravitate_top_down': lambda g: gravitate_top_down(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+topthird': lambda g: topthird(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+vcenterthird': lambda g: vcenterthird(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+bottomthird': lambda g: bottomthird(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+leftthird': lambda g: leftthird(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+hcenterthird': lambda g: hcenterthird(upscale_vertical_by_two(g)),
'upscale_vertical_by_two+rightthird': lambda g: rightthird(upscale_vertical_by_two(g)),
'gravitate_right+shift_left': lambda g: shift_left(gravitate_right(g)),
'gravitate_right+shift_right': lambda g: shift_right(gravitate_right(g)),
'gravitate_right+lefthalf': lambda g: lefthalf(gravitate_right(g)),
'gravitate_right+righthalf': lambda g: righthalf(gravitate_right(g)),
'gravitate_left+shift_left': lambda g: shift_left(gravitate_left(g)),
'gravitate_left+shift_right': lambda g: shift_right(gravitate_left(g)),
'gravitate_left+lefthalf': lambda g: lefthalf(gravitate_left(g)),
'gravitate_left+righthalf': lambda g: righthalf(gravitate_left(g)),
'gravitate_up+shift_up': lambda g: shift_up(gravitate_up(g)),
'gravitate_up+shift_down': lambda g: shift_down(gravitate_up(g)),
'gravitate_up+tophalf': lambda g: tophalf(gravitate_up(g)),
'gravitate_up+bottomhalf': lambda g: bottomhalf(gravitate_up(g)),
'gravitate_down+shift_up': lambda g: shift_up(gravitate_down(g)),
'gravitate_down+shift_down': lambda g: shift_down(gravitate_down(g)),
'gravitate_down+tophalf': lambda g: tophalf(gravitate_down(g)),
'gravitate_down+bottomhalf': lambda g: bottomhalf(gravitate_down(g)),
'gravitate_left_right+shift_left': lambda g: shift_left(gravitate_left_right(g)),
'gravitate_left_right+shift_right': lambda g: shift_right(gravitate_left_right(g)),
'gravitate_left_right+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(gravitate_left_right(g)),
'gravitate_left_right+gravitate_top_down': lambda g: gravitate_top_down(gravitate_left_right(g)),
'gravitate_top_down+shift_up': lambda g: shift_up(gravitate_top_down(g)),
'gravitate_top_down+shift_down': lambda g: shift_down(gravitate_top_down(g)),
'gravitate_top_down+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(gravitate_top_down(g)),
'topthird+shift_up': lambda g: shift_up(topthird(g)),
'topthird+shift_down': lambda g: shift_down(topthird(g)),
'topthird+bottomhalf': lambda g: bottomhalf(topthird(g)),
'topthird+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(topthird(g)),
'topthird+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(topthird(g)),
'topthird+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(topthird(g)),
'topthird+topthird': lambda g: topthird(topthird(g)),
'topthird+vcenterthird': lambda g: vcenterthird(topthird(g)),
'topthird+bottomthird': lambda g: bottomthird(topthird(g)),
'topthird+leftthird': lambda g: leftthird(topthird(g)),
'topthird+hcenterthird': lambda g: hcenterthird(topthird(g)),
'topthird+rightthird': lambda g: rightthird(topthird(g)),
'vcenterthird+shift_up': lambda g: shift_up(vcenterthird(g)),
'vcenterthird+shift_down': lambda g: shift_down(vcenterthird(g)),
'vcenterthird+hmirror': lambda g: hmirror(vcenterthird(g)),
'vcenterthird+rot90': lambda g: rot90(vcenterthird(g)),
'vcenterthird+tophalf': lambda g: tophalf(vcenterthird(g)),
'vcenterthird+bottomhalf': lambda g: bottomhalf(vcenterthird(g)),
'vcenterthird+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(vcenterthird(g)),
'vcenterthird+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(vcenterthird(g)),
'vcenterthird+topthird': lambda g: topthird(vcenterthird(g)),
'vcenterthird+vcenterthird': lambda g: vcenterthird(vcenterthird(g)),
'vcenterthird+bottomthird': lambda g: bottomthird(vcenterthird(g)),
'vcenterthird+leftthird': lambda g: leftthird(vcenterthird(g)),
'vcenterthird+hcenterthird': lambda g: hcenterthird(vcenterthird(g)),
'vcenterthird+rightthird': lambda g: rightthird(vcenterthird(g)),
'bottomthird+shift_up': lambda g: shift_up(bottomthird(g)),
'bottomthird+shift_down': lambda g: shift_down(bottomthird(g)),
'bottomthird+tophalf': lambda g: tophalf(bottomthird(g)),
'bottomthird+symmetrize_top_around_horizontal': lambda g: symmetrize_top_around_horizontal(bottomthird(g)),
'bottomthird+symmetrize_bottom_around_horizontal': lambda g: symmetrize_bottom_around_horizontal(bottomthird(g)),
'bottomthird+upscale_vertical_by_two': lambda g: upscale_vertical_by_two(bottomthird(g)),
'bottomthird+topthird': lambda g: topthird(bottomthird(g)),
'bottomthird+vcenterthird': lambda g: vcenterthird(bottomthird(g)),
'bottomthird+bottomthird': lambda g: bottomthird(bottomthird(g)),
'bottomthird+leftthird': lambda g: leftthird(bottomthird(g)),
'bottomthird+hcenterthird': lambda g: hcenterthird(bottomthird(g)),
'bottomthird+rightthird': lambda g: rightthird(bottomthird(g)),
'leftthird+shift_left': lambda g: shift_left(leftthird(g)),
'leftthird+shift_right': lambda g: shift_right(leftthird(g)),
'leftthird+righthalf': lambda g: righthalf(leftthird(g)),
'leftthird+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(leftthird(g)),
'leftthird+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(leftthird(g)),
'leftthird+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(leftthird(g)),
'leftthird+leftthird': lambda g: leftthird(leftthird(g)),
'leftthird+hcenterthird': lambda g: hcenterthird(leftthird(g)),
'leftthird+rightthird': lambda g: rightthird(leftthird(g)),
'hcenterthird+shift_left': lambda g: shift_left(hcenterthird(g)),
'hcenterthird+shift_right': lambda g: shift_right(hcenterthird(g)),
'hcenterthird+vmirror': lambda g: vmirror(hcenterthird(g)),
'hcenterthird+lefthalf': lambda g: lefthalf(hcenterthird(g)),
'hcenterthird+righthalf': lambda g: righthalf(hcenterthird(g)),
'hcenterthird+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(hcenterthird(g)),
'hcenterthird+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(hcenterthird(g)),
'hcenterthird+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(hcenterthird(g)),
'hcenterthird+leftthird': lambda g: leftthird(hcenterthird(g)),
'hcenterthird+hcenterthird': lambda g: hcenterthird(hcenterthird(g)),
'hcenterthird+rightthird': lambda g: rightthird(hcenterthird(g)),
'rightthird+shift_left': lambda g: shift_left(rightthird(g)),
'rightthird+shift_right': lambda g: shift_right(rightthird(g)),
'rightthird+lefthalf': lambda g: lefthalf(rightthird(g)),
'rightthird+symmetrize_left_around_vertical': lambda g: symmetrize_left_around_vertical(rightthird(g)),
'rightthird+symmetrize_right_around_vertical': lambda g: symmetrize_right_around_vertical(rightthird(g)),
'rightthird+upscale_horizontal_by_two': lambda g: upscale_horizontal_by_two(rightthird(g)),
'rightthird+leftthird': lambda g: leftthird(rightthird(g)),
'rightthird+hcenterthird': lambda g: hcenterthird(rightthird(g)),
'rightthird+rightthird': lambda g: rightthird(rightthird(g)),
'bottomhalf+rot270': lambda g: rot270(bottomhalf(g)),
'tophalf+rot270': lambda g: rot270(tophalf(g)),
'lefthalf+rot270': lambda g: rot270(lefthalf(g)),
'righthalf+rot270': lambda g: rot270(righthalf(g)),
'bottomhalf+rot180': lambda g: rot180(bottomhalf(g)),
'tophalf+rot180': lambda g: rot180(tophalf(g)),
'lefthalf+rot180': lambda g: rot180(lefthalf(g)),
'righthalf+rot180': lambda g: rot180(righthalf(g)),
'upscale_by_three': lambda g: upscale_by_three(g),
'rot180': lambda g: rot180(g),                # TODO: this is redundant with a pre-existing primitive. Which one?
'rot270': lambda g: rot270(g),
'duplicate_top_row': lambda g: duplicate_top_row(g),
'duplicate_bottom_row': lambda g: duplicate_bottom_row(g),
'duplicate_left_column': lambda g: duplicate_left_column(g),
'duplicate_right_column': lambda g: duplicate_right_column(g),
'duplicate_top_row+duplicate_bottom_row': lambda g: duplicate_bottom_row(duplicate_top_row(g)),
'duplicate_left_column+duplicate_right_column': lambda g: duplicate_right_column(duplicate_left_column(g)),
'duplicate_top_row+duplicate_bottom_row+duplicate_left_column+duplicate_right_column': lambda g: duplicate_right_column(duplicate_left_column(duplicate_bottom_row(duplicate_top_row(g)))),
'compress': lambda g: compress(g)
}
