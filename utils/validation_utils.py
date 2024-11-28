import ARC_gym.utils.visualization as viz

def same_shape(a, b):
    if len(a) == len(b) and len(a[0]) == len(b[0]):
        return True
    else:
        return False

def is_same_gridset(grids1, grids2):
    for idx in range(len(grids1)):
        if grids1[idx] != grids2[idx]:
            return False

    return True

def is_identity(list_of_lists, list_of_grids):
    for a, _ in list_of_lists:
        if not is_same_gridset(a, list_of_grids):
            return False

    return True

def check_color_redundancies(prim_name, colors_used, prims):

    if prims.is_color_related(prim_name):
        # validate choice of color-related primitive to reduce redundancies
        if prim_name.startswith("set_fg_color"):
            for cprim in colors_used:
                if cprim.startswith("set_fg_color"):
                    # Rule #1: multiple set_fg_color usages are redundant.
                    return False

            # Rule #2: there is one or more color_swap usage(s) and none of them include
            #  the color parameter 0.
            color_swap_uses = 0
            color_swap_zero = False
            for cprim in colors_used:
                if cprim.startswith("color_swap"):
                    color_swap_uses += 1

                    if '0' in cprim:
                        color_swap_zero = True

            if color_swap_uses > 0 and not color_swap_zero:
                return False

        elif prim_name.startswith("color_swap"):
            # Rule #3: color_swap_a_b is redundant if we previously have color_swap_x_a
            from_color = prim_name[11]
            for cprim in colors_used:
                if cprim.startswith("color_swap"):
                    to_color = prim_name[12]
                    if from_color == to_color:
                        return False

    return True

@staticmethod
def is_empty(a_list, prims):
    all_empty = 0
    for a in a_list:
        if prims.palette(a) == {0}:
            all_empty += 1

    if all_empty == len(a_list):
        return True
    else:
        return False

@staticmethod
def is_valid_shape(a_list, grid_size):
    for a in a_list:
        if len(a) > grid_size or len(a) == 0 or len(a[0]) > grid_size or len(a[0]) == 0:
            return False

        ncols = len(a[0])
        for a_row in a:
            if len(a_row) != ncols:
                return False

    return True
