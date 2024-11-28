# The following primitives are a subset from Michael's Hodel's DSL that consists of grid-to-grid transformations only.
# Michael Hodel's DSL: https://github.com/michaelhodel/arc-dsl

import inspect
from typing import List
import numpy as np
import math
import copy
import ARC_gym.utils.visualization as viz


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

class Grid:

    def __init__(self, cells, ul_x=0, ul_y=0, prev_width=None, prev_height=None, prev_ul_x=None, prev_ul_y=None, mask=[]):
        self.height = len(cells)
        self.width = len(cells[0])

        if prev_height is None:
            self.orig_height = self.height
        else:
            self.orig_height = prev_height

        if prev_width is None:
            self.orig_width = self.width
        else:
            self.orig_width = prev_width

        if prev_ul_x is None:
            self.orig_ul_x = ul_x
        else:
            self.orig_ul_x = prev_ul_x

        if prev_ul_y is None:
            self.orig_ul_y = ul_y
        else:
            self.orig_ul_y = prev_ul_y

        self.mask = mask
        self.ul_x = int(ul_x)
        self.ul_y = int(ul_y)

        self.pixels = self.from_grid(cells)  # pixels is a list of [(x, y, color)]

    def from_grid(self, cells):
        pixels = []
        for y, row in enumerate(cells):
            for x, color in enumerate(row):
                pixels.append((int(self.ul_x + x), int(self.ul_y + y), int(color)))
        return pixels

    @property
    def cells(self):
        return self.to_grid()
    
    def to_grid(self):
        if not self.pixels:
            return tuple()

        # Find the maximum x and y coordinates
        max_x = int(max(pixel[0] for pixel in self.pixels))
        max_y = int(max(pixel[1] for pixel in self.pixels))

        # Create a 2D list filled with zeros (black)
        grid = [[0 for _ in range(max_x - self.ul_x + 1)] for _ in range(max_y - self.ul_y + 1)]

        # Fill in the colors from self.pixels
        for x, y, color in self.pixels:
            grid[y - self.ul_y][x - self.ul_x] = color

        # Convert the 2D list to a tuple of tuples
        return tuple(tuple(row) for row in grid)

    def get_shifted_cells(self):
        # Get original cells and dimensions
        cells = self.cells
        width = self.width
        height = self.height
        
        # Get shift amounts
        x_shift = self.ul_x
        y_shift = self.ul_y
        
        # Create empty grid filled with background color (0)
        result = [[0 for _ in range(width)] for _ in range(height)]
        
        # Copy cells to shifted position
        for y in range(len(cells)):
            if y + y_shift < 0 or y + y_shift >= height:
                continue
            for x in range(len(cells[y])):
                if x + x_shift < 0 or x + x_shift >= width:
                    continue
                result[y + y_shift][x + x_shift] = cells[y][x]
                
        # Convert to tuple of tuples
        return tuple(tuple(row) for row in result)

    def __str__(self):
        """
        Returns a string representation of the Grid instance.
        """
        header = f"Upper-left coords: ({self.ul_x}, {self.ul_y})\n"
        header += f"Height: {self.height}, Width: {self.width}\n"
        header += "Cells:\n"
        
        # Convert cells to a formatted string
        cells_str = '\n'.join([' '.join(map(str, row)) for row in self.cells])
        
        return header + cells_str

    def __repr__(self):
        """
        Returns a string representation of the Grid instance.
        This method is used when the object is represented in the interactive shell.
        """
        return self.__str__()


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
    if name not in prim_indices:
        print("==> ERROR: primitive %s does not exist!" % name)

    return prim_indices[name]

def inverse_lookup(idx):
    for key, val in prim_indices.items():
        if val == idx:
            return key

    print("==> ERROR: primitive index %i does not exist!" % idx)

# ================================================================== New Objectness Primitives ===============================================================================

# This formulation of get_objects only gets 4-connected same-color pixels as objects.
def get_objects2(grid: Grid, bg_color=None) -> List[Grid]:
    nrows = len(grid.cells)
    ncols = len(grid.cells[0])

    if bg_color is None:
        background_color = get_bg_color(grid, nrows, ncols)
    else:
        background_color = bg_color

    objects = []
    visited = set()
    
    def dfs(i, j, current_object, color):
        if (i, j) in visited or i < 0 or i >= nrows or j < 0 or j >= ncols or grid.cells[i][j] != color or grid.cells[i][j] == background_color:
            return
        
        visited.add((i, j))
        current_object.append((i, j))
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Only consider 4-connected neighbors
            dfs(i + di, j + dj, current_object, color)
    
    for i in range(nrows):
        for j in range(ncols):
            if (i, j) not in visited and grid.cells[i][j] != background_color:
                current_object = []
                dfs(i, j, current_object, grid.cells[i][j])
                
                if len(current_object) > 2:  # Only process objects with more than 2 pixels
                    # Get the rectangular area around the object's pixels
                    min_i = min(px[0] for px in current_object)
                    max_i = max(px[0] for px in current_object)
                    min_j = min(px[1] for px in current_object)
                    max_j = max(px[1] for px in current_object)
                    
                    object_cells = []
                    for y in range(min_i, max_i + 1):
                        row = []
                        for x in range(min_j, max_j + 1):
                            if (y, x) in current_object:
                                row.append(grid.cells[y][x])
                            else:
                                row.append(background_color)

                        object_cells.append(tuple(row))
                    
                    object_grid = Grid(tuple(object_cells), min_j, min_i)
                    objects.append(object_grid)
    
    return objects

def get_bg_color(grid: Grid, nrows, ncols) -> int:
    color_counts = {}
    edge_colors = set()
    for i, row in enumerate(grid.cells):
        for j, cell in enumerate(row):
            color_counts[cell] = color_counts.get(cell, 0) + 1
            if i == 0 or i == nrows - 1 or j == 0 or j == ncols - 1:
                edge_colors.add(cell)
    
    # Bias towards black (0) as background color
    if 0 in color_counts:
        color_counts[0] *= 1.5  # Increase weight for black
    
    # If black is an edge color, further increase its likelihood
    if 0 in edge_colors:
        color_counts[0] *= 1.2
    
    background_color = max(color_counts, key=color_counts.get)
    
    # If black is present and close in count to the max, choose black
    if 0 in color_counts and color_counts[0] >= 0.8 * color_counts[background_color]:
        background_color = 0

    return background_color

# This formulation of get_objects only gets 8-connected same-color pixels as objects. Includes notion of containment.
def get_objects3(grid: Grid) -> List[Grid]:
    nrows = len(grid.cells)
    ncols = len(grid.cells[0])

    background_color = 0

    objects = []
    visited = set()
    
    def dfs(i, j, current_object, color):
        if (i, j) in visited or i < 0 or i >= nrows or j < 0 or j >= ncols or grid.cells[i][j] != color or grid.cells[i][j] == background_color:
            return
        
        visited.add((i, j))
        current_object.append((i, j))
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:  # Consider 8-connected neighbors
            dfs(i + di, j + dj, current_object, color)
    
    for i in range(nrows):
        for j in range(ncols):
            if (i, j) not in visited and grid.cells[i][j] != background_color:
                current_object = []
                dfs(i, j, current_object, grid.cells[i][j])
                
                if current_object:
                    # Get the rectangular area around the object's pixels
                    min_i = min(px[0] for px in current_object)
                    max_i = max(px[0] for px in current_object)
                    min_j = min(px[1] for px in current_object)
                    max_j = max(px[1] for px in current_object)
                    
                    # Only keep objects whose width and height are both 3 or greater
                    if (max_i - min_i + 1) >= 3 or (max_j - min_j + 1) >= 3 and ((max_i - min_i + 1) >= 2 and (max_j - min_j + 1) >= 2):
                        object_cells = []
                        foreground_pixels = 0
                        total_pixels = (max_i - min_i + 1) * (max_j - min_j + 1)
                        
                        for y in range(min_i, max_i + 1):
                            row = []
                            for x in range(min_j, max_j + 1):
                                if (y, x) in current_object:
                                    row.append(grid.cells[y][x])
                                    foreground_pixels += 1
                                else:
                                    row.append(background_color)

                            object_cells.append(tuple(row))
                        
                        # Check if foreground pixel density is above a certain threshold
                        if foreground_pixels / total_pixels > 0.2:
                            object_grid = Grid(tuple(object_cells), min_j, min_i)
                            objects.append(object_grid)
                    
    # Integrate fully contained objects into bigger objects
    integrated_objects = []
    for i, obj in enumerate(objects):
        is_contained = False
        for j, other_obj in enumerate(objects):
            if i != j and is_fully_contained(obj, other_obj):
                is_contained = True
                # Add the content of the inner object to the outer object
                for pixel in obj.pixels:
                    x, y, color = pixel
                    outer_x = obj.ul_x - other_obj.ul_x + x
                    outer_y = obj.ul_y - other_obj.ul_y + y
                    if 0 <= outer_x < other_obj.width and 0 <= outer_y < other_obj.height:
                        if color != background_color:
                            other_obj.pixels = [p for p in other_obj.pixels if p[:2] != (outer_x, outer_y)]
                            other_obj.pixels.append((outer_x, outer_y, color))
                other_obj.pixels.sort(key=lambda p: (p[1], p[0]))  # Sort by y, then x
                break

        if not is_contained:
            integrated_objects.append(obj)
    
    objects = integrated_objects
    return objects

def is_fully_contained(inner: Grid, outer: Grid) -> bool:
    # Check if the outer object's rectangle area entirely overlaps the inner object
    if (inner.ul_x < outer.ul_x or
        inner.ul_y < outer.ul_y or
        inner.ul_x + inner.width > outer.ul_x + outer.width or
        inner.ul_y + inner.height > outer.ul_y + outer.height):
        return False

    # Check if the foreground pixels of the outer object form a rectangle that encompasses the inner object
    top_edge = np.zeros(inner.width)
    bottom_edge = np.zeros(inner.width)
    right_edge = np.zeros(inner.height)
    left_edge = np.zeros(inner.height)

    # Count top edge containment pixels
    for x in range(inner.ul_x, inner.ul_x + inner.width):
        y = inner.ul_y
        outer_x = x - outer.ul_x
        outer_y = y - outer.ul_y
        for out_y in range(outer_y):
            if outer.cells[out_y][outer_x] != 0:
                top_edge[x - inner.ul_x] = 1
                break

    # Count bottom edge containment pixels
    for x in range(inner.ul_x, inner.ul_x + inner.width):
        y = inner.ul_y + inner.height
        outer_x = x - outer.ul_x
        outer_y = y - outer.ul_y
        for out_y in range(outer_y, outer.height):
            if outer.cells[out_y][outer_x] != 0:
                bottom_edge[x - inner.ul_x] = 1
                break

    # Count left edge containment pixels
    for y in range(inner.ul_y, inner.ul_y + inner.height):
        x = inner.ul_x
        outer_x = x - outer.ul_x
        outer_y = y - outer.ul_y
        for out_x in range(outer_x):
            if outer.cells[outer_y][out_x] != 0:
                left_edge[y - inner.ul_y] = 1
                break

    # Count right edge containment pixels
    for y in range(inner.ul_y, inner.ul_y + inner.height):
        x = inner.ul_x + inner.width
        outer_x = x - outer.ul_x
        outer_y = y - outer.ul_y
        for out_x in range(outer_x, outer.width):
            if outer.cells[outer_y][out_x] != 0:
                right_edge[y - inner.ul_y] = 1
                break

    return np.mean(top_edge) > 0.5 and np.mean(bottom_edge) > 0.5 and np.mean(left_edge) > 0.5 and np.mean(right_edge) > 0.5

def find_objects(grid: Grid, w: int, h: int) -> List[Grid]:
    nrows = len(grid.cells)
    ncols = len(grid.cells[0])
    background_color = get_bg_color(grid, nrows, ncols)
    objects = []

    def calculate_density(obj):
        return sum(1 for row in obj.cells for cell in row if cell != background_color) / (w * h)

    def objects_overlap(obj1, obj2):
        x1, y1 = obj1.ul_x, obj1.ul_y
        x2, y2 = obj2.ul_x, obj2.ul_y
        return not (x1 + w <= x2 or x2 + w <= x1 or y1 + h <= y2 or y2 + h <= y1)

    for i in range(nrows - h + 1):
        for j in range(ncols - w + 1):
            object_cells = []
            for y in range(i, i + h):
                row = []
                for x in range(j, j + w):
                    row.append(grid.cells[y][x])
                object_cells.append(tuple(row))
            
            object_grid = Grid(tuple(object_cells), j, i)
            density = calculate_density(object_grid)

            if density > 0.25:
                objects.append((object_grid, density))

    # Sort objects by density in descending order
    sorted_objects = sorted(objects, key=lambda x: x[1], reverse=True)

    # Remove overlapping objects with smaller density
    final_objects = []
    for obj, density in sorted_objects:
        if not any(objects_overlap(obj, existing_obj) for existing_obj, _ in final_objects):
            final_objects.append((obj, density))

    def get_non_object_pixels(grid: Grid, objects: List[Grid]) -> List[tuple]:
        nrows = len(grid.cells)
        ncols = len(grid.cells[0])
        background_color = get_bg_color(grid, nrows, ncols)
        
        # Create a set of all object pixels
        object_pixels = set()
        for obj, _ in objects:
            for y in range(obj.ul_y, obj.ul_y + obj.height):
                for x in range(obj.ul_x, obj.ul_x + obj.width):
                    if 0 <= y < nrows and 0 <= x < ncols:
                        object_pixels.add((x, y))
        
        # Identify foreground pixels not in any object
        non_object_pixels = []
        for i in range(nrows):
            for j in range(ncols):
                if grid.cells[i][j] != background_color and (j, i) not in object_pixels:
                    non_object_pixels.append((j, i, grid.cells[i][j]))
        
        return non_object_pixels

    # Add non-object pixels as single-pixel objects
    non_object_pixels = get_non_object_pixels(grid, final_objects)
    attempts = 0
    while len(non_object_pixels) > 0 and attempts < 10:
        attempts += 1
        for pixel in non_object_pixels:
            x, y, color = pixel

            # Function to calculate Euclidean distance between two points
            def distance(p1, p2):
                return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
            
            # Find the nearest object
            nearest_obj = None
            min_distance = float('inf')
            
            for obj, _ in final_objects:
                # Calculate distance to the object's top-left corner
                dist = distance((x, y), (obj.ul_x, obj.ul_y))
                if dist < min_distance:
                    min_distance = dist
                    nearest_obj = obj

            if nearest_obj:
                # Determine the axis of movement
                if x < nearest_obj.ul_x:
                    dx = -1
                    dy = 0
                elif x > nearest_obj.ul_x + w:
                    dx = 1
                    dy = 0
                elif y < nearest_obj.ul_y:
                    dx = 0
                    dy = -1
                else:
                    dx = 0
                    dy = 1

                # Move adjacent objects
                for obj, _ in final_objects:
                    if obj != nearest_obj:
                        if dx == -1 and obj.ul_x == nearest_obj.ul_x+w:
                            obj.ul_x += round(dx)
                            obj.ul_y += round(dy)
                        elif dx == 1 and obj.ul_x == nearest_obj.ul_x-w:
                            obj.ul_x += round(dx)
                            obj.ul_y += round(dy)
                        elif dy == -1  and obj.ul_y == nearest_obj.ul_y+h:
                            obj.ul_x += round(dx)
                            obj.ul_y += round(dy)
                        elif dy == 1 and obj.ul_y == nearest_obj.ul_y-h:
                            obj.ul_x += round(dx)
                            obj.ul_y += round(dy)
                        else:
                            continue

                        # Update obj.pixels based on new position
                        obj.pixels = []
                        for y in range(obj.ul_y, obj.ul_y + h):
                            for x in range(obj.ul_x, obj.ul_x + w):
                                if 0 <= y < grid.height and 0 <= x < grid.width:
                                    color = grid.cells[y][x]
                                    obj.pixels.append((x, y, color))
                        break

                # Move the nearest object one step towards the pixel
                new_x = nearest_obj.ul_x + round(dx)
                new_y = nearest_obj.ul_y + round(dy)
                
                # Update the position of the nearest object
                nearest_obj.ul_x = new_x
                nearest_obj.ul_y = new_y

                # Update obj.pixels based on new position
                nearest_obj.pixels = []
                for y in range(nearest_obj.ul_y, nearest_obj.ul_y + h):
                    for x in range(nearest_obj.ul_x, nearest_obj.ul_x + w):
                        if 0 <= y < grid.height and 0 <= x < grid.width:
                            color = grid.cells[y][x]
                            nearest_obj.pixels.append((x, y, color))

        non_object_pixels = get_non_object_pixels(grid, final_objects)

    return [obj for obj, _ in final_objects if obj.cells]

def extract_grid(grid: Grid, x_offset: int, y_offset:int, dims: int, bg_color: int, with_padding=False) -> List[Grid]:
    sub_grids = []
    nrows = len(grid.cells)
    ncols = len(grid.cells[0])
    total_density = 0

    def has_empty_edge(sub_grid):
        cells = sub_grid.cells
        return (all(cell == bg_color for cell in cells[0]) or  # Top edge
                all(cell == bg_color for cell in cells[-1]) or  # Bottom edge
                all(row[0] == bg_color for row in cells) or  # Left edge
                all(row[-1] == bg_color for row in cells))  # Right edge

    step = dims + 1 if with_padding else dims
    for i in range(y_offset, nrows, step):
        for j in range(x_offset, ncols, step):
            if i + dims > nrows or j + dims > ncols:
                continue
            sub_grid_cells = []
            for y in range(i, i + dims):
                row = []
                for x in range(j, j + dims):
                    row.append(grid.cells[y][x])
                sub_grid_cells.append(tuple(row))
            sub_grid = Grid(tuple(sub_grid_cells), j, i)
            
            # Calculate foreground pixel density
            total_pixels = dims * dims
            fg_pixels = sum(1 for row in sub_grid.cells for cell in row if cell != bg_color)
            density = fg_pixels / total_pixels

            # Only add sub_grid if density is > 0.4 and it does not have an empty edge
            if density > 0.4 and not has_empty_edge(sub_grid):
                sub_grids.append(sub_grid)
                total_density += density

    average_density = total_density / len(sub_grids) if sub_grids else 0

    return sub_grids, average_density


def get_objects6(grid: Grid) -> List[Grid]:
    bg_color = max((color for row in grid.cells for color in row), key=lambda x: sum(1 for row in grid.cells for cell in row if cell == x))
    dims_range = [6, 5, 4, 3]

    scores = []
    obj_lists = []

    max_score = float('-inf')
    total_fg_pixels = sum(1 for row in grid.cells for cell in row if cell != bg_color)

    for dim in dims_range:
        for x_offset in range(dim):
            if max_score == 1:
               break

            for y_offset in range(dim):
                if max_score == 1:
                    break

                obj_list, density = extract_grid(grid, x_offset, y_offset, dim, bg_color)

                # Calculate the number of foreground pixels in the objects
                obj_fg_pixels = sum(1 for obj in obj_list for row in obj.cells for cell in row if cell != bg_color)
                
                # Calculate the penalty
                penalty = total_fg_pixels - obj_fg_pixels

                # Calculate the score (density minus penalty)
                score = density - (penalty / total_fg_pixels)

                #print(f"Dim: {dim}, x_offset: {x_offset}, y_offset: {y_offset}, score: {score:.4f}")
                if score > max_score + 0.01:
                    max_score = score
                    # print("New max score! dim = %i, x_offset = %i, y_offset = %i, score: %.4f, num objects: %i" % (
                    #     dim,
                    #     x_offset,
                    #     y_offset,
                    #     score,
                    #     len(obj_list)
                    # ))

                scores.append(score)
                obj_lists.append(obj_list)

        if max_score == 1:
            break

    if max_score < 1:
        for dim in dims_range:
            for x_offset in range(dim):
                if max_score == 1:
                    break

                for y_offset in range(dim):
                    if max_score == 1:
                        break

                    obj_list, density = extract_grid(grid, x_offset, y_offset, dim, bg_color, with_padding=True)

                    # Calculate the number of foreground pixels in the objects
                    obj_fg_pixels = sum(1 for obj in obj_list for row in obj.cells for cell in row if cell != bg_color)
                    
                    # Calculate the penalty
                    penalty = total_fg_pixels - obj_fg_pixels

                    # Calculate the score (density minus penalty)
                    score = density - (penalty / total_fg_pixels)

                    #print(f"Dim: {dim}, x_offset: {x_offset}, y_offset: {y_offset}, score: {score:.4f}")
                    if score > max_score + 0.01:
                        max_score = score
                        # print("WITH PADDING: New max score! dim = %i, x_offset = %i, y_offset = %i, score: %.4f, num objects: %i" % (
                        #     dim,
                        #     x_offset,
                        #     y_offset,
                        #     score,
                        #     len(obj_list)
                        # ))

                    scores.append(score)
                    obj_lists.append(obj_list)

                    if max_score == 1:
                        break

            if max_score == 1:
                break

    # Find the index of the highest score
    best_index = scores.index(max(scores))
    
    # Get the object list corresponding to the highest score
    best_obj_list = obj_lists[best_index]

    # print(f"Best score: {max(scores):.4f}")
    # print(f"Number of objects in best list: {len(best_obj_list)}")
    return best_obj_list

# This formulation of get_objects only gets 8-connected non-bg pixels as objects, regardless of color.
def get_objects5(grid: Grid) -> List[Grid]:
    nrows = len(grid.cells)
    ncols = len(grid.cells[0])

    background_color = get_bg_color(grid, nrows, ncols)
    
    objects = []
    visited = set()
    
    def dfs(i, j, current_object, color):
        if (i, j) in visited or i < 0 or i >= nrows or j < 0 or j >= ncols or grid.cells[i][j] == background_color:
            return
        
        visited.add((i, j))
        current_object.append((i, j))
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:  # Consider 8-connected neighbors
            dfs(i + di, j + dj, current_object, color)
    
    for i in range(nrows):
        for j in range(ncols):
            if (i, j) not in visited and grid.cells[i][j] != background_color:
                current_object = []
                dfs(i, j, current_object, grid.cells[i][j])
                
                if current_object:
                    # Get the rectangular area around the object's pixels
                    min_i = min(px[0] for px in current_object)
                    max_i = max(px[0] for px in current_object)
                    min_j = min(px[1] for px in current_object)
                    max_j = max(px[1] for px in current_object)
                    
                    object_cells = []
                    for y in range(min_i, max_i + 1):
                        row = []
                        for x in range(min_j, max_j + 1):
                            if (y, x) in current_object:
                                row.append(grid.cells[y][x])
                            else:
                                row.append(background_color)

                        object_cells.append(tuple(row))
                    
                    object_grid = Grid(tuple(object_cells), min_j, min_i)
                    objects.append(object_grid)

    return objects

# Finds uniform rectangular frames amidst noisy backgrounds.
def get_objects1(grid: Grid) -> List[Grid]:
    def is_uniform_rect_frame(subgrid, bg_color):
        if not subgrid or len(subgrid) < 3 or len(subgrid[0]) < 3:
            return False
        frame_color = subgrid[0][0]
        if frame_color == bg_color:
            return False
        
        # Check top and bottom rows
        if not all(cell == frame_color for cell in subgrid[0]) or \
           not all(cell == frame_color for cell in subgrid[-1]):
            return False
        
        # Check left and right columns
        return all(row[0] == frame_color and row[-1] == frame_color for row in subgrid[1:-1])

    bg_color = 0
    
    objects = []
    cells = grid.cells  # Cache grid.cells
    height, width = len(cells), len(cells[0])  # Cache dimensions
    
    # Pre-compute all possible frame sizes
    frame_sizes = [(h, w) for h in range(3, height + 1) for w in range(3, width + 1)]
    
    for y in range(height):
        for x in range(width):
            for h, w in frame_sizes:
                if y + h > height or x + w > width:
                    continue
                subgrid = [row[x:x+w] for row in cells[y:y+h]]
                if is_uniform_rect_frame(subgrid, bg_color):
                    objects.append(Grid(subgrid, x, y))

    # Remove overlapping objects, keeping the largest one
    objects.sort(key=lambda obj: obj.width * obj.height, reverse=True)
    final_objects = []
    for obj in objects:
        if not any(
            (obj.ul_x < other.ul_x + other.width and
             obj.ul_x + obj.width > other.ul_x and
             obj.ul_y < other.ul_y + other.height and
             obj.ul_y + obj.height > other.ul_y)
            for other in final_objects
        ):
            final_objects.append(obj)

    return final_objects

# # Finds dense rectangular frames (color doesn't need to be uniform) amidst noisy backgrounds.
# def get_objects4(grid: Grid) -> List[Grid]:
#     def is_dense_rect_frame(subgrid, bg_color):
#         if not subgrid.cells or len(subgrid.cells) < 2 or len(subgrid.cells[0]) < 2:
#             return False
       
#         # Check top and bottom rows
#         if any(cell == bg_color for cell in subgrid.cells[0]) or \
#            any(cell == bg_color for cell in subgrid.cells[-1]):
#             return False
        
#         # Check left and right columns
#         for row in subgrid.cells[1:-1]:
#             if row[0] == bg_color or row[-1] == bg_color:
#                 return False
        
#         return True

#     bg_color = 0
    
#     objects = []
#     for y in range(grid.height):
#         for x in range(grid.width):
#             for h in range(1, grid.height - y + 1):
#                 for w in range(1, grid.width - x + 1):
#                     subgrid = Grid([row[x:x+w] for row in grid.cells[y:y+h]], x, y)
#                     if is_dense_rect_frame(subgrid, bg_color) and subgrid.width >= 3 and subgrid.height >= 3:
#                         objects.append(subgrid)

#     # Remove overlapping objects, keeping the largest one
#     objects.sort(key=lambda obj: obj.width * obj.height, reverse=True)
#     final_objects = []
#     for obj in objects:
#         if not any(
#             (obj.ul_x < other.ul_x + other.width and
#              obj.ul_x + obj.width > other.ul_x and
#              obj.ul_y < other.ul_y + other.height and
#              obj.ul_y + obj.height > other.ul_y)
#             for other in final_objects
#         ):
#             final_objects.append(obj)

#     return final_objects

# This formulation of get_objects only gets 4-connected non-bg pixels as objects, regardless of color.
def get_objects4(grid: Grid) -> List[Grid]:
    nrows = len(grid.cells)
    ncols = len(grid.cells[0])

    background_color = get_bg_color(grid, nrows, ncols)
    
    objects = []
    visited = set()
    
    def dfs(i, j, current_object, color):
        if (i, j) in visited or i < 0 or i >= nrows or j < 0 or j >= ncols or grid.cells[i][j] == background_color:
            return
        
        visited.add((i, j))
        current_object.append((i, j))
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Consider 4-connected neighbors
            dfs(i + di, j + dj, current_object, color)
    
    for i in range(nrows):
        for j in range(ncols):
            if (i, j) not in visited and grid.cells[i][j] != background_color:
                current_object = []
                dfs(i, j, current_object, grid.cells[i][j])
                
                if current_object:
                    # Get the rectangular area around the object's pixels
                    min_i = min(px[0] for px in current_object)
                    max_i = max(px[0] for px in current_object)
                    min_j = min(px[1] for px in current_object)
                    max_j = max(px[1] for px in current_object)
                    
                    object_cells = []
                    for y in range(min_i, max_i + 1):
                        row = []
                        for x in range(min_j, max_j + 1):
                            if (y, x) in current_object:
                                row.append(grid.cells[y][x])
                            else:
                                row.append(background_color)

                        object_cells.append(tuple(row))
                    
                    object_grid = Grid(tuple(object_cells), min_j, min_i)
                    objects.append(object_grid)

    return objects

def compress_objects_linear(grid_list: List[Grid]) -> Grid:
    if not grid_list:
        return Grid([[]])

    # Determine orientation based on largest span
    min_x = min(grid.ul_x for grid in grid_list)
    max_x = max(grid.ul_x + grid.width for grid in grid_list)
    min_y = min(grid.ul_y for grid in grid_list)
    max_y = max(grid.ul_y + grid.height for grid in grid_list)

    horizontal_span = max_x - min_x
    vertical_span = max_y - min_y

    is_horizontal = horizontal_span >= vertical_span

    # Sort grids based on their position
    if is_horizontal:
        sorted_grids = sorted(grid_list, key=lambda g: g.ul_x)
    else:
        sorted_grids = sorted(grid_list, key=lambda g: g.ul_y)

    # Calculate dimensions for the new grid
    if is_horizontal:
        new_width = sum(grid.width for grid in sorted_grids)
        new_height = max(grid.height for grid in sorted_grids)
    else:
        new_width = max(grid.width for grid in sorted_grids)
        new_height = sum(grid.height for grid in sorted_grids)

    # Create a new grid
    new_cells = [[0 for _ in range(new_width)] for _ in range(new_height)]

    # Copy grids to their respective positions
    current_pos = 0
    for grid in sorted_grids:
        if is_horizontal:
            for y in range(min(grid.height, new_height)):
                for x in range(grid.width):
                    if current_pos + x < new_width:
                        new_cells[y][current_pos + x] = grid.cells[y][x]
            current_pos += grid.width
        else:
            for y in range(grid.height):
                for x in range(min(grid.width, new_width)):
                    if current_pos + y < new_height:
                        new_cells[current_pos + y][x] = grid.cells[y][x]
            current_pos += grid.height

    # Convert the 2D list to a tuple of tuples
    new_grid_tuple = tuple(tuple(row) for row in new_cells)

    # Convert the tuple of tuples to a Grid object
    return Grid(new_grid_tuple)

def compress_objects_quad(grid_list: List[Grid]) -> Grid:
    if not grid_list:
        return Grid([[]])

    # Special case: If there are exactly 4 objects, place them in the corners
    if len(grid_list) == 4:
        # Calculate the center point of all grids
        center_x = sum((g.ul_x + g.width/2) for g in grid_list) / len(grid_list)
        center_y = sum((g.ul_y + g.height/2) for g in grid_list) / len(grid_list)

        top_left = min(grid_list, key=lambda g: (g.ul_x + g.width/2 - center_x) + (g.ul_y + g.height/2 - center_y))
        top_right = max(grid_list, key=lambda g: (g.ul_x + g.width/2 - center_x) - (g.ul_y + g.height/2 - center_y))
        bottom_left = min(grid_list, key=lambda g: (g.ul_x + g.width/2 - center_x) - (g.ul_y + g.height/2 - center_y))
        bottom_right = max(grid_list, key=lambda g: (g.ul_x + g.width/2 - center_x) + (g.ul_y + g.height/2 - center_y))

        # Calculate dimensions for the new grid
        max_width = max(top_left.width + top_right.width, bottom_left.width + bottom_right.width)
        max_height = max(top_left.height + bottom_left.height, top_right.height + bottom_right.height)

        # Create a new grid
        new_grid = [[0 for _ in range(max_width)] for _ in range(max_height)]

        # Copy grids to their respective corners
        def copy_grid(src, dest_x, dest_y):
            for y in range(src.height):
                for x in range(src.width):
                    new_grid[dest_y + y][dest_x + x] = src.cells[y][x]

        copy_grid(top_left, 0, 0)
        copy_grid(top_right, max_width - top_right.width, 0)
        copy_grid(bottom_left, 0, max_height - bottom_left.height)
        copy_grid(bottom_right, max_width - bottom_right.width, max_height - bottom_right.height)

        # Convert the 2D list to a tuple of tuples
        new_grid_tuple = tuple(tuple(row) for row in new_grid)

        # Convert the tuple of tuples to a Grid object
        return Grid(new_grid_tuple)

    # Calculate vector lengths from center to each grid
    def calculate_vector(grid, center_x, center_y):
        dx = (grid.ul_x + grid.width/2) - center_x
        dy = (grid.ul_y + grid.height/2) - center_y
        return (dx, dy)

    # Calculate the center point of all grids
    center_x = sum((g.ul_x + g.width/2) for g in grid_list) / len(grid_list)
    center_y = sum((g.ul_y + g.height/2) for g in grid_list) / len(grid_list)

    # Sort grids based on their vector lengths from center
    object_vectors = list((calculate_vector(grid, center_x, center_y) for grid in grid_list))
    vector_lengths = list((math.sqrt(vec[0]**2. + vec[1] ** 2) for vec in object_vectors))

    center_obj_idx = np.argmin(vector_lengths)

    # Re-calculate vectors relative to the center object's centerpoint.
    cobj = grid_list[center_obj_idx]

    object_vectors = list((calculate_vector(grid, cobj.ul_x + cobj.width/2, cobj.ul_y + cobj.height/2) for grid in grid_list))
    vector_lengths = list((math.sqrt(vec[0]**2. + vec[1] ** 2) for vec in object_vectors))

    # Define OctoTreeNode class to represent the octo-tree structure
    class OctoTreeNode:
        def __init__(self, obj_idx):
            self.obj_idx = obj_idx
            vec = object_vectors[obj_idx]
            self.dx = vec[0]
            self.dy = vec[1]
            self.vector_length = vector_lengths[obj_idx]
            self.children = {
                "right": None, "left": None, "up": None, "down": None,
                "top-right": None, "top-left": None, "bottom-right": None, "bottom-left": None
            }

    # Create the root node with the center object
    root = OctoTreeNode(center_obj_idx)

    # Function to insert a grid into the octo-tree
    def insert_into_tree(node, obj_idx):
        child = OctoTreeNode(obj_idx)

        import math

        def get_direction(angle):
            if -22.5 <= angle < 22.5:
                return "right"
            elif 22.5 <= angle < 67.5:
                return "top-right"
            elif 67.5 <= angle < 112.5:
                return "up"
            elif 112.5 <= angle < 157.5:
                return "top-left"
            elif 157.5 <= angle <= 180 or -180 <= angle < -157.5:
                return "left"
            elif -157.5 <= angle < -112.5:
                return "bottom-left"
            elif -112.5 <= angle < -67.5:
                return "down"
            elif -67.5 <= angle < -22.5:
                return "bottom-right"

        angle_to_center = math.degrees(math.atan2(-child.dy, child.dx))
        direction = get_direction(angle_to_center)

        if node.children[direction] is not None:
            # Calculate angle relative to the potential parent node
            relative_dx = child.dx - node.children[direction].dx
            relative_dy = child.dy - node.children[direction].dy
            angle_to_parent = math.degrees(math.atan2(-relative_dy, relative_dx))

            # If the angles differ significantly, consider the second-best direction
            if abs(angle_to_center - angle_to_parent) > 45:

                directions = ["right", "top-right", "up", "top-left", "left", "bottom-left", "down", "bottom-right"]
                second_best = min(directions, key=lambda d: abs(angle_to_parent - (directions.index(d) * 45)))
                if abs(angle_to_parent - (directions.index(second_best) * 45)) < abs(angle_to_parent - (directions.index(direction) * 45)):
                    direction = second_best
                    node.children[direction] = OctoTreeNode(obj_idx)
            else:
                if node.children[direction].vector_length < child.vector_length:
                    insert_into_tree(node.children[direction], obj_idx)
                else:
                    to_swap_idx = node.children[direction].obj_idx
                    node.children[direction] = child
                    insert_into_tree(child, to_swap_idx)
        else:
            node.children[direction] = OctoTreeNode(obj_idx)
        
    # Insert all other grids into the tree
    for obj_idx, obj in enumerate(grid_list):
        if obj_idx != center_obj_idx:
            insert_into_tree(root, obj_idx)

    # Process the OctoTree to generate the target grid
    def process_octotree(node, x, y, grid):
        if node is None:
            return

        # Copy the current object to the grid
        obj = grid_list[node.obj_idx]
        for i in range(obj.height):
            for j in range(obj.width):
                if y + i < len(grid) and x + j < len(grid[0]):
                    grid[y + i][x + j] = obj.cells[i][j]

        # Process children
        if node.children["right"]:
            process_octotree(node.children["right"], x + obj.width, y, grid)
        if node.children["left"]:
            process_octotree(node.children["left"], x - grid_list[node.children["left"].obj_idx].width, y, grid)
        if node.children["down"]:
            process_octotree(node.children["down"], x, y + obj.height, grid)
        if node.children["up"]:
            process_octotree(node.children["up"], x, y - grid_list[node.children["up"].obj_idx].height, grid)
        if node.children["bottom-right"]:
            process_octotree(node.children["bottom-right"], x + obj.width, y + obj.height, grid)
        if node.children["top-right"]:
            process_octotree(node.children["top-right"], x + obj.width, y - grid_list[node.children["top-right"].obj_idx].height, grid)
        if node.children["bottom-left"]:
            process_octotree(node.children["bottom-left"], x - grid_list[node.children["bottom-left"].obj_idx].width, y + obj.height, grid)
        if node.children["top-left"]:
            process_octotree(node.children["top-left"], x - grid_list[node.children["top-left"].obj_idx].width, y - grid_list[node.children["top-left"].obj_idx].height, grid)

    # Calculate the dimensions of the new grid
    max_x = max_y = min_x = min_y = 0

    def calculate_dimensions(node, x, y):
        nonlocal max_x, max_y, min_x, min_y
        if node is None:
            return

        obj = grid_list[node.obj_idx]
        max_x = max(max_x, x + obj.width)
        max_y = max(max_y, y + obj.height)
        min_x = min(min_x, x)
        min_y = min(min_y, y)

        for direction, child in node.children.items():
            if child:
                dx, dy = 0, 0
                if "right" in direction:
                    dx = obj.width
                elif "left" in direction:
                    dx = -grid_list[child.obj_idx].width
                if "down" in direction or "bottom" in direction:
                    dy = obj.height
                elif "up" in direction or "top" in direction:
                    dy = -grid_list[child.obj_idx].height
                calculate_dimensions(child, x + dx, y + dy)

    calculate_dimensions(root, 0, 0)

    # Create the new grid with calculated dimensions
    new_grid = [[0 for _ in range(max_x - min_x)] for _ in range(max_y - min_y)]

    # Process the OctoTree to fill the new grid
    process_octotree(root, -min_x, -min_y, new_grid)

    # Convert the 2D list to a tuple of tuples
    new_grid_tuple = tuple(tuple(row) for row in new_grid)

    # Convert the tuple of tuples to a Grid object
    return Grid(new_grid_tuple)

def compress_objects_quad_pad(grid_list: List[Grid]) -> Grid:
    if not grid_list:
        return Grid([[]])

    # Special case: If there are exactly 4 objects, place them in the corners without outline padding
    if len(grid_list) == 4:
        # Sort grids based on their upper-left coordinates
        sorted_grids = sorted(grid_list, key=lambda g: (g.ul_x, g.ul_y))
        
        # Assign corners based on relative positions
        top_left = sorted_grids[0]
        bottom_left = sorted_grids[1]
        top_right = sorted_grids[2]
        bottom_right = sorted_grids[3]

        # Calculate dimensions for the new grid with padding only between objects
        max_width = max(top_left.width + top_right.width, bottom_left.width + bottom_right.width) + 1  # +1 for padding between objects
        max_height = max(top_left.height + bottom_left.height, top_right.height + bottom_right.height) + 1  # +1 for padding between objects

        # Create a new grid
        new_grid = [[0 for _ in range(max_width)] for _ in range(max_height)]

        # Copy grids to their respective corners
        def copy_grid(src, dest_x, dest_y):
            for y in range(src.height):
                for x in range(src.width):
                    new_grid[dest_y + y][dest_x + x] = src.cells[y][x]

        copy_grid(top_left, 0, 0)
        copy_grid(top_right, max_width - top_right.width, 0)
        copy_grid(bottom_left, 0, max_height - bottom_left.height)
        copy_grid(bottom_right, max_width - bottom_right.width, max_height - bottom_right.height)

        # Convert the 2D list to a tuple of tuples
        new_grid_tuple = tuple(tuple(row) for row in new_grid)

        # Convert the tuple of tuples to a Grid object
        return Grid(new_grid_tuple)

    # Calculate vector lengths from center to each grid
    def calculate_vector(grid, center_x, center_y):
        dx = (grid.ul_x + grid.width/2) - center_x
        dy = (grid.ul_y + grid.height/2) - center_y
        return (dx, dy)

    # Calculate the center point of all grids
    center_x = sum((g.ul_x + g.width/2) for g in grid_list) / len(grid_list)
    center_y = sum((g.ul_y + g.height/2) for g in grid_list) / len(grid_list)

    # Sort grids based on their vector lengths from center
    object_vectors = list((calculate_vector(grid, center_x, center_y) for grid in grid_list))
    vector_lengths = list((math.sqrt(vec[0]**2. + vec[1] ** 2) for vec in object_vectors))

    center_obj_idx = np.argmin(vector_lengths)

    # Re-calculate vectors relative to the center object's centerpoint.
    cobj = grid_list[center_obj_idx]
    object_vectors = list((calculate_vector(grid, cobj.ul_x + cobj.width/2, cobj.ul_y + cobj.height/2) for grid in grid_list))
    vector_lengths = list((math.sqrt(vec[0]**2. + vec[1] ** 2) for vec in object_vectors))

    # Define OctoTreeNode class to represent the octo-tree structure
    class OctoTreeNode:
        def __init__(self, obj_idx):
            self.obj_idx = obj_idx
            vec = object_vectors[obj_idx]
            self.dx = vec[0]
            self.dy = vec[1]
            self.vector_length = vector_lengths[obj_idx]
            self.children = {
                "right": None, "left": None, "up": None, "down": None,
                "top-right": None, "top-left": None, "bottom-right": None, "bottom-left": None
            }

    # Create the root node with the center object
    root = OctoTreeNode(center_obj_idx)

    # Function to insert a grid into the octo-tree
    def insert_into_tree(node, obj_idx):
        child = OctoTreeNode(obj_idx)

        import math

        def get_direction(angle):
            if -22.5 <= angle < 22.5:
                return "right"
            elif 22.5 <= angle < 67.5:
                return "top-right"
            elif 67.5 <= angle < 112.5:
                return "up"
            elif 112.5 <= angle < 157.5:
                return "top-left"
            elif 157.5 <= angle <= 180 or -180 <= angle < -157.5:
                return "left"
            elif -157.5 <= angle < -112.5:
                return "bottom-left"
            elif -112.5 <= angle < -67.5:
                return "down"
            elif -67.5 <= angle < -22.5:
                return "bottom-right"

        angle_to_center = math.degrees(math.atan2(-child.dy, child.dx))
        direction = get_direction(angle_to_center)
        if node.children[direction] is not None:
            # Calculate angle relative to the potential parent node
            relative_dx = child.dx - node.children[direction].dx
            relative_dy = child.dy - node.children[direction].dy
            angle_to_parent = math.degrees(math.atan2(-relative_dy, relative_dx))

            # If the angles differ significantly, consider the second-best direction
            if abs(angle_to_center - angle_to_parent) > 45:
                directions = ["right", "top-right", "up", "top-left", "left", "bottom-left", "down", "bottom-right"]
                second_best = min(directions, key=lambda d: abs(angle_to_parent - (directions.index(d) * 45)))
                if abs(angle_to_parent - (directions.index(second_best) * 45)) < abs(angle_to_parent - (directions.index(direction) * 45)):
                    direction = second_best

            if node.children[direction].vector_length < child.vector_length:
                insert_into_tree(node.children[direction], obj_idx)
            else:
                to_swap_idx = node.children[direction].obj_idx
                node.children[direction] = child
                insert_into_tree(child, to_swap_idx)

        else:
            node.children[direction] = OctoTreeNode(obj_idx)
        
    # Insert all other grids into the tree
    for obj_idx, obj in enumerate(grid_list):
        if obj_idx != center_obj_idx:
            insert_into_tree(root, obj_idx)

    # Process the OctoTree to generate the target grid
    def process_octotree(node, x, y, grid):
        if node is None:
            return

        # Copy the current object to the grid
        obj = grid_list[node.obj_idx]
        for i in range(obj.height):
            for j in range(obj.width):
                if y + i < len(grid) and x + j < len(grid[0]):
                    grid[y + i][x + j] = obj.cells[i][j]

        # Process children with padding only between objects
        if node.children["right"]:
            process_octotree(node.children["right"], x + obj.width + 1, y, grid)
        if node.children["left"]:
            process_octotree(node.children["left"], x - grid_list[node.children["left"].obj_idx].width - 1, y, grid)
        if node.children["down"]:
            process_octotree(node.children["down"], x, y + obj.height + 1, grid)
        if node.children["up"]:
            process_octotree(node.children["up"], x, y - grid_list[node.children["up"].obj_idx].height - 1, grid)
        if node.children["bottom-right"]:
            process_octotree(node.children["bottom-right"], x + obj.width + 1, y + obj.height + 1, grid)
        if node.children["top-right"]:
            process_octotree(node.children["top-right"], x + obj.width + 1, y - grid_list[node.children["top-right"].obj_idx].height - 1, grid)
        if node.children["bottom-left"]:
            process_octotree(node.children["bottom-left"], x - grid_list[node.children["bottom-left"].obj_idx].width - 1, y + obj.height + 1, grid)
        if node.children["top-left"]:
            process_octotree(node.children["top-left"], x - grid_list[node.children["top-left"].obj_idx].width - 1, y - grid_list[node.children["top-left"].obj_idx].height - 1, grid)

    # Calculate the dimensions of the new grid
    max_x = max_y = min_x = min_y = 0

    def calculate_dimensions(node, x, y):
        nonlocal max_x, max_y, min_x, min_y
        if node is None:
            return

        obj = grid_list[node.obj_idx]
        max_x = max(max_x, x + obj.width)
        max_y = max(max_y, y + obj.height)
        min_x = min(min_x, x)
        min_y = min(min_y, y)

        for direction, child in node.children.items():
            if child:
                dx, dy = 0, 0
                if "right" in direction:
                    dx = obj.width + 1
                elif "left" in direction:
                    dx = -grid_list[child.obj_idx].width - 1
                if "down" in direction or "bottom" in direction:
                    dy = obj.height + 1
                elif "up" in direction or "top" in direction:
                    dy = -grid_list[child.obj_idx].height - 1
                calculate_dimensions(child, x + dx, y + dy)

    calculate_dimensions(root, 0, 0)

    # Create the new grid with calculated dimensions without additional padding
    new_grid = [[0 for _ in range(max_x - min_x)] for _ in range(max_y - min_y)]

    # Process the OctoTree to fill the new grid
    process_octotree(root, -min_x, -min_y, new_grid)

    # Convert the 2D list to a tuple of tuples
    new_grid_tuple = tuple(tuple(row) for row in new_grid)

    # Convert the tuple of tuples to a Grid object
    new_grid = Grid(new_grid_tuple)

    return new_grid

def apply_to_grid(original_grid: Grid, obj_list: List[Grid]) -> Grid:
    # Create a copy of the original grid to work on
    target_cells = [list(row) for row in original_grid.cells]

    # Get the background color
    nrows = len(original_grid.cells)
    ncols = len(original_grid.cells[0])
    bg_color = get_bg_color(original_grid, nrows, ncols)

    # Go through each object in obj_list
    for obj in obj_list:
        # Get the upper-left coordinates of the object
        x, y = obj.orig_ul_x, obj.orig_ul_y

        # print("==> apply_to_grid: Processing object at (%i, %i)" % (x, y))
        # print("\tBlanking out previous object, bg_color = %i, width = %i, height = %i" % (bg_color, obj.orig_width, obj.orig_height))
        # First, blank out the previous object using the background color
        for i in range(obj.orig_height):
            for j in range(obj.orig_width):
                if (0 <= y + i < original_grid.height) and (0 <= x + j < original_grid.width):
                    target_cells[y + i][x + j] = bg_color

        # Then, paste the cells content of the object onto the target grid
        for pixel in obj.pixels:
            px, py, color = pixel
            target_x, target_y = px, py
            # Check if the target position is within the bounds of the target grid
            if (0 <= target_y < original_grid.height) and (0 <= target_x < original_grid.width):
                target_cells[target_y][target_x] = color

    # Convert the 2D list back to a tuple of tuples
    new_grid_tuple = tuple(tuple(row) for row in target_cells)

    # Return a new Grid object with the modified cells
    return Grid(new_grid_tuple)

def for_each(obj_list, func):
    output = []
    #print("==> Identified %i objects overall." % len(obj_list))
    for obj_idx, obj in enumerate(obj_list):

        out = func(obj)

        # print("Identified object:")
        # viz.draw_single_grid(obj.cells)

        output.append(out)

    return output

def shear_grid_left(grid: Grid) -> Grid:
    # Get the height of the grid
    bottom_y = max(y for _, y, _ in grid.pixels)

    # Create a new list to store the sheared pixels
    new_pixels = []

    # Iterate through each pixel
    for x, y, color in grid.pixels:
        # Calculate the offset for this row
        offset = bottom_y - y
        # Calculate the new x position after shearing
        new_x = x - offset

        # Add the pixel with its new position to the new_pixels list
        new_pixels.append((new_x, y, color))

    output_grid = Grid([[]], 
                       grid.ul_x, 
                       grid.ul_y, 
                       prev_width=grid.orig_width, 
                       prev_height=grid.orig_height,
                       prev_ul_x=grid.orig_ul_x,
                       prev_ul_y=grid.orig_ul_y)
    output_grid.pixels = new_pixels

    return output_grid

def shear_grid_right(grid: Grid) -> Grid:
    # Get the height of the grid
    bottom_y = max(y for _, y, _ in grid.pixels)

    # Create a new list to store the sheared pixels
    new_pixels = []

    # Iterate through each pixel
    for x, y, color in grid.pixels:
        # Calculate the offset for this row
        offset = bottom_y - y
        # Calculate the new x position after shearing
        new_x = x + offset

        # Add the pixel with its new position to the new_pixels list
        new_pixels.append((new_x, y, color))


    output_grid = Grid([[]], 
                       grid.ul_x, 
                       grid.ul_y, 
                       prev_width=grid.orig_width, 
                       prev_height=grid.orig_height,
                       prev_ul_x=grid.orig_ul_x,
                       prev_ul_y=grid.orig_ul_y)
    output_grid.pixels = new_pixels

    return output_grid

def shear_grid_zigzag(grid: Grid) -> Grid:
    # Get the height of the grid
    bottom_y = max(y for _, y, _ in grid.pixels)

    # Create a new list to store the sheared pixels
    new_pixels = []

    # Iterate through each pixel
    for x, y, color in grid.pixels:
        # Calculate the offset for this row
        offset = [0, -1, 0, 1][(bottom_y - y) % 4]

        # Calculate the new x position after shearing
        new_x = x + offset

        # Add the pixel with its new position to the new_pixels list
        new_pixels.append((new_x, y, color))


    output_grid = Grid([[]], 
                       grid.ul_x, 
                       grid.ul_y, 
                       prev_width=grid.orig_width, 
                       prev_height=grid.orig_height,
                       prev_ul_x=grid.orig_ul_x,
                       prev_ul_y=grid.orig_ul_y)
    output_grid.pixels = new_pixels

    return output_grid

def get_object_size(grid: Grid) -> int:
    return sum(1 for _, _, color in grid.pixels if color != 0)

def count(grid_list: List[Grid]) -> int:
    return len(grid_list)

def filter_largest(grid_list: List[Grid], value_list: List[int]) -> list[Grid]:
    if not grid_list or not value_list or len(grid_list) != len(value_list):
        return grid_list

    # Find the index of the maximum value
    max_index = value_list.index(max(value_list))

    # Create a new list with the largest object's pixels replaced by zero-color pixels
    filtered_list = []
    for i, grid in enumerate(grid_list):
        if i == max_index:
            # Replace all pixels with zero-color pixels
            new_pixels = [(x, y, 0) for x, y, _ in grid.pixels]
            new_grid = Grid([[]], grid.ul_x, grid.ul_y)
            new_grid.pixels = new_pixels
            new_grid.height = grid.height
            new_grid.width = grid.width
            filtered_list.append(new_grid)
        else:
            filtered_list.append(grid)

    return filtered_list

def keep_largest(grid_list: List[Grid], value_list: List[int]) -> Grid:
    if not grid_list or not value_list or len(grid_list) != len(value_list):
        return grid_list

    # Find the index of the maximum value
    max_index = value_list.index(max(value_list))

    return grid_list[max_index]

def filter_smallest(grid_list: List[Grid], value_list: List[int]) -> list[Grid]:
    if not grid_list or not value_list or len(grid_list) != len(value_list):
        return grid_list

    # Find the index of the minimum value
    min_index = value_list.index(min(value_list))

    # Create a new list with the smallest object's pixels replaced by zero-color pixels
    filtered_list = []
    for i, grid in enumerate(grid_list):
        if i == min_index:
            # Replace all pixels with zero-color pixels
            new_pixels = [(x, y, 0) for x, y, _ in grid.pixels]
            new_grid = Grid([[]], grid.ul_x, grid.ul_y)
            new_grid.pixels = new_pixels
            new_grid.height = grid.height
            new_grid.width = grid.width
            filtered_list.append(new_grid)
        else:
            filtered_list.append(grid)

    return filtered_list

def keep_smallest(grid_list: List[Grid], value_list: List[int]) -> Grid:
    if not grid_list or not value_list or len(grid_list) != len(value_list):
        return grid_list

    # Find the index of the minimum value
    min_index = value_list.index(min(value_list))

    return grid_list[min_index]

def get_major_pixel(grid: Grid) -> Grid:
    major_color = get_major_color(grid)
    cells = ((major_color,),)
    return Grid(cells, 
                (grid.ul_x + (grid.ul_x + grid.width)) / 2, 
                (grid.ul_y + (grid.ul_y + grid.height)) / 2, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def get_minor_pixel(grid: Grid) -> Grid:
    minor_color = get_minor_color(grid)
    cells = ((minor_color,),)
    return Grid(cells,                 
                (grid.ul_x + (grid.ul_x + grid.width)) / 2, 
                (grid.ul_y + (grid.ul_y + grid.height)) / 2, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def get_pixels(grid: Grid) -> List[int]:
    pixel_colors = [color for _, _, color in grid.pixels]
    return pixel_colors

def is_h_symmetrical(grid: Grid) -> bool:
    # Get the non-zero pixels
    non_zero_pixels = [(x-grid.ul_x, y-grid.ul_y) for x, y, color in grid.pixels if color != 0]
    
    if not non_zero_pixels:
        return True  # An empty grid is considered symmetrical
    
    # Check if each pixel has a corresponding symmetrical pixel
    for x, y in non_zero_pixels:
        symmetrical_x = grid.width - 1 - x
        if (symmetrical_x, y) not in non_zero_pixels:
            return False

    return True

def is_v_symmetrical(grid: Grid) -> bool:
    # Get the non-zero pixels
    non_zero_pixels = [(x-grid.ul_x, y-grid.ul_y) for x, y, color in grid.pixels if color != 0]
    
    if not non_zero_pixels:
        return True  # An empty grid is considered symmetrical

    # Check if each pixel has a corresponding symmetrical pixel
    for x, y in non_zero_pixels:
        symmetrical_y = grid.height - 1- y
        if (x, symmetrical_y) not in non_zero_pixels:
            return False

    return True

def logical_not(v: bool) -> bool:
    return not v

def greater_than(a: int, b: int) -> bool:
    return a > b

def less_than(a: int, b: int) -> bool:
    return a < b

def equal(a: int, b: int) -> bool:
    return a == b

def keep_boolean(grid_list: List[Grid], values: List[bool]) -> Grid:
    # Check if the lengths of grid_list and values match
    if len(grid_list) != len(values):
        raise ValueError("The lengths of grid_list and values must be the same.")

    # Find the index of the first True value
    try:
        first_true_index = values.index(True)
    except ValueError:
        # If no True value is found, return an empty Grid
        return Grid([[]])

    # Return the grid associated with the first True value
    return grid_list[first_true_index]

def filter_boolean(grid_list: List[Grid], values: List[bool]) -> List[Grid]:
    # Check if the lengths of grid_list and values match
    if len(grid_list) != len(values):
        raise ValueError("The lengths of grid_list and values must be the same.")

    # Create a new list with the smallest object's pixels replaced by zero-color pixels
    filtered_list = []
    for i, grid in enumerate(grid_list):
        if values[i]:
            # Replace all pixels with zero-color pixels
            new_pixels = [(x, y, 0) for x, y, _ in grid.pixels]
            new_grid = Grid([[]], grid.ul_x, grid.ul_y)
            new_grid.pixels = new_pixels
            new_grid.height = grid.height
            new_grid.width = grid.width
            filtered_list.append(new_grid)
        else:
            filtered_list.append(grid)

    return filtered_list


# ============================================================= Pre-existing primitives ===================================================================================

def remove_outline(grid: Grid) -> Grid:
    # If the grid is too small to remove outline, return the original grid
    if grid.height <= 2 or grid.width <= 2:
        return grid
    
    # Create a new grid without the outline
    new_cells = [row[1:-1] for row in grid.cells[1:-1]]
    
    # Create and return the new Grid object
    return Grid(tuple(map(tuple, new_cells)), 
                grid.ul_x + 1, 
                grid.ul_y + 1, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def insert_outline(grid: Grid) -> Grid:
    # If the grid is empty, return an empty grid
    if not grid.cells:
        return grid

    # Get the dimensions of the original grid
    original_height = grid.height
    original_width = grid.width

    # Create a new grid with increased dimensions
    new_height = original_height + 2
    new_width = original_width + 2
    new_cells = [[0 for _ in range(new_width)] for _ in range(new_height)]

    # Copy the original grid into the center of the new grid
    for i in range(original_height):
        for j in range(original_width):
            new_cells[i+1][j+1] = grid.cells[i][j]

    # Update the upper-left coordinates
    new_ul_x = grid.ul_x - 1
    new_ul_y = grid.ul_y - 1

    # Create and return the new Grid object
    return Grid(tuple(map(tuple, new_cells)), 
                new_ul_x, 
                new_ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

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

def insert_top_row(grid: Grid) -> Grid:
    new_grid = (tuple(0 for _ in range(grid.width)),) + grid.cells
    return Grid(new_grid, 
                grid.ul_x, 
                grid.ul_y - 1,
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def insert_bottom_row(grid: Grid) -> Grid:
    new_grid = grid.cells + (tuple(0 for _ in range(grid.width)),)
    return Grid(new_grid, 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def insert_left_col(grid: Grid) -> Grid:
    new_grid = tuple((0,) + row for row in grid.cells)
    return Grid(new_grid, 
                grid.ul_x - 1, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def insert_right_col(grid: Grid) -> Grid:
    new_grid = tuple(row + (0,) for row in grid.cells)
    return Grid(new_grid, 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def stack_rows_horizontally(grid: Grid) -> Grid:
    func = compose(rbind(repeat, ONE), merge)
    return Grid(func(grid.cells), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height)

def stack_rows_vertically(grid: Grid) -> Grid:
    func = chain(dmirror, compose(rbind(repeat, ONE), merge), dmirror)
    return Grid(func(grid.cells), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height)

def stack_rows_horizontally_compress(grid: Grid) -> Grid:
    func = chain(rbind(repeat, ONE), lbind(remove, ZERO), chain(first, rbind(repeat, ONE), merge))
    return Grid(func(grid.cells), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height)

def stack_columns_vertically_compress(grid: Grid) -> Grid:
    func = chain(dmirror, chain(rbind(repeat, ONE), lbind(remove, ZERO), chain(first, rbind(repeat, ONE), merge)), dmirror)
    return Grid(func(grid.cells), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height)

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
    return Grid(func(grid.cells), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def shear_rows_right(grid):
    func = chain(vmirror, compose(lbind(apply, fork(combine, compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, compose(increment, last)), lbind(rbind, greater), first))), compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, last), lbind(lbind, greater), first))))), fork(pair, compose(rbind(lbind(interval, ZERO), ONE), height), identity)), vmirror)
    return Grid(func(grid.cells), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def shear_cols_down(grid):
    func = chain(dmirror, compose(lbind(apply, fork(combine, compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, compose(increment, last)), lbind(rbind, greater), first))), compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, last), lbind(lbind, greater), first))))), fork(pair, compose(rbind(lbind(interval, ZERO), ONE), height), identity)), dmirror)
    return Grid(func(grid.cells), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def shear_cols_up(grid):
    func = chain(dmirror, chain(vmirror, compose(lbind(apply, fork(combine, compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, compose(increment, last)), lbind(rbind, greater), first))), compose(lbind(apply, first), fork(sfilter, fork(pair, last, chain(rbind(lbind(interval, ZERO), ONE), size, last)), chain(rbind(compose, last), lbind(lbind, greater), first))))), fork(pair, compose(rbind(lbind(interval, ZERO), ONE), height), identity)), vmirror), dmirror)
    return Grid(func(grid.cells), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

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

def remove_top_row(grid: Grid) -> Grid:
    new_grid = tuple(row for i, row in enumerate(grid.cells) if i != 0)
    return Grid(new_grid, 
                grid.ul_x, 
                grid.ul_y + 1, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def remove_bottom_row(grid: Grid) -> Grid:
    new_grid = tuple(row for i, row in enumerate(grid.cells) if i != len(grid.cells) - 1)
    return Grid(new_grid, 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def remove_left_column(grid: Grid) -> Grid:
    new_grid = tuple(tuple(cell for x, cell in enumerate(row) if x != 0) for row in grid.cells)
    return Grid(new_grid, 
                grid.ul_x + 1, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def remove_right_column(grid: Grid) -> Grid:
    new_grid = tuple(tuple(cell for x, cell in enumerate(row) if x != len(row) - 1) for row in grid.cells)
    return Grid(new_grid, 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def gravitate_right(grid: Grid) -> Grid:
    func = lbind(apply, fork(combine, compose(lbind(repeat, ZERO), compose(rbind(colorcount, ZERO), rbind(repeat, ONE))), lbind(remove, ZERO)))
    return func(grid)

def gravitate_left(grid: Grid) -> Grid:
    func = lbind(apply, fork(combine, lbind(remove, ZERO), chain(lbind(repeat, ZERO), rbind(colorcount, ZERO), rbind(repeat, ONE))))
    return func(grid)

def gravitate_up(grid: Grid) -> Grid:
    func = chain(rot270, lbind(apply, fork(combine, compose(lbind(repeat, ZERO), compose(rbind(colorcount, ZERO), rbind(repeat, ONE))), lbind(remove, ZERO))), rot90)
    return func(grid)

def gravitate_down(grid: Grid) -> Grid:
    func = chain(rot270, lbind(apply, fork(combine, lbind(remove, ZERO), chain(lbind(repeat, ZERO), rbind(colorcount, ZERO), rbind(repeat, ONE)))), rot90)
    return func(grid)

def gravitate_left_right(grid: Grid) -> Grid:
    func = fork(hconcat, compose(lbind(apply, fork(combine, lbind(remove, ZERO), chain(lbind(repeat, ZERO), rbind(colorcount, ZERO), rbind(repeat, ONE)))), lefthalf), compose(lbind(apply, fork(combine, compose(lbind(repeat, ZERO), compose(rbind(colorcount, ZERO), rbind(repeat, ONE))), lbind(remove, ZERO))), righthalf))
    return func(grid)

def gravitate_top_down(grid: Grid) -> Grid:
    func = fork(vconcat, compose(chain(rot270, lbind(apply, fork(combine, compose(lbind(repeat, ZERO), compose(rbind(colorcount, ZERO), rbind(repeat, ONE))), lbind(remove, ZERO))), rot90), tophalf), compose(chain(rot270, lbind(apply, fork(combine, lbind(remove, ZERO), chain(lbind(repeat, ZERO), rbind(colorcount, ZERO), rbind(repeat, ONE)))), rot90), bottomhalf))
    return func(grid)

def shift_left(grid: Grid) -> Grid:
    shifted_grid = Grid(grid.cells,
                        grid.ul_x - 1,
                        grid.ul_y,
                        prev_width = grid.orig_width,
                        prev_height = grid.orig_height,
                        prev_ul_x=grid.orig_ul_x,
                        prev_ul_y=grid.orig_ul_y)

    return shifted_grid

def shift_right(grid: Grid) -> Grid:
    shifted_grid = Grid(grid.cells,
                        grid.ul_x + 1,
                        grid.ul_y,
                        prev_width = grid.orig_width,
                        prev_height = grid.orig_height,
                        prev_ul_x=grid.orig_ul_x,
                        prev_ul_y=grid.orig_ul_y)

    return shifted_grid

def shift_up(grid: Grid) -> Grid:
    shifted_grid = Grid(grid.cells,
                        grid.ul_x,
                        grid.ul_y - 1,
                        prev_width = grid.orig_width,
                        prev_height = grid.orig_height,
                        prev_ul_x=grid.orig_ul_x,
                        prev_ul_y=grid.orig_ul_y)

    return shifted_grid

def shift_down(grid: Grid) -> Grid:
    shifted_grid = Grid(grid.cells,
                        grid.ul_x,
                        grid.ul_y + 1,
                        prev_width = grid.orig_width,
                        prev_height = grid.orig_height,
                        prev_ul_x=grid.orig_ul_x,
                        prev_ul_y=grid.orig_ul_y)

    return shifted_grid

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
    result_grid = tophalf(righthalf(grid))

    return Grid(result_grid.cells,
                result_grid.ul_x,
                result_grid.ul_y,
                prev_width=result_grid.orig_width,
                prev_height=result_grid.orig_height,
                prev_ul_x=result_grid.orig_ul_x,
                prev_ul_y=result_grid.orig_ul_y)

def third_quadrant(grid):
    result_grid = bottomhalf(lefthalf(grid))

    return Grid(result_grid.cells,
                result_grid.ul_x,
                result_grid.ul_y,
                prev_width=result_grid.orig_width,
                prev_height=result_grid.orig_height,
                prev_ul_x=result_grid.orig_ul_x,
                prev_ul_y=result_grid.orig_ul_y)

def fourth_quadrant(grid):
    result_grid = bottomhalf(righthalf(grid))

    return Grid(result_grid.cells,
                result_grid.ul_x,
                result_grid.ul_y,
                prev_width=result_grid.orig_width,
                prev_height=result_grid.orig_height,
                prev_ul_x=result_grid.orig_ul_x,
                prev_ul_y=result_grid.orig_ul_y)

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
    if isinstance(container, Grid):
        return Grid(list(function(e) for e in container.cells), 
                    container.ul_x, 
                    container.ul_y, 
                    prev_width=container.orig_width, 
                    prev_height=container.orig_height,
                    prev_ul_x=container.orig_ul_x,
                    prev_ul_y=container.orig_ul_y)
    else:
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

def height(grid: Grid) -> int:
    return grid.height

def width(grid: Grid) -> int:
    return grid.width

def shape(grid: Grid) -> Grid:
    return (grid.height, grid.width)

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

def rot90(grid: Grid) -> Grid:
    """ quarter clockwise rotation """
    return Grid(tuple(row for row in zip(*(grid.cells)[::-1])), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)
                

def rot180(grid: Grid) -> Grid:
    """ half rotation """
    return Grid(tuple(tuple(row[::-1]) for row in grid.cells[::-1]), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def rot270(grid: Grid) -> Grid:
    """ quarter anticlockwise rotation """
    return Grid(tuple(tuple(row[::-1]) for row in zip(*(grid.cells)[::-1]))[::-1], 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def hmirror(grid: Grid) -> Grid:
    """ mirroring along horizontal """
    return Grid(tuple(row[::-1] for row in grid.cells), 
                grid.ul_x,
                grid.ul_y,
                prev_width=grid.orig_width,
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def vmirror(grid: Grid) -> Grid:
    """ mirroring along vertical """
    return Grid(tuple(reversed(grid.cells)), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def dmirror(grid: Grid) -> Grid:
    """ mirroring along diagonal """
    return vmirror(hmirror(grid))

def cmirror(grid: Grid) -> Grid:
    """ mirroring along counterdiagonal """
    return vmirror(dmirror(vmirror(grid)))

def fill(grid, value, patch):
    """ fill value at indices """
    h, w = grid.height, grid.width
    grid_filled = list(list(row) for row in grid.cells)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return Grid(tuple(tuple(row) for row in grid_filled), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

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

def hupscale(grid: Grid, factor: int) -> Grid:
    """ upscale grid horizontally """
    upscaled_grid = tuple()
    for row in grid.cells:
        upscaled_row = tuple()
        for value in row:
            upscaled_row = upscaled_row + tuple(value for num in range(factor))
        upscaled_grid = upscaled_grid + (upscaled_row,)
    return Grid(upscaled_grid, 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def vupscale(grid: Grid, factor: int) -> Grid:
    """ upscale grid vertically """
    upscaled_grid = tuple()
    for row in grid.cells:
        upscaled_grid = upscaled_grid + tuple(row for num in range(factor))
    return Grid(upscaled_grid, 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def upscale(grid, factor):
    """ upscale object or grid """
    if isinstance(grid, Grid):
        element = grid.cells
        upscaled_grid = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            upscaled_grid = upscaled_grid + tuple(upscaled_row for num in range(factor))
        return Grid(upscaled_grid, 
                    grid.ul_x, 
                    grid.ul_y, 
                    prev_width=grid.orig_width, 
                    prev_height=grid.orig_height,
                    prev_ul_x=grid.orig_ul_x,
                    prev_ul_y=grid.orig_ul_y)

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

def hconcat(a: Grid, b: Grid):
    return hconcat_list([a, b])

def hconcat_list(grid_list: List[Grid]) -> Grid:
    if not grid_list:
        return Grid(tuple())
    result = grid_list[0]
    for grid in grid_list[1:]:
        result = Grid(tuple(row1 + row2 for row1, row2 in zip(result.cells, grid.cells)), 
                      result.ul_x, 
                      result.ul_y, 
                      prev_width=grid.orig_width, 
                      prev_height=grid.orig_height,
                      prev_ul_x=grid.orig_ul_x,
                      prev_ul_y=grid.orig_ul_y)
    return result

def vconcat(a, b):
    return vconcat_list([a, b])

def vconcat_list(grid_list: List[Grid]) -> Grid:
    if not grid_list:
        return Grid(tuple())
    result = grid_list[0]
    for grid in grid_list[1:]:
        result = Grid(result.cells + grid.cells, 
                      result.ul_x, 
                      result.ul_y, 
                      prev_width=grid.orig_width, 
                      prev_height=grid.orig_height,
                      prev_ul_x=grid.orig_ul_x,
                      prev_ul_y=grid.orig_ul_y)
    return result

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

def topthird(grid: Grid) -> Grid:
    return Grid(grid.cells[:grid.height // 3], 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def bottomthird(grid: Grid) -> Grid:
    y_offset = 2 * (grid.height // 3) + grid.height % 3
    return Grid(grid.cells[y_offset:], 
                grid.ul_x, 
                grid.ul_y + y_offset, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)
                
def vcenterthird(grid: Grid) -> Grid:
    sub_grid_dim = grid.height // 3
    offset = 1
    if 3 * sub_grid_dim == grid.height:
        offset = 0
    start_row = grid.height // 3 + offset
    end_row = 2 * start_row

    tmp = grid.cells[start_row:end_row]
    return Grid(tmp, 
                grid.ul_x, 
                grid.ul_y + start_row, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def hcenterthird(grid):
    result_grid = rot270(vcenterthird(rot90(grid)))

    offset = 1
    if 3 * (grid.width // 3) == grid.width:
        offset = 0

    start_col = grid.width // 3 + offset
    return Grid(result_grid.cells,
                result_grid.ul_x + start_col,
                result_grid.orig_ul_y,
                prev_width=result_grid.orig_width,
                prev_height=result_grid.orig_height,
                prev_ul_x=result_grid.orig_ul_x,
                prev_ul_y=result_grid.orig_ul_y)

def leftthird(grid):
    return rot270(topthird(rot90(grid)))

def rightthird(grid):
    result_grid = rot270(bottomthird(rot90(grid)))

    x_offset = 2 * (grid.width // 3) + grid.width % 3
    return Grid(result_grid.cells,
                result_grid.ul_x + x_offset,
                result_grid.orig_ul_y,
                prev_width=result_grid.orig_width,
                prev_height=result_grid.orig_height,
                prev_ul_x=result_grid.orig_ul_x,
                prev_ul_y=result_grid.orig_ul_y)

def tophalf(grid: Grid) -> Grid:
    """ upper half of grid """
    return Grid(grid.cells[:len(grid.cells) // 2], 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def bottomhalf(grid: Grid) -> Grid:
    """ lower half of grid """
    y_offset = len(grid.cells) // 2 + len(grid.cells) % 2
    return Grid(grid.cells[y_offset:], 
                grid.ul_x, 
                grid.ul_y + y_offset, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def lefthalf(grid: Grid) -> Grid:
    """ left half of grid """
    result_grid = rot270(tophalf(rot90(grid)))
    return Grid(result_grid.cells,
                grid.ul_x,
                grid.ul_y,
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)                

def righthalf(grid: Grid) -> Grid:
    """ right half of grid """
    result_grid = rot270(bottomhalf(rot90(grid)))
    x_offset = len(grid.cells[0]) // 2 + len(grid.cells[0]) % 2
    return Grid(result_grid.cells,
                grid.ul_x + x_offset,
                grid.ul_y,
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)                

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
    ri = tuple(i for i, r in enumerate(grid.cells) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid.cells)) if len(set(c)) == 1)
    return Grid(tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid.cells) if i not in ri), grid.ul_x, grid.ul_y, prev_width=grid.orig_width, prev_height=grid.orig_height)

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

def duplicate_top_row(grid: Grid) -> Grid:
    """ Duplicates the top row of the grid """
    top_row = grid.cells[0]
    return Grid((top_row,) + grid.cells, 
                grid.ul_x, 
                grid.ul_y - 1, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def duplicate_bottom_row(grid: Grid) -> Grid:
    """ Duplicates the bottom row of the grid """
    bottom_row = grid.cells[-1]
    return Grid(grid.cells + (bottom_row,), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def duplicate_left_column(grid: Grid) -> Grid:
    """ Duplicates the left column of the grid """
    return Grid(tuple(row[:1] + row for row in grid.cells), 
                grid.ul_x - 1, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def duplicate_right_column(grid: Grid) -> Grid:
    """ Duplicates the right column of the grid """
    return Grid(tuple(row + row[-1:] for row in grid.cells), 
                grid.ul_x, 
                grid.ul_y, 
                prev_width=grid.orig_width, 
                prev_height=grid.orig_height,
                prev_ul_x=grid.orig_ul_x,
                prev_ul_y=grid.orig_ul_y)

def set_fg_color(grid: Grid, color):

    px_indices = difference(asindices(grid.cells), ofcolor(grid.cells, 0))

    output_grid = fill(grid, color, px_indices)
    return output_grid

def color_change(grid, from_color, to_color):
    px_indices = ofcolor(grid.cells, from_color)

    return fill(grid, to_color, px_indices)

def color_swap(grid, color1, color2):
    px_indices1 = ofcolor(grid, color1)
    px_indices2 = ofcolor(grid, color2)

    tmp = fill(grid, color2, px_indices1)
    return fill(tmp, color1, px_indices2)

def get_minor_color(grid: Grid) -> int:
    values = [v for r in grid.cells for v in r] if isinstance(grid.cells, tuple) else [v for v, _ in grid.cells]
    return min(set(values), key=values.count)

def get_major_color(grid: Grid) -> int:
    values = [v for r in grid.cells for v in r] if isinstance(grid.cells, tuple) else [v for v, _ in grid.cells]
    return max(set(values), key=values.count)
    
def invert_colors(grid: Grid):
    colors = set(v for row in grid.cells for v in row)

    if len(colors) > 2 and 0 in colors:
        fg_colors = colors - {0}
        minor_color = min(fg_colors, key=lambda c: sum(row.count(c) for row in grid.cells))
        major_color = max(fg_colors, key=lambda c: sum(row.count(c) for row in grid.cells))
    else:
        # If there are fewer than two non-zero colors, use the original functions
        minor_color = get_minor_color(grid)
        major_color = get_major_color(grid)

    minor_indices = ofcolor(grid.cells, minor_color)
    major_indices = ofcolor(grid.cells, major_color)

    tmp = fill(grid, major_color, minor_indices)
    return fill(tmp, minor_color, major_indices)

def cellwiseAND(a, b):
    return cellwiseAND_list([a, b])

def cellwiseAND_list(grid_list: List[Grid]) -> Grid:
    if len(grid_list) == 1:
        return grid_list[0]
    
    h = min(grid.height for grid in grid_list)
    w = min(grid.width for grid in grid_list)

    # Start with the first grid in the list
    resulting_grid = grid_list[0]

    # Iterate through the remaining grids in the list
    for grid in grid_list[1:]:
        new_grid = tuple()
        for i in range(h):
            row = tuple()
            for j in range(w):
                a_value = resulting_grid.cells[i][j]
                b_value = grid.cells[i][j]
                value = 0
                if a_value != 0 and b_value != 0:
                    value = a_value
                row = row + (value,)
            new_grid = new_grid + (row,)
        resulting_grid = Grid(new_grid)

    return resulting_grid

def cellwiseXOR(a, b):
    return cellwiseXOR_list([a, b])

def cellwiseXOR_list(grid_list: List[Grid]) -> Grid:
    if len(grid_list) == 1:
        return grid_list[0]

    h = min(grid.height for grid in grid_list)
    w = min(grid.width for grid in grid_list)

    # Start with the first grid in the list
    resulting_grid = grid_list[0]

    # Iterate through the remaining grids in the list
    for grid in grid_list[1:]:
        new_grid = tuple()
        for i in range(h):
            row = tuple()
            for j in range(w):
                a_value = resulting_grid.cells[i][j]
                b_value = grid.cells[i][j]
                value = 0
                if a_value != 0 and b_value == 0:
                    value = a_value
                elif a_value == 0 and b_value != 0:
                    value = b_value
                row = row + (value,)
            new_grid = new_grid + (row,)
        resulting_grid = Grid(new_grid)

    return resulting_grid

def cellwiseOR(a, b):
    return cellwiseOR_list([a, b])

def cellwiseOR_list(grid_list: List[Grid]) -> Grid:
    if len(grid_list) == 1:
        return grid_list[0]
    
    h = min(grid.height for grid in grid_list)
    w = min(grid.width for grid in grid_list)

    # Start with the first grid in the list
    resulting_grid = grid_list[0]

    # Iterate through the remaining grids in the list
    for grid in grid_list[1:]:
        new_grid = tuple()
        for i in range(h):
            row = tuple()
            for j in range(w):
                a_value = resulting_grid.cells[i][j]
                b_value = grid.cells[i][j]
                value = 0
                if a_value != 0:
                    value = a_value
                else:
                    value = b_value
                row = row + (value,)
            new_grid = new_grid + (row,)
        resulting_grid = Grid(new_grid, grid.ul_x, grid.ul_y, prev_width=grid.orig_width, prev_height=grid.orig_height)

    return resulting_grid

def cellwiseNOR(a, b):
    return cellwiseNOR_list([a, b])

def cellwiseNOR_list(grid_list: List[Grid]) -> Grid:
    if len(grid_list) == 1:
        return grid_list[0]

    h = min(grid.height for grid in grid_list)
    w = min(grid.width for grid in grid_list)

    # Start with the first grid in the list
    resulting_grid = grid_list[0]
    draw_color = get_major_color(grid_list[0])
    if draw_color == 0:
        draw_color = 1

    # Iterate through the remaining grids in the list
    for grid in grid_list[1:]:
        new_grid = tuple()
        for i in range(h):
            row = tuple()
            for j in range(w):
                a_value = resulting_grid.cells[i][j]
                b_value = grid.cells[i][j]
                value = 0
                if a_value == 0 and b_value == 0:
                    value = draw_color
                row = row + (value,)
            new_grid = new_grid + (row,)
        resulting_grid = Grid(new_grid)

    return resulting_grid

def cellwiseDifference(a, b):
    return cellwiseDifference_list([a, b])

def cellwiseDifference_list(grid_list: List[Grid]) -> Grid:
    if len(grid_list) == 1:
        return grid_list[0]

    h = min(grid.height for grid in grid_list)
    w = min(grid.width for grid in grid_list)

    # Start with the first grid in the list
    resulting_grid = grid_list[0]

    # Iterate through the remaining grids in the list
    for grid in grid_list[1:]:
        new_grid = tuple()
        for i in range(h):
            row = tuple()
            for j in range(w):
                a_value = resulting_grid.cells[i][j]
                b_value = grid.cells[i][j]
                value = 0
                if a_value != 0 and b_value == 0:
                    value = a_value
                row = row + (value,)
            new_grid = new_grid + (row,)
        resulting_grid = Grid(new_grid)

    return resulting_grid

# ================================================================== Various utility functions ===========================================================================

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

import inspect
import types

def get_num_args(prim_name):
    if prim_name == 'for_each':
        return 2
    
    if prim_name == 'keep_largest' or prim_name == 'keep_smallest' or prim_name == 'filter_smallest' or prim_name == 'filter_largest':
        return 2

    if prim_name == 'keep_boolean' or prim_name == 'filter_boolean':
        return 2

    if prim_name in ['cellwiseOR', 'cellwiseNOR', 'cellwiseXOR', 'cellwiseAND', 'cellwiseDifference', 'vconcat', 'hconcat']:
        return 2

    return 1

def get_param_type(prim_name):
    if prim_name in ['keep_largest', 'keep_smallest', 'filter_largest', 'filter_smallest', 'keep_boolean', 'filter_boolean', 'for_each', 'apply_to_grid', 'vconcat', 'hconcat']:
        return List[Grid]
    elif prim_name.startswith('cellwise'):
        return List[Grid]
    else:
        return Grid

def is_diagonal_primitive(prim_name):
    if 'keep_only_diagonal' in prim_name or 'keep_only_counterdiagonal' in prim_name or \
       'clear_diagonal' in prim_name or 'clear_counterdiagonal' in prim_name or 'dmirror' in prim_name or \
       'cmirror' in prim_name:
        return True
    else:
        return False

def is_color_primitive(prim_name):
    if 'color_change' in prim_name:
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
    'cellwiseOR': 39,
    'cellwiseAND': 40,
    'cellwiseXOR': 41,
    'cellwiseDifference': 42,
    'cellwiseNOR': 43,
    'vconcat': 44,
    'hconcat': 45,
    'color_change': 46,
    'invert_colors': 47,
    'first_quadrant': 48,
    'second_quadrant': 49,
    'third_quadrant': 50,
    'fourth_quadrant': 51,
    'hfirstfourth': 52,
    'hsecondfourth': 53,
    'hthirdfourth': 54,
    'hlastfourth': 55,
    'vfirstfourth': 56,
    'vsecondfourth': 57,
    'vthirdfourth': 58,
    'vlastfourth': 59,
    'rot180': 60,
    'rot270': 61,
    'duplicate_top_row': 62,
    'duplicate_bottom_row': 63,
    'duplicate_left_column': 64,
    'duplicate_right_column': 65,
    'compress': 66,
    'get_objects1': 67,
    'get_objects2': 68,
    'get_objects3': 69,
    'get_objects4': 70,
    'get_objects5': 71,
    'compress_objects_linear': 72,
    'compress_objects_quad': 73,
    'compress_objects_quad_pad': 74,
    'apply_to_grid': 75,
    'for_each': 76,
    'remove_outline': 77,
    'shear_grid_left': 78,
    'shear_grid_right': 79,
    'shear_grid_zigzag': 80,
    'get_object_size': 81,
    'count': 82,
    'filter_largest': 83,
    'keep_largest': 84,
    'filter_smallest': 85,
    'keep_smallest': 86,
    'get_pixels': 87,
    'is_h_symmetrical': 88,
    'is_v_symmetrical': 89,
    'logical_not': 90,
    'keep_boolean': 91,
    'filter_boolean': 92,
    'get_major_pixel': 93,
    'get_minor_pixel': 94,
    'insert_outline': 95,
    'upscale_by_three': 96,
    'cellwiseOR_list': 97,
    'get_objects6': 98
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

    'cellwiseOR': lambda g: lambda h: cellwiseOR(g, h),
    'cellwiseAND': lambda g: lambda h: cellwiseAND(g, h),
    'cellwiseXOR': lambda g: lambda h: cellwiseXOR(g, h),
    'cellwiseDifference': lambda g: lambda h: cellwiseDifference(g, h),
    'cellwiseNOR': lambda g: lambda h: cellwiseNOR(g, h),
    'vconcat': lambda g: lambda h: vconcat_list([g, h]),
    'hconcat': lambda g: lambda h: hconcat_list([g, h]),

    # separate 1-arg primitives so they're not included automatically in atomic primitives for generate_2deep_DSL
    'color_change': lambda g: lambda c1: lambda c2: color_change(g, c1, c2),
    'invert_colors': lambda g: invert_colors(g),

    # these are not 2-deep compositions of others, but I don't want to compose them either
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
    'duplicate_top_row': lambda g: duplicate_top_row(g),
    'duplicate_bottom_row': lambda g: duplicate_bottom_row(g),
    'duplicate_left_column': lambda g: duplicate_left_column(g),
    'duplicate_right_column': lambda g: duplicate_right_column(g),
    'compress': lambda g: compress(g),
    'get_objects1': lambda g: get_objects1(g),
    'get_objects2': lambda g: get_objects2(g),
    'get_objects3': lambda g: get_objects3(g),
    'get_objects4': lambda g: get_objects4(g),
    'get_objects5': lambda g: get_objects5(g),
    'compress_objects_linear': lambda g: compress_objects_linear(g),
    'compress_objects_quad': lambda g: compress_objects_quad(g),
    'compress_objects_quad_pad': lambda g: compress_objects_quad_pad(g),
    'apply_to_grid': lambda g0: lambda g: apply_to_grid(g0, g),
    'for_each': lambda g: lambda f: for_each(g, f),
    'remove_outline': lambda g: remove_outline(g),
    'shear_grid_left': lambda g: shear_grid_left(g),
    'shear_grid_right': lambda g: shear_grid_right(g),
    'shear_grid_zigzag': lambda g: shear_grid_zigzag(g),
    'get_object_size': lambda g: get_object_size(g),
    'count': lambda g: count(g),
    'filter_largest': lambda g: lambda v: filter_largest(g, v),
    'keep_largest': lambda g: lambda v: keep_largest(g, v),
    'filter_smallest': lambda g: lambda v: filter_smallest(g, v),
    'keep_smallest': lambda g: lambda v: keep_smallest(g, v),
    'get_pixels': lambda g: get_pixels(g),
    'is_h_symmetrical': lambda g: is_h_symmetrical(g),
    'is_v_symmetrical': lambda g: is_v_symmetrical(g),
    'logical_not': lambda v: logical_not(v),
    'keep_boolean': lambda g: lambda v: keep_boolean(g, v),
    'filter_boolean': lambda g: lambda v: filter_boolean(g, v),
    'get_major_pixel': lambda g: get_major_pixel(g),
    'get_minor_pixel': lambda g: get_minor_pixel(g),
    'insert_outline': lambda g: insert_outline(g),
    'upscale_by_three': lambda g: upscale_by_three(g),
    'cellwiseOR_list': lambda g: cellwiseOR_list(g),
    'get_objects6': lambda g: get_objects6(g)
}
