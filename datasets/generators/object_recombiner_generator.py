import numpy as np
import random
import search.program_interpreter_V3 as pi
import ARC_gym.utils.visualization as viz
import copy


class ObjectRecombinerGenerator:

    def __init__(self, DSL, validation=False):
        self.DSL = DSL
        self.validation = validation
        self.NEW_LEVEL = 0
        self.IDENTITY = 1
        
    @staticmethod
    def paste_objects_on_grid(target_grid, list_of_objs, adjacency_ok=True):
        height, width = len(target_grid), len(target_grid[0])

        overall_attempts = 0
        while True:
            overall_attempts += 1

            if overall_attempts == 50:
                return None
            
            # Reset the result grid
            result_grid = [[target_grid[i][j] for j in range(width)] for i in range(height)]
            memory_grid = [[-1 for j in range(width)] for i in range(height)]
                   
            random.shuffle(list_of_objs)
            placed_all = True
            
            for obj in list_of_objs:
                obj_h = len(obj)
                obj_w = len(obj[0])
                
                # Try to place the object 50 times
                placed = False
                for _ in range(50):
                    # Pick a random position
                    pos_y = random.randint(0, height - obj_h)
                    pos_x = random.randint(0, width - obj_w)
                    
                    # Check if position is clear
                    can_place = True
                    if adjacency_ok:
                        for i in range(obj_h):
                            for j in range(obj_w):
                                if memory_grid[pos_y + i][pos_x + j] != -1:
                                    can_place = False
                                    break
                            if not can_place:
                                break
                    else:
                        start_x = max(0, pos_x - 1)
                        start_y = max(0, pos_y - 1)
                        end_x = min(pos_x + obj_w + 2, width)
                        end_y = min(pos_y + obj_h + 2, height)
                        for i in range(start_y, end_y):
                            for j in range(start_x, end_x):
                                if memory_grid[i][j] != -1:
                                    can_place = False
                                    break

                            if not can_place:
                                break
                           
                    if can_place:
                        # Place the object
                        for i in range(obj_h):
                            for j in range(obj_w):
                                result_grid[pos_y + i][pos_x + j] = obj[i][j]
                                memory_grid[pos_y + i][pos_x + j] = obj[i][j]
                        placed = True
                        break
                        
                if not placed:
                    placed_all = False
                    break
                    
            if placed_all:
                return result_grid

    @staticmethod
    def draw_subobjects(get_inner_objects, target_grid, num_sub_objects, obj_color, bg_color):
        def get_valid_positions(memory_grid, height, width, adjacency_ok):
            valid_positions = []
            for j in range(1, len(memory_grid) - height):
                for i in range(1, len(memory_grid[0]) - width):
                    valid = True

                    if adjacency_ok:
                        start_val = 0
                        end_width = width
                        end_height = height
                    else:
                        start_val = -1
                        end_width = width + 1
                        end_height = height + 1

                    for y in range(start_val, end_height):
                        for x in range(start_val, end_width):
                            if adjacency_ok:
                                if j+y >= len(memory_grid) - 1 or i+x >= len(target_grid[0]) - 1:
                                    valid = False
                                    break
                            else:
                                if j+y >= len(memory_grid) or i+x >= len(target_grid[0]):
                                    valid = False
                                    break

                            if memory_grid[max(0, j+y)][max(0, i+x)] != -1:
                                valid = False
                                break

                        if not valid:
                            break
                    
                    if valid:
                        valid_positions.append((j, i))

            return valid_positions

        if get_inner_objects == 'get_objects2':
            adjacency_ok = True
            min_sobj_size = 2
        else:
            adjacency_ok = False
            min_sobj_size = 1

        # Fill grid with most common color except edges
        for i in range(1, len(target_grid)-1):
            for j in range(1, len(target_grid[0])-1):
                target_grid[i][j] = obj_color

        failed_before = False
        while True:
            tmp_target_grid = copy.deepcopy(target_grid)
            memory_grid = np.ones_like(target_grid) * -1
            sub_obj_idx = 0
            obj_processing_attempts = 0

            while sub_obj_idx < num_sub_objects and obj_processing_attempts < 10:
                obj_processing_attempts += 1

                # Get random color between 1-9, excluding most_common_color
                possible_colors = list(range(10))
                possible_colors.remove(obj_color)
                if bg_color in possible_colors:
                    possible_colors.remove(bg_color)

                if 0 in possible_colors:
                    possible_colors.remove(0)
                color = random.choice(possible_colors)
                    
                # Randomly choose shape type
                if get_inner_objects == 'get_objects5':
                    shape_type = random.choice(['rectangle', 'plus', 'x'])
                else:
                    shape_type = random.choice(['rectangle', 'plus'])
                
                if shape_type == 'rectangle':
                    # Randomly choose width and height between 2-4
                    width = random.randint(min_sobj_size, 4)
                    height = random.randint(min_sobj_size, 4)

                    if failed_before:
                        width = 2
                        height = 2

                    # Get valid position that fits object
                    valid_positions = get_valid_positions(memory_grid, height, width, adjacency_ok)

                    if not valid_positions:
                        continue

                    # Choose random valid position
                    pos_y, pos_x = random.choice(valid_positions)
                
                    if failed_before and sub_obj_idx == 0:
                        pos_y = 1
                        pos_x = 1

                    # Draw rectangle
                    for i in range(pos_y, pos_y + height):
                        for j in range(pos_x, pos_x + width):
                            tmp_target_grid[i][j] = color
                            memory_grid[i][j] = 0

                    sub_obj_idx += 1
                    obj_processing_attempts = 0
                else:
                    dim = 3

                    # Get valid position that fits object
                    valid_positions = get_valid_positions(memory_grid, dim, dim, adjacency_ok)

                    if not valid_positions:
                        continue

                    # Choose random valid position
                    pos_y, pos_x = random.choice(valid_positions)

                    if failed_before and sub_obj_idx == 0:
                        pos_y = 1
                        pos_x = 1

                    # Mark full 3x3 area in memory grid
                    for i in range(pos_y, pos_y + dim):
                        for j in range(pos_x, pos_x + dim):
                            memory_grid[i][j] = 0

                    if shape_type == 'plus':
                        # Draw plus shape
                        mid_y = pos_y + (dim//2)
                        mid_x = pos_x + (dim//2)
                        
                        # Vertical line
                        for i in range(pos_y, pos_y + dim):
                            tmp_target_grid[i][mid_x] = color
                            
                        # Horizontal line
                        for j in range(pos_x, pos_x + dim):
                            tmp_target_grid[mid_y][j] = color
                            
                    else: # x shape
                        # Draw x shape
                        for i in range(dim):
                            tmp_target_grid[pos_y + i][pos_x + i] = color
                            tmp_target_grid[pos_y + i][pos_x + dim - 1 - i] = color

                    sub_obj_idx += 1
                    obj_processing_attempts = 0

            if obj_processing_attempts < 10:
                break

            failed_before = True

        return tmp_target_grid

    @staticmethod
    def generate_objects_with_subobjs(num_objects, add_noise):

        def generate_object(is_empty, obj_width, obj_height, num_sub_objects, bg_color):
            available_colors = [c for c in range(1, 10) if c != bg_color]
            frame_color = np.random.choice(available_colors)

            if is_empty:
                # Choose random color between 1-9
                target_grid = [[0 for _ in range(obj_width)] for _ in range(obj_height)]

                # Top and bottom rows
                for j in range(obj_width):
                    target_grid[0][j] = frame_color
                    target_grid[obj_height-1][j] = frame_color
                    
                # Left and right columns 
                for i in range(obj_height):
                    target_grid[i][0] = frame_color
                    target_grid[i][obj_width-1] = frame_color
            else:
                target_grid = [[frame_color for _ in range(obj_width)] for _ in range(obj_height)]
            
            if is_empty:
                obj_color = bg_color
            else:
                obj_color = frame_color

            target_grid = WindowingGenerator.draw_subobjects('get_objects4', target_grid, num_sub_objects, obj_color, bg_color)

            return tuple(tuple(row) for row in target_grid), frame_color
        
        while True:
            
            if num_objects == 1:
                case = np.random.choice([1, 2])
                if case == 1:
                    print("Num objects == 1, case == 1")
                    min_obj_widths = [8]
                    min_obj_heights = [8]
                    max_obj_widths = [12]
                    max_obj_heights = [12]
                    width = np.random.choice(np.arange(15, 31))
                    height = np.random.choice(np.arange(15, 31))

                    sub_obj_counts = [np.random.choice([1, 2, 3])]
                elif case == 2:
                    print("Num objects == 1, case == 2")
                    min_obj_widths = [15]
                    min_obj_heights = [15]
                    max_obj_widths = [20]
                    max_obj_heights = [20]
                    width = np.random.choice(np.arange(25, 31))
                    height = np.random.choice(np.arange(25, 31))

                    sub_obj_counts = [np.random.choice([3, 4, 5])]

            elif num_objects == 2:
                case = np.random.choice([1, 2])

                if case == 1:
                    print("Num objects == 2, case == 1")
                    min_obj_widths = [6, 8]
                    min_obj_heights = [6, 8]
                    max_obj_widths = [10, 10]
                    max_obj_heights = [10, 10]
                    width = 22
                    height = 20
                    sub_obj_counts = [1, 2]
                else:
                    print("Num objects == 2, case == 2")
                    min_obj_widths = [10, 10]
                    min_obj_heights = [9, 9]
                    max_obj_widths = [14, 14]
                    max_obj_heights = [14, 14]
                    width = 27
                    height = 27
                    sub_obj_counts = [2, 3]

            elif num_objects == 3:
                case = np.random.choice([1, 2])
                if case == 1:
                    print("Num objects == 3, case == 1")
                    min_obj_widths = [9, 12, 6]
                    min_obj_heights = [9, 12, 6]
                    max_obj_widths = [11, 14, 8]
                    max_obj_heights = [11, 14, 8]
                    width = 30
                    height = 30

                    sub_obj_counts = [2, 3, 1]
                else:
                    print("Num objects == 3, case == 2")
                    min_obj_widths = [8, 7, 10]
                    min_obj_heights = [8, 7, 10]
                    max_obj_widths = [11, 8, 12]
                    max_obj_heights = [11, 8, 12]
                    width = 29
                    height = 29

                    sub_obj_counts = [2, 1, 2]

            num_objects = len(min_obj_widths)
            obj_widths = [np.random.randint(min_w, max_w) for min_w, max_w in zip(min_obj_widths, max_obj_widths)]
            obj_heights = [np.random.randint(min_h, max_h) for min_h, max_h in zip(min_obj_heights, max_obj_heights)]

            object_list = []
            obj_colors = []

            is_empty = True

            bg_color = 0
            for obj_idx in range(num_objects):
                new_obj, obj_color = generate_object(is_empty, obj_widths[obj_idx], obj_heights[obj_idx], sub_obj_counts[obj_idx], bg_color)

                obj_colors.append(obj_color)
                object_list.append(new_obj)

            if add_noise:
                density = np.random.uniform()
                num_bg_colors = np.random.choice([2, 3, 4])

                # Generate the target grid with randomized noise background
                target_grid = []
                for _ in range(height):
                    row = []
                    for _ in range(width):
                        if np.random.random() < density:
                            row.append(1)  # Foreground pixel
                        else:
                            row.append(0)  # Background pixel
                    target_grid.append(tuple(row))
                target_grid = tuple(target_grid)

                # Generate random background colors
                bg_colors = np.random.randint(1, 10, size=num_bg_colors)  # Colors 1-9
                
                # Randomly pixelize the grid with selected background colors
                pixelized_grid = []
                for row in target_grid:
                    new_row = []
                    for pixel in row:
                        if pixel == 0:
                            new_row.append(0)  # Keep background black
                        else:
                            new_row.append(np.random.choice(bg_colors))
                    pixelized_grid.append(tuple(new_row))
                
                target_grid = tuple(pixelized_grid)
            else:
                target_grid = tuple(tuple(bg_color for _ in range(width)) for _ in range(height))

            adjacency_ok = True

            output = WindowingGenerator.paste_objects_on_grid(target_grid, object_list, adjacency_ok)

            if output is not None:
                return output

    @staticmethod
    def augment_grid(grid, hp):
        num_rotations = np.random.choice(np.arange(4))
        for _ in range(num_rotations):
            grid = hp.rot90(grid)

        transform = np.random.choice(['vmirror', 'hmirror', 'invert_colors', 'color_change'])
        a = np.random.uniform()
        if a < 0.75:
            if transform == 'vmirror':
                grid = hp.vmirror(grid)
            elif transform == 'hmirror':
                grid = hp.hmirror(grid)
            elif transform == 'invert_colors':
                grid = hp.invert_colors(grid)
            elif transform == 'color_change':
                colors_in_grid = set(cell for row in grid.cells for cell in row)
                if colors_in_grid:
                    c1 = np.random.choice(list(colors_in_grid))  # Randomly select a color from the grid
                    c2 = np.random.choice(np.arange(1, 10))

                    grid = hp.color_change(grid, c1, c2)

        possible_shifts = []

        # Check left column
        if all(grid.cells[i][0] == 0 for i in range(len(grid.cells))):
            possible_shifts.append('shift_left')
        
        # Check right column
        if all(grid.cells[i][-1] == 0 for i in range(len(grid.cells))):
            possible_shifts.append('shift_right')
        
        # Check top row
        if all(grid.cells[0][j] == 0 for j in range(len(grid.cells[0]))):
            possible_shifts.append('shift_up')
        
        # Check bottom row
        if all(grid.cells[-1][j] == 0 for j in range(len(grid.cells[0]))):
            possible_shifts.append('shift_down')

        b = np.random.uniform()
        if b < 0.5 and len(possible_shifts) > 0:
            apply_shift = np.random.choice(possible_shifts)
            if apply_shift == 'shift_left':
                grid = hp.shift_left(grid)
            if apply_shift == 'shift_right':
                grid = hp.shift_right(grid)
            if apply_shift == 'shift_up':
                grid = hp.shift_up(grid)
            if apply_shift == 'shift_down':
                grid = hp.shift_down(grid)

        return grid

    @staticmethod
    def get_min_max_sizes(width, height, num_objects):
        # Calculate maximum object dimensions to fit all objects
        max_obj_width = min(width // int(np.ceil(np.sqrt(num_objects))), 9)
        max_obj_height = min(height // int(np.ceil(np.sqrt(num_objects))), 9)

        min_obj_width = 3
        min_obj_height = 3

        obj_widths = np.random.randint(min_obj_width, max_obj_width + 1, size=num_objects)
        obj_heights = np.random.randint(min_obj_height, max_obj_height + 1, size=num_objects)
        
        # Ensure total area of objects doesn't exceed grid area
        clip_attempts = 0
        while np.sum([w * h for w, h in zip(obj_widths, obj_heights)]) > (width * height * 0.7):  # 70% of grid area
            clip_attempts += 1
            if clip_attempts == 10:
                break

            obj_widths = [max(min_obj_width, min(w - 1, max_obj_width)) for w in obj_widths]
            obj_heights = [max(min_obj_height, min(h - 1, max_obj_height)) for h in obj_heights]

        if clip_attempts >= 10:
            print("==> clip_attempts >= 10, leaving")
            return None, None
        
        return obj_widths, obj_heights

    @staticmethod
    def generate_8way_adj_object(obj_width, obj_height, colors):
        grid = np.zeros((obj_height, obj_width))

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        start_y = np.random.choice(np.arange(obj_height))
        start_x = np.random.choice(np.arange(obj_width))
        start_pos = (start_y, start_x)
        full_span = False

        max_attempts = obj_width * obj_height * 5
        num_attempts = 0
        while not full_span and num_attempts < max_attempts:
            num_attempts += 1

            rnd_color = np.random.choice(colors)
            grid[start_pos[0], start_pos[1]] = rnd_color

            # Check adjacent cells for zeros
            possible_moves = []
            for dy, dx in directions:
                next_y = start_pos[0] + dy
                next_x = start_pos[1] + dx
                
                # Check if position is within grid bounds
                if 0 <= next_y < obj_height and 0 <= next_x < obj_width:
                    if grid[next_y, next_x] == 0:
                        possible_moves.append((dy, dx))
            
            # Filter out moves that would go outside grid boundaries
            def clip_possible_moves(start_pos, possible_moves):
                valid_moves = []
                for dy, dx in possible_moves:
                    next_y = start_pos[0] + dy
                    next_x = start_pos[1] + dx
                    if 0 <= next_y < obj_height and 0 <= next_x < obj_width:
                        valid_moves.append((dy, dx))
                return valid_moves

            possible_moves = clip_possible_moves(start_pos, possible_moves)

            # If no zero cells found, use all directions
            if len(possible_moves) == 0:
                possible_moves = directions
                possible_moves = clip_possible_moves(start_pos, possible_moves)

            rnd_move_idx = np.random.choice(np.arange(len(possible_moves)))
            rnd_move = possible_moves[rnd_move_idx]
            new_pos = (start_pos[0] + rnd_move[0], start_pos[1] + rnd_move[1])
            start_pos = new_pos

            # Check if grid spans full width and height
            def check_full_span(grid):
                # Check if each column has at least one non-zero value
                cols_spanned = all(any(grid[i,j] != 0 for i in range(grid.shape[0])) 
                                 for j in range(grid.shape[1]))
                
                # Check if each row has at least one non-zero value  
                rows_spanned = all(any(grid[i,j] != 0 for j in range(grid.shape[1]))
                                 for i in range(grid.shape[0]))
                
                return cols_spanned and rows_spanned

            full_span = check_full_span(grid)

        return tuple(tuple(row) for row in grid)    

    @staticmethod
    def generate_simple_shape(num_objects):

        while True:
            if num_objects > 6:
                width = np.random.choice(np.arange(20, 31))
                height = np.random.choice(np.arange(20, 31))
            else:
                width = np.random.choice(np.arange(15, 25))
                height = np.random.choice(np.arange(15, 25))

            obj_widths, obj_heights = ObjectRecombinerGenerator.get_min_max_sizes(width, height, num_objects)
            if obj_widths is None:
                continue

            object_list = []
            obj_colors = []
            for obj_idx in range(num_objects):
                colors = [np.random.choice(np.arange(1, 10))]
                obj_colors.append(colors[0])
                new_obj = ObjectRecombinerGenerator.generate_8way_adj_object(obj_widths[obj_idx], obj_heights[obj_idx], colors)
                object_list.append(new_obj)

            target_grid = tuple(tuple(0 for _ in range(width)) for _ in range(height))

            if len(set(obj_colors)) == len(obj_colors):
                adjacency_ok = True
            else:
                adjacency_ok = False
                
            output = ObjectRecombinerGenerator.paste_objects_on_grid(target_grid, object_list, adjacency_ok)

            if output is not None:
                return output

    @staticmethod
    def generate_4way_adj_object(obj_width, obj_height, colors):

        grid = np.zeros((obj_height, obj_width))
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        start_y = np.random.choice(np.arange(obj_height))
        start_x = np.random.choice(np.arange(obj_width))
        start_pos = (start_y, start_x)
        full_span = False

        max_attempts = obj_width * obj_height * 5
        num_attempts = 0
        while not full_span and num_attempts < max_attempts:
            num_attempts += 1

            rnd_color = np.random.choice(colors)
            grid[start_pos[0], start_pos[1]] = rnd_color

            # Check adjacent cells for zeros
            possible_moves = []
            for dy, dx in directions:
                next_y = start_pos[0] + dy
                next_x = start_pos[1] + dx
                
                # Check if position is within grid bounds
                if 0 <= next_y < obj_height and 0 <= next_x < obj_width:
                    if grid[next_y, next_x] == 0:
                        possible_moves.append((dy, dx))
            
            # Filter out moves that would go outside grid boundaries
            def clip_possible_moves(start_pos, possible_moves):
                valid_moves = []
                for dy, dx in possible_moves:
                    next_y = start_pos[0] + dy
                    next_x = start_pos[1] + dx
                    if 0 <= next_y < obj_height and 0 <= next_x < obj_width:
                        valid_moves.append((dy, dx))
                return valid_moves

            possible_moves = clip_possible_moves(start_pos, possible_moves)

            # If no zero cells found, use all directions
            if len(possible_moves) == 0:
                possible_moves = directions
                possible_moves = clip_possible_moves(start_pos, possible_moves)

            rnd_move_idx = np.random.choice(np.arange(len(possible_moves)))
            rnd_move = possible_moves[rnd_move_idx]
            new_pos = (start_pos[0] + rnd_move[0], start_pos[1] + rnd_move[1])
            start_pos = new_pos

            # Check if grid spans full width and height
            def check_full_span(grid):
                # Check if each column has at least one non-zero value
                cols_spanned = all(any(grid[i,j] != 0 for i in range(grid.shape[0])) 
                                 for j in range(grid.shape[1]))
                
                # Check if each row has at least one non-zero value  
                rows_spanned = all(any(grid[i,j] != 0 for j in range(grid.shape[1]))
                                 for i in range(grid.shape[0]))
                
                return cols_spanned and rows_spanned

            full_span = check_full_span(grid)

        return tuple(tuple(row) for row in grid)    

    @staticmethod
    def generate_multicolor_4adj(num_objects):

        while True:
            # 1) select bg color
            bg_color = np.random.choice([0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

            # 2) randomly pick number of objects colors between 2 and 5 inclusively.
            num_colors = np.random.randint(2, 6)

            # 3) randomly select the objects' dimensions, considering num_objects
            if num_objects > 6:
                width = np.random.choice(np.arange(20, 31))
                height = np.random.choice(np.arange(20, 31))
            else:
                width = np.random.choice(np.arange(15, 31))
                height = np.random.choice(np.arange(15, 31))

            obj_widths, obj_heights = ObjectRecombinerGenerator.get_min_max_sizes(width, height, num_objects)
            if obj_widths is None:
                continue

            # 4) generate 4-way connected shapes
            object_list = []
            for obj_idx in range(num_objects):
                colors = np.random.choice([i for i in range(1,10) if i != bg_color], size=num_colors)
                new_obj = ObjectRecombinerGenerator.generate_4way_adj_object(obj_widths[obj_idx], obj_heights[obj_idx], colors)
                object_list.append(new_obj)

            # 5) paste them onto the target grid
            target_grid = tuple(tuple(bg_color for _ in range(width)) for _ in range(height))
            output = ObjectRecombinerGenerator.paste_objects_on_grid(target_grid, object_list, False)

            if output is not None:
                return output

    def select_grid(self, get_objects, num_objects):
        examples = []
        if self.validation:
            if get_objects == 'get_objects3':
                if num_objects == 3:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 3, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]
                elif num_objects == 2:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]
                elif num_objects == 1:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],                        
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0], [0, 0, 0, 0, 6, 0, 6, 0, 0, 0, 0], [0, 0, 0, 0, 6, 0, 6, 0, 0, 0, 0], [0, 0, 0, 0, 6, 0, 6, 0, 0, 0, 0], [0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 3, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 0, 0, 0], [0, 8, 8, 8, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 7, 7, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 0, 0, 7, 7, 0, 0, 0], [0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0], [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0], [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0], [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0], [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0], [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0], [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]
            elif get_objects == 'get_objects4':
                if num_objects == 3:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 9, 3, 3, 3, 3, 0, 0, 0, 3, 3, 9, 3, 9], [3, 3, 3, 3, 3, 9, 3, 0, 0, 0, 3, 3, 3, 3, 3], [3, 9, 3, 9, 3, 3, 3, 0, 0, 0, 3, 9, 3, 3, 9], [3, 3, 3, 3, 9, 3, 3, 0, 0, 0, 3, 3, 3, 3, 3], [3, 3, 9, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 9, 3, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 3, 3, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 9, 3, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 4, 0, 0, 0, 0, 0, 0], [0, 4, 8, 4, 0, 0, 0, 0, 0, 0], [0, 8, 8, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 3, 2, 2, 0], [0, 0, 0, 0, 0, 0, 3, 3, 2, 0], [0, 0, 0, 0, 0, 0, 3, 2, 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 6, 3, 0, 0, 0, 0, 0], [0, 0, 3, 6, 3, 0, 0, 0, 0, 0], [0, 0, 3, 3, 3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]
                elif num_objects == 2:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 2, 2, 0, 0, 0, 0, 0], [0, 3, 3, 2, 0, 0, 0, 0, 0], [0, 3, 2, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 6, 6, 6, 0], [0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1, 6, 6, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]
                elif num_objects == 1:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0], [0, 5, 8, 5, 0, 0, 0], [0, 5, 8, 5, 0, 0, 0], [0, 8, 8, 8, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 8, 2, 2, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 8, 2, 2, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 3, 3, 8, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 3, 3, 8, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 8, 8, 1, 1, 8, 8, 8, 3, 3, 8, 8, 8, 8, 0], [0, 8, 8, 1, 1, 8, 8, 8, 3, 3, 8, 8, 8, 8, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 8, 8, 8, 8, 8, 8, 8, 2, 2, 8, 8, 8, 8, 0], [0, 8, 8, 8, 8, 8, 8, 8, 2, 2, 8, 8, 8, 8, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 5, 5, 8, 8, 4, 4, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 5, 5, 8, 8, 4, 4, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 3, 3, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 3, 3, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 0, 8, 8, 2, 2, 8, 8, 8, 8, 6, 6, 8, 8, 0], [0, 0, 8, 8, 2, 2, 8, 8, 8, 8, 6, 6, 8, 8, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 0, 8, 8, 1, 1, 8, 8, 8, 8, 3, 3, 8, 8, 0], [0, 0, 8, 8, 1, 1, 8, 8, 8, 8, 3, 3, 8, 8, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]
            elif get_objects == 'get_objects5':
                if num_objects == 3:
                    examples = [
                        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1], [1, 0, 2, 2, 0, 1, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 3, 8, 8, 0, 0, 8, 8, 3, 8, 8, 3, 8, 8, 0, 0, 0, 0], [0, 0, 8, 3, 8, 8, 8, 8, 8, 8, 0, 0, 8, 3, 3, 3, 8, 8, 3, 8, 0, 0, 0, 0], [0, 0, 8, 8, 8, 3, 8, 8, 8, 3, 0, 0, 8, 8, 3, 8, 8, 3, 3, 3, 0, 0, 0, 0], [0, 0, 8, 8, 3, 3, 3, 8, 8, 3, 0, 0, 8, 8, 8, 8, 8, 8, 3, 8, 0, 0, 0, 0], [0, 0, 8, 8, 8, 3, 8, 8, 8, 8, 0, 0, 3, 8, 8, 8, 3, 8, 8, 3, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 8, 8, 3, 3, 3, 8, 8, 0, 0, 0, 0], [0, 0, 8, 3, 8, 8, 8, 8, 8, 8, 0, 0, 8, 3, 8, 8, 3, 8, 8, 8, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 3, 8, 8, 0, 0, 8, 8, 8, 8, 8, 8, 3, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 3, 8, 8, 8, 3, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 3, 8, 8, 8, 8, 8, 3, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 3, 3, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 3, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 3, 8, 8, 8, 3, 8, 0, 0, 0, 3, 8, 8, 3, 8, 8, 3, 8], [0, 0, 3, 3, 3, 8, 3, 3, 3, 0, 0, 0, 8, 8, 3, 8, 8, 3, 3, 3], [0, 0, 8, 3, 8, 8, 8, 3, 8, 0, 0, 0, 8, 3, 3, 3, 8, 8, 3, 8], [0, 0, 8, 8, 8, 3, 8, 8, 8, 0, 0, 0, 8, 8, 3, 8, 8, 8, 8, 8], [0, 0, 8, 3, 8, 8, 8, 8, 8, 0, 0, 0, 3, 8, 8, 8, 8, 3, 8, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 3, 3, 3, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 3, 8, 8, 3, 8, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 8, 8, 8, 8, 8, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 3, 8, 8, 3, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 8, 3, 3, 3, 8, 8, 8, 3, 8, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 3, 8, 8, 8, 3, 3, 3, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 3, 8, 8, 8, 8, 8, 8, 3, 8, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 3, 3, 8, 8, 3, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 3, 8, 8, 3, 3, 3, 8, 8, 3, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]

                elif num_objects == 2:
                    examples = [
                        [[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 0, 1, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 1, 1, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 0, 1, 1, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 0, 0, 2, 2, 0, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 0, 0, 0, 2, 0, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 0, 0, 0, 2, 0, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]
                    ]

                elif num_objects == 1:
                    examples = [
                        [[0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], [0, 2, 3, 1, 1, 3, 2, 0, 0, 0, 0, 0], [2, 2, 1, 0, 0, 1, 2, 2, 0, 0, 0, 0], [2, 2, 1, 0, 0, 1, 2, 2, 0, 0, 0, 0], [0, 2, 3, 1, 1, 3, 2, 0, 0, 0, 0, 0], [0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 2, 2, 5, 5, 0, 0, 0], [0, 0, 0, 5, 3, 3, 3, 3, 5, 0, 0, 0], [0, 0, 2, 2, 3, 1, 1, 3, 2, 2, 0, 0], [0, 0, 2, 2, 3, 1, 1, 3, 2, 2, 0, 0], [0, 0, 0, 5, 3, 3, 3, 3, 5, 0, 0, 0], [0, 0, 0, 5, 5, 2, 2, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 7, 0, 0, 7, 7, 0], [0, 0, 0, 0, 7, 2, 2, 3, 3, 2, 2, 7], [0, 0, 0, 0, 7, 2, 8, 8, 8, 8, 2, 7], [0, 0, 0, 0, 0, 3, 8, 0, 0, 8, 3, 0], [0, 0, 0, 0, 0, 3, 8, 0, 0, 8, 3, 0], [0, 0, 0, 0, 7, 2, 8, 8, 8, 8, 2, 7], [0, 0, 0, 0, 7, 2, 2, 3, 3, 2, 2, 7], [0, 0, 0, 0, 0, 7, 7, 0, 0, 7, 7, 0]],
                        [[0, 0, 1, 0, 0, 5, 5, 0, 0, 1, 0, 0], [0, 0, 0, 5, 3, 8, 8, 3, 5, 0, 0, 0], [0, 0, 0, 3, 2, 8, 8, 2, 3, 0, 0, 0], [0, 0, 5, 8, 8, 6, 6, 8, 8, 5, 0, 0], [0, 0, 5, 8, 8, 6, 6, 8, 8, 5, 0, 0], [0, 0, 0, 3, 2, 8, 8, 2, 3, 0, 0, 0], [0, 0, 0, 5, 3, 8, 8, 3, 5, 0, 0, 0], [0, 0, 1, 0, 0, 5, 5, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0, 0]],
                        [[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]
        else:        
            # Use grids from the ARC training set
            if get_objects == 'get_objects3':
                if num_objects == 3:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0]],    
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 2, 2, 0, 0, 7, 7, 0], [0, 0, 0, 0, 0, 0, 7, 0, 7, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 6, 6, 6, 6, 0, 0, 0], [0, 0, 0, 0, 6, 6, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 4, 0, 0, 8, 8, 8, 0], [0, 0, 0, 0, 0, 0, 8, 0, 8, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 2, 2, 2, 0, 0, 0, 0], [0, 2, 2, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 3, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 5, 0, 0, 5, 0], [0, 0, 3, 0, 0, 5, 5, 5, 5, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 8, 8, 8, 0, 0, 0, 0], [8, 8, 8, 8, 0, 8, 8, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 3, 0, 0, 3, 0], [0, 9, 9, 0, 0, 3, 3, 3, 3, 0], [0, 9, 9, 0, 0, 0, 0, 0, 3, 0], [9, 9, 9, 9, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 4, 4, 4, 4, 0], [0, 0, 0, 0, 4, 0, 0, 4, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [0, 2, 2, 2, 0, 8, 0, 0, 0, 0, 0, 8, 0, 2, 2, 2, 0], [0, 0, 2, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 2, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 2, 0, 2, 0, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 2, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [0, 2, 2, 2, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 4, 4, 0, 7, 0, 0, 0, 0, 7], [0, 4, 4, 0, 0, 7, 0, 0, 7, 0], [0, 0, 0, 0, 0, 0, 7, 7, 0, 0], [0, 0, 0, 0, 0, 0, 7, 7, 0, 0], [0, 3, 0, 0, 0, 0, 3, 0, 0, 0], [0, 0, 3, 0, 0, 3, 0, 0, 0, 0], [0, 0, 0, 3, 3, 0, 0, 0, 0, 0], [0, 0, 0, 3, 3, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 8, 0, 0, 0, 0, 0, 6, 6, 0, 0, 0], [0, 0, 8, 8, 8, 0, 0, 0, 0, 6, 6, 0, 0, 0], [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]
                elif num_objects == 2:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 6, 6, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 6, 6, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 6, 6, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 6, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 6, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0], [0, 0, 0, 0, 8, 8, 8, 8, 5, 5, 5, 0, 5, 0, 0, 0], [0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 5, 5, 0, 0], [0, 0, 0, 0, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 0, 0], [0, 0, 0, 0, 8, 8, 8, 8, 0, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 5, 5, 0, 5, 0, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 0, 7, 0, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 0, 0, 0, 0, 0, 0], [4, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 4, 4, 0, 0], [0, 0, 0, 0, 0, 0, 4, 4, 0], [0, 0, 0, 0, 0, 4, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 0, 0, 0, 0, 0, 0], [0, 4, 4, 0, 0, 0, 0, 0, 0], [4, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 4, 4, 4, 0], [0, 0, 0, 0, 0, 0, 4, 0, 0], [0, 0, 0, 0, 0, 0, 4, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0], [0, 0, 4, 4, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 4, 4, 4, 0], [0, 0, 0, 0, 0, 0, 4, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 6, 6, 6, 0, 0, 0, 0, 0], [0, 6, 0, 0, 6, 0, 0, 0, 0], [0, 0, 6, 0, 0, 6, 0, 0, 0], [0, 0, 0, 6, 0, 0, 6, 0, 0], [0, 0, 0, 0, 6, 6, 6, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 2, 2, 0, 0, 0, 0], [0, 0, 2, 0, 0, 2, 0, 0, 0], [0, 0, 0, 2, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 2, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]],    
                        [[0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [0, 4, 4, 4, 0, 9, 0, 4, 4, 0, 0, 9, 0, 0, 0, 0, 0], [0, 4, 0, 4, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [0, 4, 4, 4, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [0, 4, 0, 4, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 9, 0, 4, 4, 0, 0, 9, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 9, 0, 4, 0, 4, 0, 9, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 4, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0, 4, 4, 0, 0], [0, 0, 0, 0, 0, 0, 4, 4, 0, 0]],
                        [[6, 0, 0, 0, 0, 6, 0, 0, 0, 0], [0, 6, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0, 0, 8, 0], [0, 0, 8, 0, 0, 0, 0, 8, 0, 0], [0, 0, 0, 8, 0, 0, 8, 0, 0, 0], [0, 0, 0, 0, 8, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 8, 0, 0, 0, 0]]
                    ]
                elif num_objects == 1:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],        
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 8, 0], [0, 8, 0, 8], [0, 0, 8, 0]],
                        [[0, 0, 3, 3], [0, 3, 0, 3], [3, 3, 3, 0]],
                        [[3, 3, 3, 3], [3, 0, 0, 0], [3, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]],
                        [[3, 0, 3, 0, 8, 0, 0, 0, 0], [3, 3, 0, 0, 8, 0, 0, 0, 0], [3, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0, 0, 0], [8, 8, 8, 8, 8, 8, 8, 8, 8], [0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0, 0, 0]],
                        [[2, 0, 0, 4, 0, 0, 0], [0, 2, 2, 4, 0, 0, 0], [0, 2, 0, 4, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4], [0, 0, 0, 4, 0, 0, 0], [0, 0, 0, 4, 0, 0, 0], [0, 0, 0, 4, 0, 0, 0]],   
                        [[0, 0, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [8, 0, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 8, 8, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 0, 0, 0], [0, 8, 0, 0, 0, 0, 8, 0, 0], [0, 0, 8, 0, 0, 0, 0, 8, 0], [0, 0, 0, 8, 0, 0, 0, 0, 8], [0, 0, 0, 0, 8, 8, 8, 8, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 4, 4, 4, 4, 4, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 4, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 4, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 4, 4, 4, 4, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 3, 0, 3, 0, 1, 0, 3, 0, 0, 0, 1, 0, 0, 0, 3, 0], [0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 0, 0, 0, 0, 0, 0, 3, 0], [0, 0, 3, 0, 0, 0, 0, 3, 0, 0], [0, 0, 0, 3, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 3, 3, 0, 0, 0, 0], [0, 0, 0, 0, 3, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]
            elif get_objects == 'get_objects4':
                if num_objects == 3:
                    examples = [
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8], [8, 8, 8, 6, 6, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8], [8, 8, 8, 6, 6, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8], [8, 8, 8, 8, 8, 8, 8, 3, 6, 6, 6, 6, 3, 8, 8], [8, 8, 3, 3, 3, 3, 8, 3, 6, 4, 4, 6, 3, 8, 8], [8, 8, 3, 6, 6, 3, 8, 3, 6, 4, 4, 6, 3, 8, 8], [8, 8, 3, 6, 6, 3, 8, 3, 6, 4, 4, 6, 3, 8, 8], [8, 8, 3, 3, 3, 3, 8, 3, 6, 6, 6, 6, 3, 8, 8], [8, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 6, 6, 6, 6, 3, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 6, 6, 6, 6, 3, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 6, 6, 6, 6, 3, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 6, 6, 6, 6, 3, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8]],
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 8, 8, 8, 8], [8, 8, 6, 6, 6, 6, 8, 8, 6, 6, 6, 8, 8, 8, 8], [8, 8, 6, 8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 6, 8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 8, 8, 8], [8, 8, 6, 6, 6, 6, 8, 8, 8, 6, 6, 6, 8, 8, 8], [8, 8, 6, 8, 8, 6, 8, 8, 8, 6, 8, 6, 8, 8, 8], [8, 8, 6, 8, 8, 6, 8, 8, 8, 6, 8, 6, 8, 8, 8], [8, 8, 6, 6, 6, 6, 8, 8, 8, 6, 8, 6, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8], [8, 8, 8, 8, 6, 6, 8, 8, 6, 6, 6, 8, 8, 8, 8], [8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,2,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,3,2,3,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,2,2,3,2,2,8,8],[8,8,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,3,2,3,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,2,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8],[8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8],[8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8],[8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8],[8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8],[8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8],[8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8],[8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8],[8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,8,8,8,8],[8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
                        [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,1,1,1],[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,3,3,4,3,3,1,1,1],[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,3,1,1,1,1,1],[1,2,2,2,2,2,2,2,2,2,2,2,2,2,4,2,2,2,1,1,1,1,1,1,3,1,1,1,1,1],[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,2,2,2,4,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,4,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],
                        [[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,1,2,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,3,3,3,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,8,8,2,8,8,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,3,3,3,4,4,4,4,4,4,4,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]],
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8], [8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8], [8, 8, 3, 3, 3, 8, 8, 8, 8, 4, 8, 8], [8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0], [0, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 2, 1, 0, 0, 0, 1, 2, 1, 1, 1, 1, 2, 1, 0, 0], [0, 0, 1, 1, 2, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 0], [0, 0, 1, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 8, 8, 8, 8], [8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8], [8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 8, 8, 8, 8], [6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8], [6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 8, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 8, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 1, 3, 3, 1, 1, 1, 1], [2, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 3, 1, 1, 1, 1, 1], [2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 4, 0, 0, 4, 0, 0, 0], [0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 4, 4, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 4], [4, 0, 0, 0, 1, 4, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 4], [0, 0, 4, 4, 0, 0, 0, 1, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 8, 8, 8, 8], [0, 8, 8, 8, 8, 0, 8, 2, 2, 8], [0, 8, 1, 8, 8, 0, 8, 8, 8, 8], [0, 8, 8, 2, 8, 0, 8, 2, 1, 8], [0, 8, 8, 8, 8, 0, 8, 8, 8, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 8, 8, 8, 8, 8, 8, 0], [0, 0, 0, 8, 8, 8, 2, 8, 8, 0], [0, 0, 0, 8, 2, 8, 1, 8, 8, 0], [0, 0, 0, 8, 1, 8, 8, 8, 8, 0]],
                        [[2, 8, 8, 8, 0, 0, 0, 0, 0, 0], [8, 8, 1, 8, 0, 0, 2, 8, 1, 0], [1, 2, 8, 1, 0, 0, 8, 8, 8, 0], [8, 8, 8, 8, 0, 0, 2, 1, 8, 0], [0, 0, 0, 0, 0, 0, 8, 8, 2, 0], [0, 0, 0, 0, 0, 0, 2, 8, 1, 0], [0, 1, 2, 8, 2, 0, 1, 8, 8, 0], [0, 8, 8, 1, 8, 0, 0, 0, 0, 0], [0, 1, 2, 8, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 0, 0, 0, 0, 0, 0], [0, 2, 4, 4, 0, 0, 0, 0, 0, 0], [0, 4, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 5, 5, 0], [0, 0, 0, 0, 0, 0, 5, 5, 5, 0], [0, 0, 0, 0, 0, 0, 5, 5, 5, 0], [0, 0, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 5, 5, 5, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 5, 5, 5, 5], [0, 6, 6, 6, 6, 0, 5, 5, 5, 5], [0, 8, 8, 6, 8, 0, 5, 5, 5, 5], [0, 6, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 5, 0, 0], [0, 0, 0, 0, 5, 5, 5, 5, 0, 0], [0, 0, 0, 0, 5, 5, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 4, 0, 0, 9, 8, 0, 0], [0, 0, 6, 7, 0, 0, 8, 9, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 7, 6, 0, 0, 0, 0], [0, 0, 0, 0, 6, 6, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 2, 9, 0, 0], [2, 1, 0, 0, 0, 0, 1, 6, 0, 0], [0, 0, 0, 4, 7, 0, 0, 0, 0, 0], [0, 0, 0, 8, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]

                elif num_objects == 2:
                    examples = [
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 7, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 7, 2, 7, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 4, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 4, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2], [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2], [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0, 8, 3, 3, 8, 0, 0], [0, 0, 0, 0, 0, 0, 8, 3, 3, 8, 0, 0], [0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 4, 4, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 4, 4, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 2, 0, 0, 0, 0, 0, 0, 0], [2, 2, 1, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0, 0, 0], [0, 0, 3, 1, 0, 0, 0, 0], [0, 3, 3, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 1, 1, 1, 9, 9, 9, 1, 9, 9, 9], [9, 1, 9, 1, 9, 9, 9, 1, 9, 9, 9], [9, 1, 9, 1, 9, 9, 1, 1, 1, 1, 9], [9, 1, 1, 1, 9, 9, 9, 1, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], [0, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0], [0, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0], [0, 3, 2, 2, 2, 3, 0, 0, 0, 0, 8], [0, 3, 2, 2, 2, 3, 0, 0, 0, 0, 0], [0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 6, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 6, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 6, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0], [0, 0, 0, 6, 4, 4, 4, 6, 0, 0, 0, 0, 0], [0, 0, 0, 6, 4, 4, 4, 6, 0, 0, 0, 0, 0], [0, 0, 0, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0]],
                        [[1, 1, 1, 8, 0, 0, 0, 0, 0, 0], [1, 8, 1, 1, 0, 1, 8, 8, 1, 8], [8, 2, 8, 1, 0, 8, 1, 8, 2, 8], [1, 1, 1, 8, 0, 8, 8, 8, 8, 1], [8, 1, 8, 8, 0, 8, 1, 2, 8, 2], [0, 0, 0, 0, 0, 8, 8, 8, 1, 8], [0, 0, 0, 0, 0, 1, 1, 8, 1, 8], [0, 8, 2, 2, 0, 8, 1, 1, 8, 2], [0, 2, 2, 1, 0, 0, 0, 0, 0, 0], [0, 2, 1, 8, 0, 0, 0, 0, 0, 0]],
                        [[2, 8, 8, 8, 0, 0, 0, 0, 0, 0], [8, 8, 1, 8, 0, 0, 0, 0, 0, 0], [1, 8, 8, 8, 0, 0, 0, 0, 0, 0], [8, 8, 8, 2, 0, 0, 1, 8, 8, 2], [8, 2, 8, 1, 0, 0, 8, 8, 1, 8], [8, 1, 8, 8, 0, 0, 8, 2, 8, 8], [0, 0, 0, 0, 0, 0, 8, 8, 8, 1], [0, 0, 0, 0, 0, 0, 1, 8, 8, 8], [0, 0, 0, 0, 0, 0, 8, 8, 1, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 9, 9, 0, 0, 0, 0, 0, 0, 0], [0, 6, 6, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 4, 0, 0, 0], [0, 0, 0, 0, 0, 7, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 4, 8, 0, 0, 0, 0, 0, 0], [0, 0, 9, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    ]

                elif num_objects == 1:
                    examples = [
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[2, 8, 3, 0, 0, 0, 0], [8, 3, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 2], [0, 0, 0, 0, 1, 2, 4], [0, 0, 0, 1, 2, 4, 0], [0, 0, 1, 2, 4, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 4, 4, 0, 0, 0, 0], [0, 0, 0, 4, 6, 4, 0, 0, 0, 0], [0, 0, 0, 4, 4, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 0, 0, 0], [0, 0, 0, 2, 7, 7, 2, 0, 0, 0], [0, 0, 0, 2, 7, 7, 2, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 3, 3, 3, 0, 0, 0, 0], [0, 0, 3, 1, 1, 3, 0, 0, 0, 0], [0, 0, 3, 1, 1, 3, 0, 0, 0, 0], [0, 0, 3, 3, 3, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 8, 2, 8, 0, 0, 0, 0, 0], [0, 0, 8, 2, 2, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 8, 8, 2, 2, 8, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 4, 1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 4, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 6, 6, 3, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 3, 3, 3, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 6, 6, 6, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 6, 6, 6, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 8, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 8, 8, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 8, 8, 8, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 8, 8, 8, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 9, 3, 0, 0], [0, 0, 7, 8, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0], [0, 4, 6, 0, 0, 0], [0, 2, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 3, 6, 0, 0], [0, 0, 5, 2, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 3, 1, 0, 0], [0, 0, 2, 5, 0, 0], [0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 4, 4, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 4, 4, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 4, 6, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0], [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0], [0, 0, 3, 3, 8, 8, 3, 3, 0, 0, 0, 0, 0], [0, 0, 3, 3, 8, 8, 3, 3, 0, 0, 0, 0, 0], [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0], [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]                        
                    ]
            elif get_objects == 'get_objects5':
                if num_objects == 3:
                    examples = [
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8], [8, 8, 8, 6, 6, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8], [8, 8, 8, 6, 6, 8, 8, 8, 6, 8, 8, 6, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8], [8, 8, 8, 8, 8, 8, 8, 3, 6, 6, 6, 6, 3, 8, 8], [8, 8, 3, 3, 3, 3, 8, 3, 6, 4, 4, 6, 3, 8, 8], [8, 8, 3, 6, 6, 3, 8, 3, 6, 4, 4, 6, 3, 8, 8], [8, 8, 3, 6, 6, 3, 8, 3, 6, 4, 4, 6, 3, 8, 8], [8, 8, 3, 3, 3, 3, 8, 3, 6, 6, 6, 6, 3, 8, 8], [8, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 6, 6, 6, 6, 3, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 6, 6, 6, 6, 3, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 6, 6, 6, 6, 3, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 6, 6, 6, 6, 3, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8]],
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 8, 8, 8, 8], [8, 8, 6, 6, 6, 6, 8, 8, 6, 6, 6, 8, 8, 8, 8], [8, 8, 6, 8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 6, 8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 8, 8, 8], [8, 8, 6, 6, 6, 6, 8, 8, 8, 6, 6, 6, 8, 8, 8], [8, 8, 6, 8, 8, 6, 8, 8, 8, 6, 8, 6, 8, 8, 8], [8, 8, 6, 8, 8, 6, 8, 8, 8, 6, 8, 6, 8, 8, 8], [8, 8, 6, 6, 6, 6, 8, 8, 8, 6, 8, 6, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8], [8, 8, 8, 8, 6, 6, 8, 8, 6, 6, 6, 8, 8, 8, 8], [8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
                        [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3]],
                        [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 1, 1, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 2, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 8, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 4, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0], [0, 4, 0, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 7, 0, 0, 0, 0, 0], [0, 0, 7, 1, 7, 0, 0, 0, 0], [0, 0, 0, 7, 0, 0, 0, 0, 0]]
                    ]

                elif num_objects == 2:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0], [0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0], [0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0], [0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0], [0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0], [0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0], [0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 8, 3, 0], [0, 0, 0, 8, 3, 0, 0], [0, 0, 8, 3, 0, 0, 0], [0, 8, 3, 0, 0, 0, 4], [8, 3, 0, 0, 0, 4, 0], [3, 0, 0, 0, 4, 0, 0], [0, 0, 0, 4, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 7, 0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 2, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0], [7, 7, 2, 7, 7, 0, 0, 0, 0, 0, 0, 0], [0, 2, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 7, 0, 2, 0, 2, 0, 7, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0, 2, 7, 2, 0, 0], [0, 0, 0, 0, 0, 0, 7, 7, 2, 7, 7, 0], [0, 0, 0, 0, 0, 0, 0, 2, 7, 2, 0, 0], [0, 0, 0, 0, 0, 0, 2, 0, 7, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 6, 0, 8, 0, 6, 0, 0, 0, 0, 0, 0], [0, 0, 6, 8, 6, 0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 6, 8, 8, 0, 0, 0, 0, 0, 0], [0, 0, 6, 8, 6, 0, 0, 0, 0, 0, 0, 0], [0, 6, 0, 8, 0, 6, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 8, 0, 6, 0], [0, 0, 0, 0, 0, 0, 0, 6, 8, 6, 0, 0], [0, 0, 0, 0, 0, 0, 8, 8, 6, 8, 8, 0], [0, 0, 0, 0, 0, 0, 0, 6, 8, 6, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 8, 0, 6, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 4, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0], [0, 4, 0, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 7, 0, 0], [0, 0, 0, 0, 0, 7, 1, 7, 0], [0, 0, 0, 0, 0, 0, 7, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]                        
                    ]

                elif num_objects == 1:
                    examples = [
                        [[0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 2, 0], [0, 2, 2, 2, 2, 2, 0], [0, 2, 2, 2, 2, 2, 0], [0, 2, 2, 2, 2, 2, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 7, 7, 0, 0, 0, 0], [0, 0, 0, 6, 8, 8, 6, 0, 0, 0], [0, 0, 7, 8, 4, 4, 8, 7, 0, 0], [0, 0, 7, 8, 4, 4, 8, 7, 0, 0], [0, 0, 0, 6, 8, 8, 6, 0, 0, 0], [0, 0, 0, 0, 7, 7, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 3, 6, 5, 3, 0, 0, 0, 0], [0, 0, 5, 2, 2, 6, 0, 0, 0, 0], [0, 0, 6, 2, 2, 5, 0, 0, 0, 0], [0, 0, 3, 5, 6, 3, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 4, 4, 8, 4, 0, 0, 0], [0, 0, 8, 8, 3, 3, 4, 0, 0, 0], [0, 0, 0, 4, 3, 3, 8, 8, 0, 0], [0, 0, 0, 4, 8, 4, 4, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]                        
                    ]

        if len(examples) == 0:
            return None
        else:
            random_idx = np.random.choice(np.arange(len(examples)))
            return self.DSL.Grid(tuple(tuple(row) for row in examples[random_idx]))

    @staticmethod
    def generate_multicolor_8adj(num_objects):
        
        while True:
            # 1) select bg color
            bg_color = np.random.choice([0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

            # 2) randomly pick number of objects colors between 2 and 5 inclusively.
            num_colors = np.random.randint(2, 6)

            # 3) randomly select the objects' dimensions, considering num_objects
            if num_objects > 6:
                width = np.random.choice(np.arange(20, 31))
                height = np.random.choice(np.arange(20, 31))
            else:
                width = np.random.choice(np.arange(15, 31))
                height = np.random.choice(np.arange(15, 31))

            obj_widths, obj_heights = ObjectRecombinerGenerator.get_min_max_sizes(width, height, num_objects)
            if obj_widths is None:
                continue

            # 4) generate 8-way connected shapes
            object_list = []
            for obj_idx in range(num_objects):
                colors = np.random.choice([i for i in range(1,10) if i != bg_color], size=num_colors)
                new_obj = ObjectRecombinerGenerator.generate_8way_adj_object(obj_widths[obj_idx], obj_heights[obj_idx], colors)
                object_list.append(new_obj)

            # 5) paste them onto the target grid
            target_grid = tuple(tuple(bg_color for _ in range(width)) for _ in range(height))
            output = ObjectRecombinerGenerator.paste_objects_on_grid(target_grid, object_list, False)

            if output is not None:
                return output

    def generate_random_recombiner_grids(self, get_objects, num_objects, hp):
        
        if get_objects == 'get_objects3':
            output_grid =  ObjectRecombinerGenerator.generate_simple_shape(num_objects)
        elif get_objects == 'get_objects4':
            output_grid =  ObjectRecombinerGenerator.generate_multicolor_4adj(num_objects)
        elif get_objects == 'get_objects5':
            output_grid =  ObjectRecombinerGenerator.generate_multicolor_8adj(num_objects)
        
        if output_grid is None:
            return None
        else:
            return hp.Grid(output_grid)

    def generate_recombiner_grids(self, get_objects, num_objects, k, hp):
        input_grids = []

        for _ in range(k):
            a = np.random.uniform()
            if a < 0.2:
                # Use pre-determined grids from training or validation
                input_grid = self.select_grid(get_objects, num_objects)

                if input_grid is None:
                    # Randomly generate grids
                    input_grid = self.generate_random_recombiner_grids(get_objects, num_objects, hp)
                else:
                    input_grid = ObjectRecombinerGenerator.augment_grid(input_grid, hp)
            else:
                # Randomly generate grids
                input_grid = self.generate_random_recombiner_grids(get_objects, num_objects, hp)

            input_grids.append(input_grid)

        return input_grids

    def generate(self, hp, k=6):
        # generate 1, 2 or 3 objects with get_objects3, 4 or 5
        # split bottom/up, left/right
        # apply set fg color to one half
        # recombine with v/hconcat
        # apply_to_grid or crop (if num_objects == 1)

        num_objects = np.random.choice([1, 2, 3])
        get_objects = np.random.choice(['get_objects3', 'get_objects4', 'get_objects5'])

        a = np.random.uniform()
        if a < 0.5:
            first_half = 'tophalf'
            second_half = 'bottomhalf'
            concat = 'vconcat'
        else:
            first_half = 'lefthalf'
            second_half = 'righthalf'
            concat = 'hconcat'

        set_fg_color_first = np.random.choice([True, False])

        set_fg_color = 'set_fg_color%i' % np.random.choice(np.arange(1, 10))

        if set_fg_color_first:
            program = [get_objects, first_half, second_half, 'NEW_LEVEL', 'IDENTITY', set_fg_color, 'IDENTITY', 'NEW_LEVEL', 'IDENTITY', concat, 'NEW_LEVEL', 'for_each']
        else:
            program = [get_objects, first_half, second_half, 'NEW_LEVEL', 'IDENTITY', 'IDENTITY', set_fg_color, 'NEW_LEVEL', 'IDENTITY', concat, 'NEW_LEVEL', 'for_each']
        
        if num_objects == 1:
            a = np.random.uniform()
            if a < 0.5:
                program.append('NEW_LEVEL')
                program.append('apply_to_grid')
        else:
            program.append('NEW_LEVEL')
            program.append('apply_to_grid')

        program.append('EOS')
        print("Final program list: ", program)

        task_invalid = True
        while task_invalid:

            
            try:
                input_grids = self.generate_recombiner_grids(get_objects, num_objects, k, hp)

                # compile the program string and generate the output grids from the inputs
                NUM_SPECIAL_TOKENS = 4
                def convert_to_label_seq(program):
                    label_seq = []
                    for token in program:
                        if token == 'NEW_LEVEL':
                            token_idx = 1
                        elif token == 'IDENTITY':
                            token_idx = 2
                        elif token == 'EOS':
                            token_idx = 3
                        else:
                            token_idx = hp.prim_indices[token] + NUM_SPECIAL_TOKENS

                        label_seq.append(token_idx)

                    return label_seq

                label_seq = convert_to_label_seq(program)
                
                # run the program interpreter on the task
                program_tree = pi.generate_syntax_trees(np.array(label_seq), hp)
                task_desc = pi.write_program(program_tree, np.array(label_seq), hp)
            
                print("==> Object selector generator, generating task program:")
                print(task_desc)

                program_func = pi.compile_program(task_desc, hp.semantics)

                example_grid_set = []
                for example_idx in range(len(input_grids)):

                    grid_inp = input_grids[example_idx]

                    # print("==> Processing input grid:")
                    # viz.draw_single_grid(grid_inp.cells)

                    pred_output = program_func(grid_inp)
                    if isinstance(pred_output, list):
                        pred_output = pred_output[0]

                    # print("==> pred_output:")
                    # print(pred_output)

                    # print("==> Generated output:")
                    #viz.draw_grid_pair(grid_inp.cells, pred_output.cells)

                    # Check if grid_inp and pred_output are exactly the same
                    if not np.array_equal(grid_inp.cells, pred_output.cells):
                        example_grid_set.append((grid_inp.cells, pred_output.cells))

                if len(example_grid_set) > 0:
                    task_invalid = False

            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Stopping the program.")
                raise  # Re-raise the KeyboardInterrupt to stop the program
            except IndexError:
                pass
            except Exception as e:
                print(f"==> Task FAILED with an exception:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Traceback:")
                import traceback
                traceback.print_exc()

        # The ground truth label seq must not include EOS, and has NUM_SPECIAL_TOKENS 3 instead of 4.
        ground_truth_lbl_seq = [token-1 for token in label_seq[:-1]]
        return example_grid_set, task_desc, ground_truth_lbl_seq
    