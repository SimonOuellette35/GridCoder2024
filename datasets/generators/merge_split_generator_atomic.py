import numpy as np
import random
import utils.validation_utils as val_utils
import utils.grid_utils as grid_utils

NUM_SPECIAL_TOKENS = 3


class MergeSplitGenerator:

    def __init__(self, DSL, validation=False):
        self.DSL = DSL
        self.NEW_LEVEL = 0
        self.IDENTITY = 1
        self.validation = validation

    def pick_cellwise(self):
        return np.random.choice([
            'cellwiseOR',
            'cellwiseAND',
            'cellwiseNOR',
            'cellwiseXOR',
            'cellwiseDifference'
        ])

    def generate_starting_grid(self, n_splits, split_type):
        def get_color_strategy(n_splits):
            if n_splits == 2:
                # 3 possibilities:
                # 1. 2 distinct colors
                # 2. both grids are of the same color
                # 3. multicolor, fully randomized grids (return None)
                a = np.random.uniform()
                if a < 0.33:
                    return np.random.choice(np.arange(1, 10), 2, replace=False)
                elif a < 0.66:
                    c = np.random.choice(np.arange(1, 10))
                    return np.array([c, c])
                else:
                    return None
            else:
                return np.random.choice(np.arange(1, 10), n_splits, replace=False)

        if split_type == 'horizontal':
            sub_grid_nrows = np.random.choice(np.arange(3, 31))
            sub_grid_ncols = np.random.choice(np.arange(3, (30 // n_splits) + 1))
        elif split_type == 'vertical':
            sub_grid_ncols = np.random.choice(np.arange(3, 31))
            sub_grid_nrows = np.random.choice(np.arange(3, (30 // n_splits) + 1))
        else:
            sub_grid_ncols = sub_grid_nrows = np.random.choice(np.arange(3, 16))


        if self.validation:
            return self.DSL.Grid(grid_utils.get_merge_split_validation(split_type, n_splits))
        
        colors = get_color_strategy(n_splits)
        threshold = np.random.uniform(0.1, 0.75)
        sub_grids = []
        for sub_grid_idx in range(n_splits):
            if colors is None:
                c = None
            else:
                c = colors[sub_grid_idx]

            # generate a sub-grid of the given dimensions that applies the selected color strategy.
            tmp_grid = np.zeros((sub_grid_nrows, sub_grid_ncols))
            for i in range(tmp_grid.shape[0]):
                for j in range(tmp_grid.shape[1]):
                    a = np.random.uniform()

                    if a < threshold:
                        if c is None:
                            c = np.random.choice(np.arange(1, 10))

                        tmp_grid[i, j] = c

            sub_grids.append(tmp_grid)

        add_separator = False
        a = np.random.uniform()
        if a >= 0.3:
            add_separator = True

        if split_type == 'horizontal':
            if len(sub_grids) * len(sub_grids[0][0]) > 25:
                add_separator = False

            if add_separator:
                sep_color = np.random.choice(np.arange(1, 10))
                separator = np.ones((len(sub_grids[0]), 1)) * sep_color
                output_grid = sub_grids[0]
                for i in range(1, len(sub_grids)):
                    output_grid = np.concatenate((output_grid, separator, sub_grids[i]), axis=1)
            else:
                output_grid = sub_grids[0]
                for i in range(1, len(sub_grids)):
                    output_grid = np.concatenate((output_grid, sub_grids[i]), axis=1)

        elif split_type == 'vertical':
            if len(sub_grids) * len(sub_grids[0]) > 25:
                add_separator = False

            if add_separator:
                sep_color = np.random.choice(np.arange(1, 10))
                separator = np.ones((1, len(sub_grids[0][0]))) * sep_color
                output_grid = sub_grids[0]
                for i in range(1, len(sub_grids)):
                    output_grid = np.concatenate((output_grid, separator, sub_grids[i]), axis=0)
            else:
                output_grid = sub_grids[0]
                for i in range(1, len(sub_grids)):
                    output_grid = np.concatenate((output_grid, sub_grids[i]), axis=0)

        else:
            if len(sub_grids) * len(sub_grids[0]) > 25:
                add_separator = False

            if add_separator:
                sep_color = np.random.choice(np.arange(1, 10))
                vert_sep = np.ones((len(sub_grids[0]), 1)) * sep_color
                horiz_sep = np.ones((1, len(sub_grids[0][0]) * 2 + 1)) * sep_color

                top_half = np.concatenate((sub_grids[0], vert_sep, sub_grids[1]), axis=1)
                bottom_half = np.concatenate((sub_grids[2], vert_sep, sub_grids[3]), axis=1)
                output_grid = np.concatenate((top_half, horiz_sep, bottom_half), axis=0)
            else:
                top_half = np.concatenate((sub_grids[0], sub_grids[1]), axis=1)
                bottom_half = np.concatenate((sub_grids[2], sub_grids[3]), axis=1)
                output_grid = np.concatenate((top_half, bottom_half), axis=0)

        return self.DSL.Grid(tuple(tuple(inner) for inner in output_grid.astype(int)))


    # splitting a grid and merging by priority or some other logic:
    #   - 2-way, 3-way, 4-way
    #   - Horizontally, Vertically, Quadrants
    #   - All permutations of priority orders, or merging logics (OR, NOR, AND, etc.)
    #   - K = 6
    #   - Make sure enough pixels in the randomly generated grids
    #   - Make sure in most cases the priority order can be exactly resolved (no under-specifications)
    def generate(self, k=6):

        def generate_task_program_flat(n_splits, split_type):

            if n_splits == 2:
                cellwise = self.pick_cellwise()
                s = np.random.choice(np.arange(2))
                cellwise_func = self.DSL.semantics[cellwise]
                if split_type == 'horizontal':
                    token1 = 'lefthalf'
                    token2 = 'righthalf'
                else:
                    token1 = 'tophalf'
                    token2 = 'bottomhalf'

                if s == 0:
                    desc = '%s(%s, %s)' % (cellwise, token1, token2)
                    func = lambda g: cellwise_func(self.DSL.semantics[token1](g))(self.DSL.semantics[token2](g))
                    label_seq = [
                        self.DSL.prim_indices[token1] + NUM_SPECIAL_TOKENS,
                        self.DSL.prim_indices[token2] + NUM_SPECIAL_TOKENS,
                        self.NEW_LEVEL,
                        self.DSL.prim_indices[cellwise] + NUM_SPECIAL_TOKENS
                    ]

                else:
                    desc = '%s(%s, %s)' % (cellwise, token2, token1)
                    func = lambda g: cellwise_func(self.DSL.semantics[token2](g))(self.DSL.semantics[token1](g))
                    label_seq = [
                        self.DSL.prim_indices[token2] + NUM_SPECIAL_TOKENS,
                        self.DSL.prim_indices[token1] + NUM_SPECIAL_TOKENS,
                        self.NEW_LEVEL,
                        self.DSL.prim_indices[cellwise] + NUM_SPECIAL_TOKENS
                    ]
            else:
                if n_splits == 3:
                    if split_type == 'horizontal':
                        priority = ['leftthird', 'hcenterthird', 'rightthird']
                    else:
                        priority = ['topthird', 'vcenterthird', 'bottomthird']

                    random.shuffle(priority)
                    desc = 'cellwiseOR(%s, cellwiseOR(%s, %s))' % (priority[0], priority[1], priority[2])

                    first_func = self.DSL.semantics[priority[0]]
                    second_func = self.DSL.semantics[priority[1]]
                    third_func = self.DSL.semantics[priority[2]]
                    func = lambda g: self.DSL.cellwiseOR(first_func(g), self.DSL.cellwiseOR(second_func(g), third_func(g)))

                    label_seq = [
                        self.DSL.prim_indices[priority[0]] + NUM_SPECIAL_TOKENS,
                        self.DSL.prim_indices[priority[1]] + NUM_SPECIAL_TOKENS,
                        self.DSL.prim_indices[priority[2]] + NUM_SPECIAL_TOKENS,
                        self.NEW_LEVEL,
                        self.IDENTITY,
                        self.DSL.prim_indices['cellwiseOR'] + NUM_SPECIAL_TOKENS,
                        self.NEW_LEVEL,
                        self.DSL.prim_indices['cellwiseOR'] + NUM_SPECIAL_TOKENS,
                    ]

                elif n_splits == 4:
                    if split_type == 'horizontal':
                        priority = ['hfirstfourth', 'hsecondfourth', 'hthirdfourth', 'hlastfourth']
                    else:
                        priority = ['vfirstfourth', 'vsecondfourth', 'vthirdfourth', 'vlastfourth']

                    random.shuffle(priority)
                    desc = 'cellwiseOR(%s, cellwiseOR(%s, cellwiseOR(%s, %s)))' % (
                        priority[0], priority[1], priority[2], priority[3])

                    first_func = self.DSL.semantics[priority[0]]
                    second_func = self.DSL.semantics[priority[1]]
                    third_func = self.DSL.semantics[priority[2]]
                    fourth_func = self.DSL.semantics[priority[3]]
                    func = lambda g: self.DSL.cellwiseOR(first_func(g), self.DSL.cellwiseOR(second_func(g), self.DSL.cellwiseOR(third_func(g), fourth_func(g))))

                    label_seq = [
                        self.DSL.prim_indices[priority[0]] + NUM_SPECIAL_TOKENS,
                        self.DSL.prim_indices[priority[1]] + NUM_SPECIAL_TOKENS,
                        self.DSL.prim_indices[priority[2]] + NUM_SPECIAL_TOKENS,
                        self.DSL.prim_indices[priority[3]] + NUM_SPECIAL_TOKENS,
                        self.NEW_LEVEL,
                        self.IDENTITY,
                        self.IDENTITY,
                        self.DSL.prim_indices['cellwiseOR'] + NUM_SPECIAL_TOKENS,
                        self.NEW_LEVEL,
                        self.IDENTITY,
                        self.DSL.prim_indices['cellwiseOR'] + NUM_SPECIAL_TOKENS,
                        self.NEW_LEVEL,
                        self.DSL.prim_indices['cellwiseOR'] + NUM_SPECIAL_TOKENS
                    ]

            return func, desc, label_seq

        def generate_task_program(n_splits, split_type):
            if split_type == 'horizontal':
                func, desc, label_seq = generate_task_program_flat(n_splits, split_type)
            elif split_type == 'vertical':
                func, desc, label_seq = generate_task_program_flat(n_splits, split_type)
            elif split_type == 'quadrants':
                choices = ['first_quadrant', 'second_quadrant', 'third_quadrant', 'fourth_quadrant']
                random.shuffle(choices)

                first, second, third, fourth = choices
                desc = 'cellwiseOR(%s, cellwiseOR(%s, cellwiseOR(%s, %s)))' % (first, second, third, fourth)

                first_func = self.DSL.semantics[first]
                second_func = self.DSL.semantics[second]
                third_func = self.DSL.semantics[third]
                fourth_func = self.DSL.semantics[fourth]

                func = lambda g: self.DSL.cellwiseOR(first_func(g), self.DSL.cellwiseOR(second_func(g), self.DSL.cellwiseOR(third_func(g), fourth_func(g))))

                label_seq = [
                    self.DSL.prim_indices[first] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[second] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[third] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[fourth] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.IDENTITY,
                    self.IDENTITY,
                    self.DSL.prim_indices['cellwiseOR'] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.IDENTITY,
                    self.DSL.prim_indices['cellwiseOR'] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices['cellwiseOR'] + NUM_SPECIAL_TOKENS
                ]

            color_choices = ['', 'set_fg_color']
            choice = np.random.choice(color_choices)
            if choice == '' or n_splits > 2:
                return func, desc, label_seq
            else:
                c = np.random.choice(np.arange(1, 10))
                desc = "set_fg_color%i(%s)" % (c, desc)
                new_func = lambda g: self.DSL.set_fg_color(func(g), c)
                label_seq.append(self.NEW_LEVEL)
                label_seq.append(self.DSL.prim_indices['set_fg_color%i' % c] + NUM_SPECIAL_TOKENS)

                return new_func, desc, label_seq

        def generate_splitting_pattern():
            a = np.random.uniform()

            if a <= 0.33:
                n = np.random.choice(np.arange(2, 5))
                return n, 'vertical'
            elif a <= 0.66:
                n = np.random.choice(np.arange(2, 5))
                return n, 'horizontal'
            else:
                return 4, 'quadrants'

        # n_splits can be 2, 3, 4
        # split_type can be quadrants (only works for n_splits == 4), vertical, horizontal
        n_splits, split_type = generate_splitting_pattern()

        # task_program is a function that represents the actual task. It's the ground truth.
        task_func, task_desc, label_seq = generate_task_program(n_splits, split_type)

        valid_grid_set = False
        while not valid_grid_set:
            example_grid_set = []

            valid_grid_set = True
            for example_idx in range(k):
                starting_grid = self.generate_starting_grid(n_splits, split_type)
                output_grid = task_func(starting_grid)

                # make sure we haven't generated a grid that is exactly identical to one that has been
                # generated before
                if starting_grid == output_grid:
                    valid_grid_set = False
                    break

                # don't support empty outputs
                if val_utils.is_empty([output_grid.cells], self.DSL):
                    valid_grid_set = False
                    break

                # make sure the grid shape is valid
                if not val_utils.is_valid_shape([output_grid.cells], 30):
                    valid_grid_set = False
                    break

                example_grid_set.append((starting_grid.cells, output_grid.cells))

        return example_grid_set, task_desc, label_seq