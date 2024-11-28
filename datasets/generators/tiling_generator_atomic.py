import numpy as np
import utils.validation_utils as val_utils
import utils.grid_utils as grid_utils
from datasets.task_generator import TaskGenerator

NUM_SPECIAL_TOKENS = 3


class TilingGenerator:

    def __init__(self, DSL, validation=False):
        self.DSL = DSL
        self.NEW_LEVEL = 0
        self.IDENTITY = 1
        self.task_generator = TaskGenerator(DSL)
        self.validation = validation

    def generate_starting_grid(self, shape, crop_prim, square=False):

        def sample_grid(crop_prim, num_rows_max=30, num_cols_max=30, square=False):
            
            def fill_grid(tmp_grid, tmp_palette):
                threshold = np.random.uniform(0.1, 1.)

                for i in range(tmp_grid.shape[0]):
                    for j in range(tmp_grid.shape[1]):
                        a = np.random.uniform()

                        if a < threshold:
                            color = np.random.choice(tmp_palette)
                            tmp_grid[i, j] = color

                return tmp_grid

            def sampleGridPatch(max_nrows=30, max_ncols=30, square=False):
                
                filtered_samples = []
                while len(filtered_samples) == 0:
                    grid_samples = grid_utils.get_samples(self.DSL, self.validation)

                    def augment(grid):
                        num_rotations = np.random.choice(np.arange(4))
                        for _ in range(num_rotations):
                            grid = self.DSL.rot90(grid)

                        return grid

                    filtered_samples = []
                    for gi in grid_samples:
                        tmp = augment(gi)

                        if len(tmp.cells) > max_nrows:
                            continue

                        if len(tmp.cells[0]) > max_ncols:
                            continue

                        if square:
                            if crop_prim == 'lefthalf' or crop_prim == 'righthalf':
                                # num_cols = 2*nrows
                                if len(tmp.cells[0]) != 2*len(tmp.cells):
                                    tmp = self.DSL.rot90(tmp)

                                    if len(tmp.cells[0]) != 2 * len(tmp.cells):
                                        continue
                            elif crop_prim == 'tophalf' or crop_prim == 'bottomhalf':
                                # num_rows = 2*ncols
                                if len(tmp.cells) != 2*len(tmp.cells[0]):
                                    tmp = self.DSL.rot90(tmp)

                                    if len(tmp.cells) != 2 * len(tmp.cells[0]):
                                        continue
                            else:
                                if len(tmp.cells) != len(tmp.cells[0]):
                                    continue

                        filtered_samples.append(tmp)

                idx = np.random.choice(np.arange(len(filtered_samples)))
                return filtered_samples[idx]

            MIN_DIM = 2
            # account for crop_prim
            if crop_prim == 'righthalf' or crop_prim == 'lefthalf':
                if square:
                    # must still be square after applying righthalf/lefthalf! So: numcols must be equal to numrows * 2!
                    max_row_tmp = min(30, num_rows_max * 2)
                    ncols = np.random.choice(np.arange(MIN_DIM * 2, max_row_tmp + 1, 2))
                    nrows = ncols // 2
                else:
                    nrows = np.random.choice(np.arange(MIN_DIM, num_rows_max + 1))

                    max_col_tmp = min(30, num_cols_max * 2)
                    ncols = np.random.choice(np.arange(MIN_DIM, max_col_tmp + 1))

            elif crop_prim == 'tophalf' or crop_prim == 'bottomhalf':
                if square:
                    # must still be square after applying tophalf/bottomhalf! So: numrows must be equal to numcols * 2!
                    max_col_tmp = min(30, num_cols_max * 2)
                    nrows = np.random.choice(np.arange(MIN_DIM * 2, max_col_tmp + 1, 2))
                    ncols = nrows // 2
                else:
                    max_row_tmp = min(30, num_rows_max * 2)
                    nrows = np.random.choice(np.arange(MIN_DIM, max_row_tmp + 1))

                    ncols = np.random.choice(np.arange(MIN_DIM, num_cols_max + 1))
            else:
                if square:
                    nrows = ncols = np.random.choice(np.arange(MIN_DIM, num_rows_max + 1))
                else:
                    nrows = np.random.choice(np.arange(MIN_DIM, num_rows_max + 1))
                    ncols = np.random.choice(np.arange(MIN_DIM, num_cols_max + 1))

            if self.validation:
                return self.DSL.Grid(grid_utils.get_tiling_validation())

            a = np.random.uniform()
            if a < 0.3:
                # pre-generated
                #print("==> Finding pre-generated grid with max_nrows = %i, max_ncols = %i, square = %s" % (num_rows_max, num_cols_max, square))
                grid = sampleGridPatch(max_nrows=num_rows_max, max_ncols=num_cols_max, square=square)

                return grid
            else:
                #print("==> Generated nrows = %i, ncols = %i" % (nrows, ncols))
                # randomized
                grid = np.zeros((nrows, ncols))
                grid_ncolors = np.random.choice(np.arange(1, 5))
                grid_palette = np.random.choice(np.arange(1, 10), grid_ncolors, replace=False)

                grid = fill_grid(grid, grid_palette)

                return self.DSL.Grid(tuple(tuple(inner) for inner in grid.astype(int)))

        MAX_SUBGRID_DIM = 6
        if shape[0] == 1:
            # horizontal
            # make sure the number of columns is limited so that after concatenation they're still a valid grid.
            sub_grid_max = min(MAX_SUBGRID_DIM, 30 // shape[1])
            output_grid = sample_grid(crop_prim, num_cols_max=sub_grid_max, num_rows_max=MAX_SUBGRID_DIM, square=square)
        elif shape[1] == 1:
            # vertical
            #  make sure the number of rows is limited so that after concatenation they're still a valid grid.
            sub_grid_max = min(MAX_SUBGRID_DIM, 30 // shape[0])
            output_grid = sample_grid(crop_prim, num_rows_max=sub_grid_max, num_cols_max=MAX_SUBGRID_DIM, square=square)
        else:
            # quadrant
            # make sure sub-grid dimension respects the constraints of shape. They should also be square.
            sub_grid_max = min(MAX_SUBGRID_DIM, 30 // shape[0])
            output_grid = sample_grid(crop_prim, num_rows_max=sub_grid_max, num_cols_max=sub_grid_max, square=True)

        return output_grid

    def generate_task_program(self, shape, dir, crop_prim=None):
        def process_prim(g, prim_str):
            if prim_str == 'identity':
                return g
            else:
                return self.DSL.semantics[prim_str](g)

        def get_dir():
            if dir == 'hconcat':
                return self.DSL.hconcat
            elif dir == 'vconcat':
                return self.DSL.vconcat

        transforms = []
        if (dir == 'hconcat' and shape == (1, 2)) or (dir == 'vconcat' and shape == (2, 1)):
            prim1 = self.select_tiling_primitive()
            prim2 = self.select_tiling_primitive()

            transforms.append(prim1)
            transforms.append(prim2)

            task_desc = "%s(%s, %s)" % (dir, prim1, prim2)
            task_func = lambda g: get_dir()(process_prim(g, prim1), process_prim(g, prim2))
            
            if prim1 == 'identity':
                prim1_idx = self.IDENTITY
            else:
                prim1_idx = self.DSL.prim_indices[prim1] + NUM_SPECIAL_TOKENS
            
            if prim2 == 'identity':
                prim2_idx = self.IDENTITY
            else:
                prim2_idx = self.DSL.prim_indices[prim2] + NUM_SPECIAL_TOKENS

            if crop_prim is not None and crop_prim != 'identity':
                label_seq = [
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    prim1_idx,
                    prim2_idx,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS
                ]
            else:
                label_seq = [
                    prim1_idx,
                    prim2_idx,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS
                ]
        elif (dir == 'hconcat' and shape == (1, 3)) or (dir == 'vconcat' and shape == (3, 1)):
            prim1 = self.select_tiling_primitive()
            prim2 = self.select_tiling_primitive()
            prim3 = self.select_tiling_primitive()

            transforms.append(prim1)
            transforms.append(prim2)
            transforms.append(prim3)

            task_desc = "%s(%s, %s(%s, %s))" % (dir, prim1, dir, prim2, prim3)
            task_func = lambda g: get_dir()(process_prim(g, prim1), get_dir()(process_prim(g, prim2),
                                                                              process_prim(g, prim3)))
            
            if prim1 == 'identity':
                prim1_idx = self.IDENTITY
            else:
                prim1_idx = self.DSL.prim_indices[prim1] + NUM_SPECIAL_TOKENS

            if prim2 == 'identity':
                prim2_idx = self.IDENTITY
            else:
                prim2_idx = self.DSL.prim_indices[prim2] + NUM_SPECIAL_TOKENS

            if prim3 == 'identity':
                prim3_idx = self.IDENTITY
            else:
                prim3_idx = self.DSL.prim_indices[prim3] + NUM_SPECIAL_TOKENS

            if crop_prim is not None and crop_prim != 'identity':
                label_seq = [
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    prim1_idx,
                    prim2_idx,
                    prim3_idx,
                    self.NEW_LEVEL,
                    self.IDENTITY,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS
                ]
            else:
                label_seq = [
                    prim1_idx,
                    prim2_idx,
                    prim3_idx,
                    self.NEW_LEVEL,
                    self.IDENTITY,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS
                ]

        elif (dir == 'hconcat' and shape == (1, 4)) or (dir == 'vconcat' and shape == (4, 1)):
            prim1 = self.select_tiling_primitive()
            prim2 = self.select_tiling_primitive()
            prim3 = self.select_tiling_primitive()
            prim4 = self.select_tiling_primitive()

            transforms.append(prim1)
            transforms.append(prim2)
            transforms.append(prim3)
            transforms.append(prim4)

            task_desc = "%s(%s, %s(%s, %s(%s, %s)))" % (dir, prim1, dir, prim2, dir, prim3, prim4)
            task_func = lambda g: get_dir()(process_prim(g, prim1), get_dir()(process_prim(g, prim2),
                                                                    get_dir()(process_prim(g, prim3),
                                                                              process_prim(g, prim4))))

            if prim1 == 'identity':
                prim1_idx = self.IDENTITY
            else:
                prim1_idx = self.DSL.prim_indices[prim1] + NUM_SPECIAL_TOKENS

            if prim2 == 'identity':
                prim2_idx = self.IDENTITY
            else:
                prim2_idx = self.DSL.prim_indices[prim2] + NUM_SPECIAL_TOKENS

            if prim3 == 'identity':
                prim3_idx = self.IDENTITY
            else:
                prim3_idx = self.DSL.prim_indices[prim3] + NUM_SPECIAL_TOKENS

            if prim4 == 'identity':
                prim4_idx = self.IDENTITY
            else:
                prim4_idx = self.DSL.prim_indices[prim4] + NUM_SPECIAL_TOKENS

            if crop_prim is not None and crop_prim != 'identity':
                label_seq = [
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    prim1_idx,
                    prim2_idx,
                    prim3_idx,
                    prim4_idx,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS
                ]
            else:
                label_seq = [
                    prim1_idx,
                    prim2_idx,
                    prim3_idx,
                    prim4_idx,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices[dir] + NUM_SPECIAL_TOKENS
                ]

        else:
            print("==> ERROR in generate_task_program(%s): shape = %s" % (dir, shape))
            exit(-1)

        a = np.random.uniform()
        if a < 0.2:
            post_processing = ['invert_colors', 'set_fg_color1', 'set_fg_color2', 'set_fg_color3', 'set_fg_color4', 'set_fg_color5', 'set_fg_color6', 'set_fg_color7', 'set_fg_color8', 'set_fg_color9']
            random_pp = np.random.choice(post_processing)

            pp_lam = self.DSL.semantics[random_pp]
            pp_task_desc = "%s(%s)" % (random_pp, task_desc)
            
            pp_idx = self.DSL.prim_indices[random_pp]
            label_seq.append(1)
            label_seq.append(pp_idx + NUM_SPECIAL_TOKENS)

            output_func = lambda g: pp_lam(task_func(g))
            return output_func, pp_task_desc, label_seq, transforms
        else:
            return task_func, task_desc, label_seq, transforms

    def select_tiling_primitive(self):
        return np.random.choice(['identity', 'rot90', 'rot180', 'rot270', 'hmirror', 'vmirror'])

    def generate_task_program_quad(self, shape, crop_prim=None):

        def get_prim_idx(prim_str):
            if prim_str == 'identity':
                return self.IDENTITY
            else:
                return self.DSL.prim_indices[prim_str] + NUM_SPECIAL_TOKENS

        def process_prim(g, prim_str):
            if prim_str == 'identity':
                return g
            else:
                return self.DSL.semantics[prim_str](g)

        transforms = []
        if shape == (2, 2):
            prim1 = self.select_tiling_primitive()
            prim2 = self.select_tiling_primitive()
            prim3 = self.select_tiling_primitive()
            prim4 = self.select_tiling_primitive()

            transforms.append(prim1)
            transforms.append(prim2)
            transforms.append(prim3)
            transforms.append(prim4)

            task_desc = "vconcat(hconcat(%s, %s), hconcat(%s, %s))" % (prim1, prim2, prim3, prim4)
            task_func = lambda g: self.DSL.vconcat(self.DSL.hconcat(process_prim(g, prim1), process_prim(g, prim2)),
                                                   self.DSL.hconcat(process_prim(g, prim3), process_prim(g, prim4)))

            
            prim1_idx = get_prim_idx(prim1)
            prim2_idx = get_prim_idx(prim2)
            prim3_idx = get_prim_idx(prim3)
            prim4_idx = get_prim_idx(prim4)

            if crop_prim is not None and crop_prim != 'identity':
                label_seq = [
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    prim1_idx,
                    prim2_idx,
                    prim3_idx,
                    prim4_idx,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices['vconcat'] + NUM_SPECIAL_TOKENS
                ]

            else:
                label_seq = [
                    prim1_idx,
                    prim2_idx,
                    prim3_idx,
                    prim4_idx,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices['vconcat'] + NUM_SPECIAL_TOKENS
                ]

        elif shape == (3, 3):
            prim = []
            for _ in range(9):
                tmp = self.select_tiling_primitive()
                prim.append(tmp)
                transforms.append(tmp)

            task_desc = "vconcat(hconcat(%s, hconcat(%s, %s)), vconcat(hconcat(%s, hconcat(%s, %s)), hconcat(%s, hconcat(%s, %s)))" % (
                prim[0],
                prim[1],
                prim[2],
                prim[3],
                prim[4],
                prim[5],
                prim[6],
                prim[7],
                prim[8]
            )
            task_func = lambda g: self.DSL.vconcat(self.DSL.hconcat(process_prim(g, prim[0]), self.DSL.hconcat(process_prim(g, prim[1]), process_prim(g, prim[2]))),
                                                   self.DSL.vconcat(
                                                       self.DSL.hconcat(process_prim(g, prim[3]), self.DSL.hconcat(process_prim(g, prim[4]), process_prim(g, prim[5]))),
                                                       self.DSL.hconcat(process_prim(g, prim[6]), self.DSL.hconcat(process_prim(g, prim[7]), process_prim(g, prim[8])))
                                                   ))


            prim1_idx = get_prim_idx(prim[0])
            prim2_idx = get_prim_idx(prim[1])
            prim3_idx = get_prim_idx(prim[2])
            prim4_idx = get_prim_idx(prim[3])
            prim5_idx = get_prim_idx(prim[4])
            prim6_idx = get_prim_idx(prim[5])
            prim7_idx = get_prim_idx(prim[6])
            prim8_idx = get_prim_idx(prim[7])
            prim9_idx = get_prim_idx(prim[8])

            if crop_prim is not None and crop_prim != 'identity':
                label_seq = [
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices[crop_prim] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    prim1_idx,
                    prim2_idx,
                    prim3_idx,
                    prim4_idx,
                    prim5_idx,
                    prim6_idx,
                    prim7_idx,
                    prim8_idx,
                    prim9_idx,
                    self.NEW_LEVEL,
                    self.IDENTITY,
                    self.IDENTITY,
                    self.IDENTITY,
                    self.IDENTITY,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.IDENTITY,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.IDENTITY,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices['vconcat'] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices['vconcat'] + NUM_SPECIAL_TOKENS
                ]

            else:
                label_seq = [
                    prim1_idx,
                    prim2_idx,
                    prim3_idx,
                    prim4_idx,
                    prim5_idx,
                    prim6_idx,
                    prim7_idx,
                    prim8_idx,
                    prim9_idx,
                    self.NEW_LEVEL,
                    self.IDENTITY,
                    self.IDENTITY,
                    self.IDENTITY,
                    self.IDENTITY,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.IDENTITY,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.IDENTITY,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices['hconcat'] + NUM_SPECIAL_TOKENS,
                    self.DSL.prim_indices['vconcat'] + NUM_SPECIAL_TOKENS,
                    self.NEW_LEVEL,
                    self.DSL.prim_indices['vconcat'] + NUM_SPECIAL_TOKENS
                ]

        else:
            print("==> ERROR in generate_task_program_quad: shape = ", shape)
            exit(-1)

        a = np.random.uniform()
        if a < 0.2:
            post_processing = ['invert_colors', 'set_fg_color1', 'set_fg_color2', 'set_fg_color3', 'set_fg_color4', 'set_fg_color5', 'set_fg_color6', 'set_fg_color7', 'set_fg_color8', 'set_fg_color9']
            random_pp = np.random.choice(post_processing)

            pp_lam = self.DSL.semantics[random_pp]
            pp_task_desc = "%s(%s)" % (random_pp, task_desc)
            
            pp_idx = self.DSL.prim_indices[random_pp]
            label_seq.append(1)
            label_seq.append(pp_idx + NUM_SPECIAL_TOKENS)

            output_func = lambda g: pp_lam(task_func(g))            
            return output_func, pp_task_desc, label_seq, transforms
        else:
            return task_func, task_desc, label_seq, transforms

    def tiling_mismatch(self, shape, transforms, starting_grid):
        def shape_of(transform, grid):
            #w, h = len(grid[0]), len(grid)
            w, h = grid.width, grid.height

            if transform in ['identity', 'rot180',  'hmirror', 'vmirror']:
                # The shape-preserving transformations:
                return w, h
            else:
                # 'rot90', 'rot270'
                return h, w

        if shape[0] == 1:
            # Horizontal tiling: all heights must match
            output_heights = []
            for t in transforms:
                out_w, out_h = shape_of(t, starting_grid)
                output_heights.append(out_h)

            if len(set(output_heights)) == 1:
                return False
            return True

        elif shape[1] == 1:
            # Vertical tiling: all widths must match
            output_widths = []
            for t in transforms:
                out_w, out_h = shape_of(t, starting_grid)
                output_widths.append(out_w)

            if len(set(output_widths)) == 1:
                return False
            return True

        else:
            # Quad tiling: all dimensions must match
            output_heights = []
            output_widths = []
            for t in transforms:
                out_w, out_h = shape_of(t, starting_grid)
                output_widths.append(out_w)
                output_heights.append(out_h)

            if len(set(output_heights)) == 1 and len(set(output_widths)) == 1:
                return False
            return True

    def generate(self, k=6):

        def generate_shape():
            choices = [(1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1), (2, 2), (3, 3)] # TODO: 4x4 and 5-way vertical/horizontal
            choice_idx = np.random.choice(np.arange(len(choices)))
            return choices[choice_idx]

        task_success = False
        while not task_success:
            #print("==> Generating task...")
            shape = generate_shape()

            # optionally select a sub-grid primitive to start from
            crop_prim = 'identity'

            if shape[0] == 1:
                task_func, task_desc, label_seq, transforms = self.generate_task_program(shape, dir='hconcat')
            elif shape[1] == 1:
                task_func, task_desc, label_seq, transforms = self.generate_task_program(shape, dir='vconcat')
            else:
                task_func, task_desc, label_seq, transforms = self.generate_task_program_quad(shape)

            task_valid = True
            valid_grid_set = False
            failed_count = 0
            while not valid_grid_set and task_valid:
                example_grid_set = []

                failed_count += 1
                if failed_count > 10:
                    task_valid = False
                    task_success = False
                    break

                valid_grid_set = True
                for example_idx in range(k):
                    starting_grid = self.generate_starting_grid(shape, crop_prim)

                    # given the tiling shape, and the individual transforms, confirm that they 
                    # result in grids that all match along the relevant dimension for that shape
                    if self.tiling_mismatch(shape, transforms, starting_grid):
                        valid_grid_set = False
                        break

                    try:
                        output_grid = task_func(starting_grid)
                    except:
                        try:
                            starting_grid = self.generate_starting_grid(shape, crop_prim, square=True)
                            output_grid = task_func(starting_grid)
                        except:
                            valid_grid_set = False
                            break

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
                        try:
                            starting_grid = self.generate_starting_grid(shape, crop_prim, square=True)
                            output_grid = task_func(starting_grid)
                        except:
                            valid_grid_set = False
                            break

                        if not val_utils.is_valid_shape([output_grid.cells], 30):
                            valid_grid_set = False
                            break

                    example_grid_set.append((starting_grid.cells, output_grid.cells))

                if valid_grid_set and task_valid:
                    #print("==> Task valid!")
                    task_success = True

        return example_grid_set, task_desc, label_seq