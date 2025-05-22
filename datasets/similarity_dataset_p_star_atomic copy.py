import Hodel_primitives_atomicV3 as Hodel
import Hodel_primitives_full_trainingV2 as Hodel_training
import ARC_gym.utils.tokenization as tok
from torch.utils.data import Dataset
import numpy as np
import os, json
import random
import utils.validation_utils as val_utils
import utils.grid_utils as grid_utils
from datasets.task_generator import TaskGenerator
from datasets.generators.merge_split_generator_atomic import MergeSplitGenerator
from datasets.generators.tiling_generator_atomic import TilingGenerator
from datasets.generators.trivial_objectness_generator import TrivialObjectnessGenerator
from datasets.generators.object_selector_generator import ObjectSelectorGenerator
from datasets.generators.windowing_generator import WindowingGenerator
from datasets.generators.object_recombiner_generator import ObjectRecombinerGenerator

# Define a color map
COLOR_MAP = {
    0: 'black',
    1: 'steelblue',
    2: 'green',
    3: 'yellow',
    4: 'purple',
    5: 'orange',
    6: 'red',
    7: 'salmon',
    8: 'aquamarine',
    9: 'white'
}

class ARCInspiredHodelSimilarity(Dataset):
    def __init__(self, validation=False, base_dir="ARC/data/training"):
        self.grid_size = 30
        self.num_special_tokens = 3
        self.new_level = 0
        self.identity = 1
        self.eos_token = 2
        
        self.task_ratios = np.zeros(7)

        self.training_tasks = []
        for prim_name, prim_func in Hodel_training.semantics.items():
            self.training_tasks.append((prim_func, prim_name))

        print("==> %i training tasks loaded!" % len(self.training_tasks))

        self.validation = validation
        self.task_generator = TaskGenerator(self.validation)

        self.merge_split_generator = MergeSplitGenerator(Hodel, self.validation)
        self.tiling_generator = TilingGenerator(Hodel, self.validation)
        self.trivial_objectness_generator = TrivialObjectnessGenerator(Hodel, self.validation)
        self.object_selector_generator = ObjectSelectorGenerator(Hodel, self.validation)
        self.windowing_generator = WindowingGenerator(Hodel, self.validation)
        self.object_recombiner_generator = ObjectRecombinerGenerator(Hodel, self.validation)

        self.base_dir = base_dir
        self.arc_files = os.listdir(base_dir)
        self.all_grids = []

        self.load_grids()

    def arc_to_numpy(self, fpath):
        with open(fpath) as f:
            content = json.load(f)

        grids = []
        for g in content["train"]:
            grids.append(np.array(g["input"], dtype="int8"))
            grids.append(np.array(g["output"], dtype="int8"))
        for g in content["test"]:
            grids.append(np.array(g["input"], dtype="int8"))
        return grids

    def load_grids(self):
        for fname in self.arc_files:
            fpath = os.path.join(self.base_dir, fname)
            self.all_grids.extend(self.arc_to_numpy(fpath))

    def augment(self, grid):
        num_rotations = np.random.choice(np.arange(4))
        for _ in range(num_rotations):
            grid = Hodel.rot90(grid)

        return grid

    def generateGrid(self, width, height):

        X = np.zeros((width, height))
        num_px = np.random.choice(np.arange(10, 20))

        pixel_list = []
        for x in range(width):
            for y in range(height):
                pixel_list.append((x, y))

        pixel_list = np.array(pixel_list)
        pixel_indices = np.random.choice(len(pixel_list), num_px, replace=False)
        x_list = pixel_list[pixel_indices, 0]
        y_list = pixel_list[pixel_indices, 1]
        color_list = np.random.choice(np.arange(10), num_px)

        for i in range(num_px):
            X[x_list[i], y_list[i]] = color_list[i]

        return X

    def sampleGridPatchWithColors(self, c1, c2=None):

        grid_samples = grid_utils.get_samples(Hodel, self.validation)
        random.shuffle(grid_samples)

        selected_idx = None
        for idx in range(len(grid_samples)):
            # Check if the specific value exists in the tuple of tuples
            contains_c1 = any(c1 in sub_tuple for sub_tuple in grid_samples[idx].cells)

            if contains_c1:
                if c2 is not None:
                    # Check if the specific value exists in the tuple of tuples
                    contains_c2 = any(c2 in sub_tuple for sub_tuple in grid_samples[idx].cells)
                    if contains_c2:
                        selected_idx = idx
                        break
                else:
                    selected_idx = idx
                    break

        if selected_idx is None:
            return None
        else:
            return self.augment(grid_samples[selected_idx])

    def sampleGridPatch(self, prim_name=''):

        grid_samples = grid_utils.get_samples(Hodel, self.validation)

        valid_grid = False
        while not valid_grid:
            idx = np.random.choice(np.arange(len(grid_samples)))
            tmp = self.augment(grid_samples[idx])

            if prim_name.startswith('stack_rows_horizontally') or prim_name.startswith('stack_rows_vertically'):
                if len(tmp.cells) * len(tmp.cells[0]) <= 30:
                    valid_grid = True
            else:
                valid_grid = True

        return tmp

    @staticmethod
    def all_valid_colors(a_list):
        for a in a_list:
            if not Hodel.palette(a).issubset(set(range(10))):
                return False

        return True

    # we only generate 1-arg transformations
    def select_next_primitive(self):
        tf = random.choice(self.training_tasks)
        return tf

    # inputs is:
    # [num_args, k examples, grid rows, grid cols]
    def apply_primitive(self, prim_name, prim_func, inputs, inp_paths, color1=None, color2=None):
        current_outputs = []

        for inp in inputs[0]:
            if 'color_change' in prim_name:
                go = prim_func(inp.cells)(color1)(color2)
            else:
                go = prim_func(inp.cells)

            current_outputs.append(Hodel.Grid(go))

        output_path = "%s(%s)" % (prim_name, inp_paths[0])

        return current_outputs, output_path

    def apply_selected_primitive(self,
                                 prim_name,
                                 prim_func,
                                 intermediate_grids,
                                 color1=None,
                                 color2=None):

        inputs = [intermediate_grids[0][0]]
        inp_paths = [intermediate_grids[0][1]]

        result = self.apply_primitive(prim_name, prim_func, inputs, inp_paths, color1, color2)

        current_outputs, output_path = result

        return current_outputs, output_path

    def generate_random_task(self):

        task_success = False
        while not task_success:
            tf = self.select_next_primitive()
            prim_func = tf[0]
            prim_name = tf[1]

            color1 = color2 = 0
            if 'color_change' in prim_name:
                # the two colors must NOT be the same
                color1 = np.random.choice(np.arange(10))
                colors = np.arange(10)
                colors = np.delete(colors, color1)
                color2 = np.random.choice(colors)

            task_valid = False
            failed_attempts = 0
            while not task_valid:
                failed_attempts += 1
                if failed_attempts > 20:
                    task_success = False
                    break

                task_valid = True
                task_success = True
                initial_grid_set, k = self.init_task_generation(prim_name, color1, color2)
                intermediate_grids = [(initial_grid_set, 'var0')]

                output_grids = []
                out_grid, out_path = self.apply_selected_primitive( prim_name,
                                                                    prim_func,
                                                                    intermediate_grids,
                                                                    color1,
                                                                    color2)

                # make sure we haven't generated a grid that is exactly identical to one that has been
                # generated before
                if val_utils.is_identity(intermediate_grids, out_grid):
                    #print("\tis_identity!")
                    task_valid = False
                    continue

                # don't support empty outputs
                if val_utils.is_empty([out.cells for out in out_grid], Hodel):
                    #print("\tPrimitive %s: is empty!" % prim_name)
                    task_valid = False
                    continue

                # make sure the grid shape is valid
                if not val_utils.is_valid_shape([out.cells for out in out_grid], self.grid_size):
                    #print("\tPrimitive %s: invalid shape!" % prim_name)
                    task_valid = False
                    continue

                out_cells_list = []
                for out in out_grid:
                    if out.ul_x != 0 or out.ul_y != 0:
                        out_cells = out.get_shifted_cells()
                    else:
                        out_cells = out.cells

                    out_cells_list.append(out_cells)
                
                output_grids.append((out_cells_list, out_path))

        A = self.get_program(prim_name)
        X = []
        for example_idx in range(k):
            x = tok.tokenize_grid(initial_grid_set[example_idx].cells, max_length=931)
            X.append(x)

        out_y = []
        grid_set = output_grids[0][0]
        for grid in grid_set:
            y = tok.tokenize_grid(grid, max_length=931)
            out_y.append(y)

        desc = output_grids[0][1]

        return X, out_y, A, desc

    def get_program(self, prim_name):

        def get_prim_index(name):
            if name == 'identity':
                return self.identity
            else:
                prim_idx = Hodel.get_index(name.strip())
                return prim_idx + self.num_special_tokens

        def build_2arg_program(sub_primitives, concat_name):
            # Find the index of occurrence of concat_name in sub_primitives
            concat_index = sub_primitives.index(concat_name)
            
            # Create the program list
            program = []
            
            if concat_index == 0:
                program.append(self.identity)
                program.append(self.identity)
                program.append(self.new_level)
                program.append(Hodel.get_index(concat_name) + self.num_special_tokens)
                program.append(self.new_level)
                program.append(Hodel.get_index(sub_primitives[1]) + self.num_special_tokens)
                if len(sub_primitives == 3):
                    program.append(self.new_level)
                    program.append(Hodel.get_index(sub_primitives[2]) + self.num_special_tokens)
                    #program.append(self.eos_token)
                # else:
                #     program.append(self.eos_token)

            elif concat_index == 1:
                program.append(get_prim_index(sub_primitives[0]))
                if len(sub_primitives) == 3:
                    program.append(get_prim_index(sub_primitives[2]))
                else:
                    program.append(self.identity)

                program.append(self.new_level)
                program.append(Hodel.get_index(concat_name) + self.num_special_tokens)
                #program.append(self.eos_token)

            elif concat_index == 2:
                print("Error: there shouldn't be a concat primitive at the 3rd position? (%s)" % prim_name)
                exit(-1)

            return program

        # Tokenize the primitive name into sub-primitives
        sub_primitives = prim_name.split('+')
        
        # Create a list to store the tokenized program
        program = []

        if 'hconcat' in sub_primitives:
            return build_2arg_program(sub_primitives, 'hconcat')
        elif 'vconcat' in sub_primitives:
            return build_2arg_program(sub_primitives, 'vconcat')
        
        # Add each sub-primitive to the program list
        for idx, sub_prim in enumerate(sub_primitives):
            program.append(Hodel.get_index(sub_prim) + self.num_special_tokens)
            
            if idx < len(sub_primitives) - 1:
                program.append(self.new_level)
            # else:
            #     program.append(self.eos_token)
        
        return program

    def init_task_generation(self, prim_name, color1=0, color2=0):
        k = np.random.choice(np.arange(3, 6))

        initial_grid_set = []

        for _ in range(k):
            gi = None

            if Hodel.is_color_primitive(prim_name) and Hodel.is_diagonal_primitive(prim_name):
                # don't pick non-square grids for diagonal-based primitives
                a = np.random.uniform()
                valid_grid = False

                if a <= 0.5:
                    while not valid_grid:
                        # First, try to find a grid in the pre-defined ones that matches the request color(s)
                        gi = self.sampleGridPatchWithColors(color1)

                        if len(gi) == len(gi[0]):
                            valid_grid = True
                else:
                    dim = np.random.choice(np.arange(3, 31))
                    gi, _ = self.task_generator.generate('cellwiseOR', color1=color1, color2=color2, ncols=dim, nrows=dim)
                    gi = Hodel.Grid(gi)

            elif Hodel.is_color_primitive(prim_name):
                a = np.random.uniform()

                if a <= 0.5:
                    # First, try to find a grid in the pre-defined ones that matches the request color(s)
                    gi = self.sampleGridPatchWithColors(color1)

                if gi is None:
                    # generate a randomized grid with the specified color requirements
                    gi = self.task_generator.generate(prim_name, color1, color2)
                    gi = Hodel.Grid(gi)

            elif Hodel.is_diagonal_primitive(prim_name):
                # don't pick non-square grids for diagonal-based primitives
                a = np.random.uniform()
                valid_grid = False

                if a <= 0.5:
                    while not valid_grid:
                        gi = self.sampleGridPatch()
                        if len(gi) == len(gi[0]):
                            gi = Hodel.Grid(gi)
                            valid_grid = True
                else:
                    dim = np.random.choice(np.arange(3, 31))
                    gi, _ = self.task_generator.generate('cellwiseOR', nrows=dim, ncols=dim)
                    gi = Hodel.Grid(gi)
            else:
                a = np.random.uniform()

                if a <= 0.5:
                    gi = self.sampleGridPatch(prim_name)
                else:
                    gi, _ = self.task_generator.generate(prim_name)
                    gi = Hodel.Grid(gi)

            initial_grid_set.append(gi)

        return initial_grid_set, k

    def sample_transform(self):
        task_valid = False
        while not task_valid:
            task_valid = True
            try:
                return self.generate_random_task()
            except Exception as e:
                print("An error occurred:")
                import traceback
                traceback.print_exc()
                task_valid = False

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        S = {}

        DSL_version = '3'   # '2.5', '3'
        task_valid = False
        while not task_valid:
            try:
                a = np.random.uniform()

                if DSL_version == '2':
                    if a < 0.33:
                        x_batch, y_batch, label_seq, task_desc = self.sample_transform()         
                    else:
                        if a < 0.66:
                            example_grid_set, task_desc, label_seq = self.merge_split_generator.generate()
                        else:
                            example_grid_set, task_desc, label_seq = self.tiling_generator.generate()
                    
                        k = len(example_grid_set)

                        # parse example_grid_set into x_batch and y_batch
                        tmp_x_batch = [grid[0] for grid in example_grid_set]
                        tmp_y_batch = [grid[1] for grid in example_grid_set]

                        x_batch = []
                        y_batch = []
                        for example_idx in range(k):
                            grid = tmp_x_batch[example_idx]

                            if len(grid) > 30:
                                print("==> Grid height > 30")
                                exit(0)
                            if len(grid[0]) > 30:
                                print("==> Grid width > 30")
                                exit(0)

                            tmp_x = tok.tokenize_grid(tmp_x_batch[example_idx], max_length=931)
                            x_batch.append(tmp_x)
                            tmp_y = tok.tokenize_grid(tmp_y_batch[example_idx], max_length=931)
                            y_batch.append(tmp_y)

                elif DSL_version == '2.5':
                    if a < 0.1:
                        x_batch, y_batch, label_seq, task_desc = self.sample_transform()         
                        self.task_ratios[0] += 1
                    else:
                        if a < 0.2:
                            example_grid_set, task_desc, label_seq = self.merge_split_generator.generate()
                            self.task_ratios[1] += 1
                        elif a < 0.3:
                            example_grid_set, task_desc, label_seq = self.tiling_generator.generate()
                            self.task_ratios[2] += 1
                        elif a < 0.6:
                            example_grid_set, task_desc, label_seq = self.trivial_objectness_generator.generate(Hodel)
                            self.task_ratios[3] += 1
                        elif a < 0.8:
                            example_grid_set, task_desc, label_seq = self.windowing_generator.generate(Hodel)
                            self.task_ratios[5] += 1
                        else:
                            example_grid_set, task_desc, label_seq = self.object_recombiner_generator.generate(Hodel)
                            self.task_ratios[6] += 1

                        k = len(example_grid_set)
                        
                        # parse example_grid_set into x_batch and y_batch
                        tmp_x_batch = [grid[0] for grid in example_grid_set]
                        tmp_y_batch = [grid[1] for grid in example_grid_set]

                        x_batch = []
                        y_batch = []
                        for example_idx in range(k):
                            grid = tmp_x_batch[example_idx]

                            if len(grid) > 30:
                                print("==> Grid height > 30")
                                exit(0)
                            if len(grid[0]) > 30:
                                print("==> Grid width > 30")
                                exit(0)

                            tmp_x = tok.tokenize_grid(tmp_x_batch[example_idx], max_length=931)
                            x_batch.append(tmp_x)
                            tmp_y = tok.tokenize_grid(tmp_y_batch[example_idx], max_length=931)
                            y_batch.append(tmp_y)

                elif DSL_version == '3':
                    if a < 0.1:
                        x_batch, y_batch, label_seq, task_desc = self.sample_transform()         
                        self.task_ratios[0] += 1
                    else:
                        if a < 0.2:
                            example_grid_set, task_desc, label_seq = self.merge_split_generator.generate()
                            self.task_ratios[1] += 1
                        elif a < 0.3:
                            example_grid_set, task_desc, label_seq = self.tiling_generator.generate()
                            self.task_ratios[2] += 1
                        elif a < 0.5:
                            example_grid_set, task_desc, label_seq = self.trivial_objectness_generator.generate(Hodel)
                            self.task_ratios[3] += 1
                        elif a < 0.7:
                            example_grid_set, task_desc, label_seq = self.object_selector_generator.generate(Hodel)
                            self.task_ratios[4] += 1
                        elif a < 0.8:
                            example_grid_set, task_desc, label_seq = self.windowing_generator.generate(Hodel)
                            self.task_ratios[5] += 1
                        else:
                            example_grid_set, task_desc, label_seq = self.object_recombiner_generator.generate(Hodel)
                            self.task_ratios[6] += 1
                    
                        k = len(example_grid_set)

                        # parse example_grid_set into x_batch and y_batch
                        tmp_x_batch = [grid[0] for grid in example_grid_set]
                        tmp_y_batch = [grid[1] for grid in example_grid_set]

                        x_batch = []
                        y_batch = []
                        for example_idx in range(k):
                            grid = tmp_x_batch[example_idx]

                            if len(grid) > 30:
                                print("==> Grid height > 30")
                                exit(0)
                            if len(grid[0]) > 30:
                                print("==> Grid width > 30")
                                exit(0)

                            tmp_x = tok.tokenize_grid(tmp_x_batch[example_idx], max_length=931)
                            x_batch.append(tmp_x)
                            tmp_y = tok.tokenize_grid(tmp_y_batch[example_idx], max_length=931)
                            y_batch.append(tmp_y)

                task_valid = True
            except IndexError:
                pass
            
        S['xs'] = x_batch
        S['ys'] = y_batch
        S['label_seq'] = label_seq
        S['task_desc'] = task_desc

        return S


