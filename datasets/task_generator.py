import numpy as np
import random

class TaskGenerator():

    def __init__(self, DSL, validation=False):
        self.validation = validation
        self.DSL = DSL


    def generate(self, task_name, color1=None, color2=None, ncols=None, nrows=None):
        if task_name == 'cellwiseOR':
            return self.cellwiseOR(color1=color1, color2=color2, ncols=ncols, nrows=nrows)
        elif task_name == 'cellwiseNOR':
            return self.cellwiseNOR()
        elif task_name == 'cellwiseAND':
            return self.cellwiseAND()
        elif task_name == 'cellwiseXOR':
            return self.cellwiseXOR()
        elif task_name == 'cellwiseDifference':
            return self.cellwiseDifference()
        elif task_name == 'vconcat':
            return self.vconcat()
        elif task_name == 'hconcat':
            return self.hconcat()
        elif 'color_change' in task_name:
            return self.color_change(color1)
        else:
            if task_name.startswith('stack_rows_horizontally'):
                nrows = np.random.choice(np.arange(3, 6))
                ncols = np.random.choice(np.arange(3, 6))
                return self.cellwiseOR(nrows=nrows, ncols=ncols)
            elif task_name.startswith('stack_rows_vertically') or task_name.startswith('stack_columns_vertically'):
                nrows = np.random.choice(np.arange(3, 6))
                ncols = np.random.choice(np.arange(3, 6))
                return self.cellwiseOR(nrows=nrows, ncols=ncols)
            elif task_name == 'upscale_by_three':
                nrows = np.random.choice(np.arange(3, 11))
                ncols = np.random.choice(np.arange(3, 11))
                return self.cellwiseOR(nrows=nrows, ncols=ncols)
            elif task_name == 'upscale_horizontal_by_three':
                nrows = np.random.choice(np.arange(3, 31))
                ncols = np.random.choice(np.arange(3, 11))
                return self.cellwiseOR(nrows=nrows, ncols=ncols)
            elif task_name == 'upscale_vertical_by_three':
                nrows = np.random.choice(np.arange(3, 11))
                ncols = np.random.choice(np.arange(3, 31))
                return self.cellwiseOR(nrows=nrows, ncols=ncols)
            elif task_name == 'upscale_by_two':
                nrows = np.random.choice(np.arange(3, 16))
                ncols = np.random.choice(np.arange(3, 16))
                return self.cellwiseOR(nrows=nrows, ncols=ncols)
            else:
                return self.cellwiseOR()

    def fill_grid(self, tmp_grid, tmp_palette):
        threshold = np.random.uniform(0.1, 1.)

        for i in range(tmp_grid.shape[0]):
            for j in range(tmp_grid.shape[1]):
                a = np.random.uniform()

                if a < threshold:
                    color = np.random.choice(tmp_palette)
                    tmp_grid[i, j] = color

        return tuple(tuple(inner) for inner in tmp_grid.astype(int))

    @staticmethod
    def get_dimensions(upper_size_limit):
        if upper_size_limit is None:
            nrows = np.random.choice(np.arange(3, 31))
            ncols = np.random.choice(np.arange(3, 31))
        else:
            nrows = np.random.choice(np.arange(3, int(upper_size_limit/2)+1))
            ncols = np.random.choice(np.arange(3, int(upper_size_limit/2)+1))

        return nrows, ncols

    def color_change(self, color1):

        nrows, ncols = TaskGenerator.get_dimensions(30)

        valid_grid = False
        while not valid_grid:
            grid1 = np.zeros((nrows, ncols))
            grid1_ncolors = np.random.choice(np.arange(2, 5))
            grid1_palette = np.random.choice(np.arange(1, 10), grid1_ncolors, replace=False)
            grid1_palette = np.append(grid1_palette, color1)
            grid1 = self.fill_grid(grid1, grid1_palette)

            contains_c1 = any(color1 in sub_tuple for sub_tuple in grid1)
            if contains_c1:
                valid_grid = True

        return grid1

    def color_swap(self, color1, color2):

        nrows, ncols = TaskGenerator.get_dimensions(30)
        valid_grid = False
        while not valid_grid:
            grid1 = np.zeros((nrows, ncols))
            grid1_ncolors = np.random.choice(np.arange(2, 5))
            grid1_palette = np.random.choice(np.arange(1, 10), grid1_ncolors, replace=False)
            grid1_palette = np.append(grid1_palette, color1)
            grid1_palette = np.append(grid1_palette, color2)
            grid1 = self.fill_grid(grid1, grid1_palette)

            contains_c1 = any(color1 in sub_tuple for sub_tuple in grid1)
            if contains_c1:
                contains_c2 = any(color2 in sub_tuple for sub_tuple in grid1)
                if contains_c2:
                    valid_grid = True

        return grid1

    def vconcat(self, upper_size_limit=30):

        nrows1 = np.random.choice(np.arange(3, min(15, upper_size_limit/2)+1))
        nrows2 = np.random.choice(np.arange(3, min(15, upper_size_limit/2)+1))
        ncols = np.random.choice(np.arange(3, (upper_size_limit/2)+1))

        grid1 = np.zeros((nrows1, ncols))
        grid2 = np.zeros((nrows2, ncols))

        grid1_ncolors = np.random.choice(np.arange(1, 5))
        grid2_ncolors = np.random.choice(np.arange(1, 5))

        grid1_palette = np.random.choice(np.arange(1, 10), grid1_ncolors, replace=False)
        grid2_palette = np.random.choice(np.arange(1, 10), grid2_ncolors, replace=False)

        grid1 = self.fill_grid(grid1, grid1_palette)
        grid2 = self.fill_grid(grid2, grid2_palette)

        return grid1, grid2

    def hconcat(self, upper_size_limit=30):
        nrows = np.random.choice(np.arange(3, (upper_size_limit/2)+1))
        ncols1 = np.random.choice(np.arange(3, min(15, upper_size_limit/2)+1))
        ncols2 = np.random.choice(np.arange(3, min(15, upper_size_limit/2)+1))

        grid1 = np.zeros((nrows, ncols1))
        grid2 = np.zeros((nrows, ncols2))

        grid1_ncolors = np.random.choice(np.arange(1, 5))
        grid2_ncolors = np.random.choice(np.arange(1, 5))

        grid1_palette = np.random.choice(np.arange(1, 10), grid1_ncolors, replace=False)
        grid2_palette = np.random.choice(np.arange(1, 10), grid2_ncolors, replace=False)

        grid1 = self.fill_grid(grid1, grid1_palette)
        grid2 = self.fill_grid(grid2, grid2_palette)

        return grid1, grid2


    def cellwiseOR(self, nrows=None, ncols=None, color1=None, color2=None):

        if nrows is None:
            nrows, _ = TaskGenerator.get_dimensions(30)

        if ncols is None:
            _, ncols = TaskGenerator.get_dimensions(30)

        grid1 = np.zeros((nrows, ncols))
        grid2 = np.zeros((nrows, ncols))

        grid1_ncolors = np.random.choice(np.arange(1, 5))
        grid2_ncolors = np.random.choice(np.arange(1, 5))

        grid1_palette = np.random.choice(np.arange(1, 10), grid1_ncolors, replace=False)
        grid2_palette = np.random.choice(np.arange(1, 10), grid2_ncolors, replace=False)

        if color1 is not None:
            grid1_palette.append(color1)
            grid2_palette.append(color1)

        if color2 is not None:
            grid1_palette.append(color2)
            grid2_palette.append(color2)

        grid1 = self.fill_grid(grid1, grid1_palette)
        grid2 = self.fill_grid(grid2, grid2_palette)

        return grid1, grid2

    def cellwiseNOR(self, upper_size_limit=None):

        nrows, ncols = TaskGenerator.get_dimensions(upper_size_limit)
        grid1 = np.zeros((nrows, ncols))
        grid2 = np.zeros((nrows, ncols))

        grid1_ncolors = np.random.choice(np.arange(1, 5))
        grid2_ncolors = np.random.choice(np.arange(1, 5))

        grid1_palette = np.random.choice(np.arange(1, 10), grid1_ncolors, replace=False)
        grid2_palette = np.random.choice(np.arange(1, 10), grid2_ncolors, replace=False)

        grid1 = self.fill_grid(grid1, grid1_palette, freq=0.5)
        grid2 = self.fill_grid(grid2, grid2_palette, freq=0.5)

        return grid1, grid2

    def cellwiseAND(self, upper_size_limit=None):

        nrows, ncols = TaskGenerator.get_dimensions(upper_size_limit)
        grid1 = np.zeros((nrows, ncols))
        grid2 = np.zeros((nrows, ncols))

        grid1_ncolors = np.random.choice(np.arange(1, 5))
        grid2_ncolors = np.random.choice(np.arange(1, 5))

        grid1_palette = np.random.choice(np.arange(1, 10), grid1_ncolors, replace=False)
        grid2_palette = np.random.choice(np.arange(1, 10), grid2_ncolors, replace=False)

        grid1 = self.fill_grid(grid1, grid1_palette)
        grid2 = self.fill_grid(grid2, grid2_palette)

        return grid1, grid2

    def cellwiseXOR(self, upper_size_limit=None):

        nrows, ncols = TaskGenerator.get_dimensions(upper_size_limit)
        grid1 = np.zeros((nrows, ncols))
        grid2 = np.zeros((nrows, ncols))

        grid1_ncolors = np.random.choice(np.arange(1, 5))
        grid2_ncolors = np.random.choice(np.arange(1, 5))

        grid1_palette = np.random.choice(np.arange(1, 10), grid1_ncolors, replace=False)
        grid2_palette = np.random.choice(np.arange(1, 10), grid2_ncolors, replace=False)

        grid1 = self.fill_grid(grid1, grid1_palette)
        grid2 = self.fill_grid(grid2, grid2_palette)

        return grid1, grid2

    def cellwiseDifference(self, upper_size_limit=None):

        nrows, ncols = TaskGenerator.get_dimensions(upper_size_limit)
        grid1 = np.zeros((nrows, ncols))
        grid2 = np.zeros((nrows, ncols))

        grid1_ncolors = np.random.choice(np.arange(1, 5))
        grid2_ncolors = np.random.choice(np.arange(1, 5))

        grid1_palette = np.random.choice(np.arange(1, 10), grid1_ncolors, replace=False)
        grid2_palette = np.random.choice(np.arange(1, 10), grid2_ncolors, replace=False)

        grid1 = self.fill_grid(grid1, grid1_palette)
        grid2 = self.fill_grid(grid2, grid2_palette)

        return grid1, grid2
