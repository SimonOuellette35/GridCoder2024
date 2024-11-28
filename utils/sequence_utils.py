import numpy as np

MAX_LENGTH = 16832  # For the reformer, must pad to the next multiple of 64.
MAX_LENGTH_SMALL = 1152

# 20x20 grids, k = 3
#MAX_LENGTH_TINY = 2560  # 704 was for 10x10 grids, k = 3

MAX_LENGTH_TINY = 5580

def gen_in_context_seq(xs, ys, num_args):
    #  In-context sequence format:
    #  [START_OF_EXAMPLE, SOGX, pixels/row-end tokens, EOG, if another argument: pixels/row-end tokens, EOG, SOGY, pixels/row-end tokens, EOG, END_OF_EXAMPLE,
    #   START_OF_EXAMPLE, SOGX, pixels/row-end tokens, EOG, if another argument: pixels/row-end tokens, EOG, SOGY, pixels/row-end tokens, EOG, END_OF_EXAMPLE,
    #   ...]
    #  With special tokens:
    #  - START_OF_EXAMPLE: designates the start of an (X, Y) example for 1-arg primitives, or (X1, X2, Y) example for 2-arg primitives
    #  - END_OF_EXAMPLE: designates the end of that example
    #  - SOGX: start of input grid set
    #  - EOG: end of grid
    #  - SOGY: start of target grid for this example
    #  - ROW_END: end of grid row
    #  - PADDING: pad the input sequence up to
    #      max size of an example: 2797
    #      up to 6 examples =  16782 tokens
    #  - +10 regular color tokens
    #  Input vocabulary is then: 19
    PADDING = 0
    ROW_END = 1
    EOG = 2
    START_OF_EXAMPLE = 13
    END_OF_EXAMPLE = 14
    SOGX = 15
    SOGY = 16

    # COLORS = 3 to 12 inclusively

    def get_unpadded(tmp_grid):
        result = np.where(tmp_grid == PADDING)[0]
        if result.size > 0:
            end_idx = result[0]
            return tmp_grid[:end_idx]
        else:
            return tmp_grid

    k = len(xs[0])
    input_seq = []
    if num_args == 1:
        for k_idx in range(k):
            input_seq.append(START_OF_EXAMPLE)
            input_seq.append(SOGX)
            tmp = xs[0][k_idx]
            tmp = get_unpadded(tmp)
            input_seq.extend(tmp)
            input_seq.append(SOGY)
            tmp = ys[k_idx]
            tmp = get_unpadded(tmp)
            input_seq.extend(tmp)
            input_seq.append(END_OF_EXAMPLE)
    else:
        for k_idx in range(k):
            input_seq.append(START_OF_EXAMPLE)
            input_seq.append(SOGX)
            tmp = xs[0][k_idx]
            tmp = get_unpadded(tmp)
            input_seq.extend(tmp)
            tmp = xs[1][k_idx]
            tmp = get_unpadded(tmp)
            input_seq.extend(tmp)
            input_seq.append(SOGY)
            tmp = ys[k_idx]
            tmp = get_unpadded(tmp)
            input_seq.extend(tmp)
            input_seq.append(END_OF_EXAMPLE)

    for _ in range(len(input_seq), MAX_LENGTH):
        input_seq.append(PADDING)

    return np.array(input_seq).astype(int)

def gen_in_context_seq_small(xs, ys, num_args):
    #  In-context sequence format:
    #  [START_OF_EXAMPLE, SOGX, pixels/row-end tokens, EOG, SOGY, pixels/row-end tokens, EOG, END_OF_EXAMPLE,
    #   START_OF_EXAMPLE, SOGX, pixels/row-end tokens, EOG, SOGY, pixels/row-end tokens, EOG, END_OF_EXAMPLE,
    #   ...]
    #  With special tokens:
    #  - START_OF_EXAMPLE: designates the start of an (X, Y) example for 1-arg primitives, or (X1, X2, Y) example for 2-arg primitives
    #  - END_OF_EXAMPLE: designates the end of that example
    #  - SOGX: start of input grid set
    #  - EOG: end of grid
    #  - SOGY: start of target grid for this example
    #  - ROW_END: end of grid row
    #  - PADDING: pad the input sequence up to
    #      max size of an example:
    #      5 examples = 1152
    #  - +10 regular color tokens
    #  Input vocabulary is then: 19
    PADDING = 0
    ROW_END = 1
    EOG = 2
    START_OF_EXAMPLE = 13
    END_OF_EXAMPLE = 14
    SOGX = 15
    SOGY = 16

    # COLORS = 3 to 12 inclusively

    def get_unpadded(tmp_grid):
        result = np.where(tmp_grid == PADDING)[0]
        if result.size > 0:
            end_idx = result[0]
            return tmp_grid[:end_idx]
        else:
            return tmp_grid

    input_seq = []
    for k_idx in range(len(xs[0])):
        input_seq.append(START_OF_EXAMPLE)
        input_seq.append(SOGX)
        tmp = xs[0][k_idx]
        tmp = get_unpadded(tmp)
        input_seq.extend(tmp)
        input_seq.append(SOGY)
        tmp = ys[k_idx]
        tmp = get_unpadded(tmp)
        input_seq.extend(tmp)
        input_seq.append(END_OF_EXAMPLE)

    for _ in range(len(input_seq), MAX_LENGTH_SMALL):
        input_seq.append(PADDING)

    return np.array(input_seq).astype(int)

def gen_in_context_seq_tiny(xs, ys, num_args):
    #  In-context sequence format:
    #  [START_OF_EXAMPLE, SOGX, pixels/row-end tokens, EOG, SOGY, pixels/row-end tokens, EOG, END_OF_EXAMPLE,
    #   START_OF_EXAMPLE, SOGX, pixels/row-end tokens, EOG, SOGY, pixels/row-end tokens, EOG, END_OF_EXAMPLE,
    #   ...]
    #  With special tokens:
    #  - START_OF_EXAMPLE: designates the start of an (X, Y) example for 1-arg primitives, or (X1, X2, Y) example for 2-arg primitives
    #  - END_OF_EXAMPLE: designates the end of that example
    #  - SOGX: start of input grid set
    #  - EOG: end of grid
    #  - SOGY: start of target grid for this example
    #  - ROW_END: end of grid row
    #  - PADDING: pad the input sequence up to
    #      max size of an example:
    #      5 examples = 1152
    #  - +10 regular color tokens
    #  Input vocabulary is then: 19
    PADDING = 0
    ROW_END = 1
    EOG = 2
    START_OF_EXAMPLE = 13
    END_OF_EXAMPLE = 14
    SOGX = 15
    SOGY = 16

    # COLORS = 3 to 12 inclusively

    def get_unpadded(tmp_grid):
        result = np.where(tmp_grid == PADDING)[0]
        if result.size > 0:
            end_idx = result[0]
            return tmp_grid[:end_idx]
        else:
            return tmp_grid

    input_seq = []
    for k_idx in range(len(xs[0])):
        input_seq.append(START_OF_EXAMPLE)
        input_seq.append(SOGX)
        tmp = xs[0][k_idx]
        tmp = get_unpadded(tmp)
        input_seq.extend(tmp)
        input_seq.append(SOGY)
        tmp = ys[k_idx]
        tmp = get_unpadded(tmp)
        input_seq.extend(tmp)
        input_seq.append(END_OF_EXAMPLE)

    for _ in range(len(input_seq), MAX_LENGTH_TINY):
        input_seq.append(PADDING)

    return np.array(input_seq).astype(int)

def gen_in_context_seq_tinyV2(xs, ys, num_args):
    #  In-context sequence format:
    #  example 1: [pixels/row-end tokens padded to 20x20, pixels/row-end tokens,
    #  example 2:  pixels/row-end tokens padded to 20x20, pixels/row-end tokens,
    #   ...]
    #  With special tokens:
    #  - ROW_END: end of grid row
    #  - PADDING: pad the input sequence up to
    #      max size of an example:
    #      5 examples = 1152
    #  - +10 regular color tokens
    #  Input vocabulary is then: 13
    PADDING = 0
    ROW_END = 1
    EOG = 2

    # COLORS = 3 to 12 inclusively
    GRID_LENGTH = 21 * 20               # 420
    input_seq = []
    for k_idx in range(len(xs[0])):
        tmp = xs[0][k_idx]
        input_seq.extend(tmp[:GRID_LENGTH])
        tmp = ys[k_idx]
        input_seq.extend(tmp[:GRID_LENGTH])

    for _ in range(len(input_seq), MAX_LENGTH_TINY):
        input_seq.append(PADDING)

    return np.array(input_seq).astype(int)

def gen_in_context_seq_tiny_custom(xs, ys):

    GRID_LENGTH = 31 * 30               # 420
    input_seq_batch = []

    for k_idx in range(len(xs[0])):
        tmpx = xs[0][k_idx][:GRID_LENGTH]
        tmpy = ys[k_idx][:GRID_LENGTH]

        example = np.concatenate((tmpx, tmpy))      # axis/dim?
        input_seq_batch.append(example)

    return np.array(input_seq_batch).astype(int)

def gen_in_context_seq_tiny_customV2(xs, ys):

    GRID_LENGTH = (31 * 30) + 1              # 931
    input_seq_batch = []

    for k_idx in range(len(xs)):
        tmpx1 = xs[k_idx][:GRID_LENGTH]
        tmpx2 = xs[k_idx][GRID_LENGTH:]
        tmpy = ys[k_idx][:GRID_LENGTH]

        example = np.concatenate((tmpx1, tmpx2, tmpy))  # axis/dim?
        input_seq_batch.append(example)

    return np.array(input_seq_batch).astype(int)

def gen_in_context_seq_full(xs, ys):

    GRID_LENGTH = (31 * 30) + 1              # 931
    input_seq_batch = []

    for k_idx in range(len(xs)):
        tmpx = xs[k_idx][:GRID_LENGTH]
        tmpy = ys[k_idx][:GRID_LENGTH]

        example = np.concatenate((tmpx, tmpy))
        input_seq_batch.append(example)

    return np.array(input_seq_batch).astype(int)
