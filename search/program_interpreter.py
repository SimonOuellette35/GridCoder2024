import Hodel_primitives_atomicV3 as hp
import numpy as np
import inspect

NEW_LEVEL = 1
IDENTITY_TOKEN = 2
EOS_TOKEN = 3
NUM_SPECIAL_TOKENS = 4

class TreeNode:
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __str__(self):
        return "%s(%s)" % (self.func, self.args)

    def __repr__(self):
        return self.__str__()


def get_prev_func(lvl_idx, i, used_outputs, levels):
    if lvl_idx == 0:
        return -1
    else:
        value = 0
        unused_admissible_funcs = []

        current_lvl = lvl_idx-1
        while current_lvl >= 0:
            for func in levels[current_lvl]:
                if func not in used_outputs:
                    unused_admissible_funcs.append(func)
            current_lvl -= 1

        if len(unused_admissible_funcs) > 0:
            value = unused_admissible_funcs[0]
            return value

        used_admissible_funcs = []
        current_lvl = lvl_idx-1
        while current_lvl >= 0:
            for func in levels[current_lvl]:
                if func in used_outputs:
                    used_admissible_funcs.append(func)
            current_lvl -= 1

        value = used_admissible_funcs[i]
        return value

def get_num_args(func):
    if func == IDENTITY_TOKEN:
        return 1
    
    prim_name = hp.inverse_lookup(func-NUM_SPECIAL_TOKENS)
    num_args = hp.get_num_args(prim_name)
    return num_args

def get_num_lambda_func_args(lambda_func):
    count = 0
    while callable(lambda_func):
        sig = inspect.signature(lambda_func)
        params = sig.parameters
        count += len(params)
        # Get the next nested function by invoking the lambda
        if count == 0:
            break
        try:
            lambda_func = lambda_func(*(hp.Grid(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))) for _ in params))
        except TypeError:
            break
    return count


def generate_syntax_trees(token_seq):
    num_funcs = np.count_nonzero(token_seq)
    
    # Get indices of non-zero elements (functions)
    func_indices = np.nonzero(token_seq)[0]

    # Get indices of NEW_LEVEL elements (level separators)
    level_separators = np.where(token_seq == NEW_LEVEL)[0]

    # Define levels
    levels = []
    start = -1  # Start from -1 to include the first function index
    for separator in level_separators:
        levels.append(func_indices[(func_indices > start) & (func_indices < separator)])
        start = separator
    levels.append(func_indices[func_indices > start])
    
    tree_nodes_per_level = []
    used_outputs = []

    for lvl_idx, lvl in enumerate(levels):
        level_nodes = []

        for func_idx, func in enumerate(lvl):
            if token_seq[func] == EOS_TOKEN:
                break

            node = TreeNode(func, [])

            num_args = get_num_args(token_seq[func])
            for i in range(num_args):
                prev_func = get_prev_func(lvl_idx, i, used_outputs, levels)
                used_outputs.append(prev_func)
                node.args.append(prev_func)

            level_nodes.append(node)

        tree_nodes_per_level.append(level_nodes)

    return tree_nodes_per_level

def is_valid_partial_program(label_seq):

    # Split label_seq into levels
    levels = []
    current_level = []
    last_level_complete = False
    for label in label_seq:
        if label == 1:  # NEW_LEVEL token
            if current_level:
                levels.append(current_level)
                current_level = []
            else:
                # Two "NEW LEVELS" in a row is not a legal program
                return False
        elif label == 3: # EOS token
            levels.append(current_level)
            current_level = []
            break
        else:
            current_level.append(label)
    
    if current_level:
        levels.append(current_level)
    else:
        last_level_complete = True

    if not levels:
        return False

    if len(levels) == 1:
        return True

    # Validate each level
    prev_outputs = None
    for i, level in enumerate(levels):
        outputs = len(level)
        inputs = sum(get_num_args(label) for label in level)

        # print("Level %i, outputs = %i, inputs = %i" % (i, outputs, inputs))
        # print("\tlevel content: ", level)
        # Validate inputs match previous level's outputs (except for first level)
        if i == len(levels) - 1 and not last_level_complete:
            # the last level that is not yet wrapped up: the logic is a bit different. If there are already too many nodes at that level,
            # it is invalid -- but if there are too new, it is valid because there is still room to add the missing nodes.
            if inputs > prev_outputs:
                return False
        elif i > 0 and inputs != prev_outputs:
            return False

        prev_outputs = outputs

    return True

def is_valid_program(label_seq):
    # Check if the sequence ends with EOS
    if label_seq[-1] != 3:
        return False
    
    # Split label_seq into levels
    levels = []
    current_level = []
    for label in label_seq:
        if label == 1:  # NEW_LEVEL token
            if current_level:
                levels.append(current_level)
                current_level = []
        elif label == 3: # EOS token
            levels.append(current_level)
            current_level = []
            break
        else:
            current_level.append(label)
    
    if current_level:
        levels.append(current_level)

    if not levels:
        return False

    # Validate each level
    prev_outputs = None
    for i, level in enumerate(levels):
        outputs = len(level)
        inputs = sum(get_num_args(label) for label in level)

        # Validate last level has only 1 output
        if i == len(levels) - 1 and outputs != 1:
            return False

        # Validate inputs match previous level's outputs (except for first level)
        if i > 0 and inputs != prev_outputs:
            return False

        prev_outputs = outputs

    return True

def convert_to_prim_index(label_seq, func_idx):
    return label_seq[func_idx]

def create_lambda(label_seq, node):
    if convert_to_prim_index(label_seq, node.func) == IDENTITY_TOKEN:  # Represents identity function
        return lambda x: x
    prim_name = hp.inverse_lookup(convert_to_prim_index(label_seq, node.func) - NUM_SPECIAL_TOKENS)
    return hp.semantics[prim_name]

def get_sub_tree(tree, sub_tree_root_idx):
    lower_level = tree[-2]
    for node in lower_level:
        if node.func == sub_tree_root_idx:
            top_level = [node]

            sub_tree = tree[:-2]
            sub_tree.append(top_level)
            return sub_tree

    # TODO: this is happening in Kaggle submission. Why?
    #print("ERROR: could not find referenced argument in lower level.")
    raise Exception

def apply_partial_program(token_seq, input_grid_set):
    tree = generate_syntax_trees(token_seq)

    intermediate_grid_sets = []
    
    for level_idx, level in enumerate(tree):
        level_outputs = []
        arg_idx = 0
        for node_idx, node in enumerate(level):
            prim_index = convert_to_prim_index(token_seq, node.func)
            
            if prim_index == 1:  # Identity function
                lambda_func = lambda x: x
            else:
                prim_name = hp.inverse_lookup(prim_index - NUM_SPECIAL_TOKENS)
                lambda_func = hp.semantics[prim_name]
            
            if level_idx == 0:
                input_grids = input_grid_set
            else:
                input_grids = intermediate_grid_sets[arg_idx]
                arg_idx += 1
                if len(node.args) > 1:
                    input_grids_2 = intermediate_grid_sets[arg_idx]
                    arg_idx += 1
            
            node_outputs = []
            for i, grid in enumerate(input_grids):
                if len(node.args) == 2:
                    output = lambda_func(grid)(input_grids_2[i])
                else:
                    output = lambda_func(grid)
                
                node_outputs.append(output)
            
            level_outputs.append(node_outputs)

        if level_idx < len(tree) - 1:
            intermediate_grid_sets = level_outputs
        else:
            # get the remaining non-consumed grids at the end of the partial execution...
            final_output_sets = []
            for grid in level_outputs:
                final_output_sets.append(grid)

            for grid_idx in range(len(final_output_sets), len(intermediate_grid_sets)):
                final_output_sets.append(intermediate_grid_sets[grid_idx])

    return final_output_sets


def assemble_program(tree, label_seq, args_composed=True):
    tree_root_node = tree[-1][0]

    lambda_func = create_lambda(label_seq, tree_root_node)
    is_color_prim = False
    if convert_to_prim_index(label_seq, tree_root_node.func) > IDENTITY_TOKEN:
        prim_name = hp.inverse_lookup(convert_to_prim_index(label_seq, tree_root_node.func) - NUM_SPECIAL_TOKENS)
        is_color_prim = prim_name.startswith('color_')
   
    if len(tree_root_node.args) == 2:
        if len(tree) <= 1:
            return lambda_func

        sub_tree_arg1 = get_sub_tree(tree, tree_root_node.args[0])
        sub_tree_arg2 = get_sub_tree(tree, tree_root_node.args[1])

        if is_color_prim:
            return lambda x: lambda c1: lambda c2: lambda_func(assemble_program(sub_tree_arg1, label_seq)(x))(assemble_program(sub_tree_arg2, label_seq)(x))(c1)(c2)
        else:
            return lambda x: lambda_func(assemble_program(sub_tree_arg1, label_seq)(x))(assemble_program(sub_tree_arg2, label_seq)(x))
    elif len(tree_root_node.args) == 1:
        if len(tree) == 1:
            lambda_func = create_lambda(label_seq, tree_root_node)
            if is_color_prim:
                return lambda x: lambda c1: lambda c2: lambda_func(x)(c1)(c2)
            else:
                return lambda x: lambda_func(x)
        else:   
            sub_tree_arg1 = get_sub_tree(tree, tree_root_node.args[0])
            if is_color_prim:
                return lambda x: lambda c1: lambda c2: lambda_func(assemble_program(sub_tree_arg1, label_seq)(x))(c1)(c2)
            else:
                return lambda x: lambda_func(assemble_program(sub_tree_arg1, label_seq)(x))
    else:
        # print("ERROR: Invalid number of arguments for root node.")
        # print("Root node: ", tree_root_node)
        return lambda_func()
