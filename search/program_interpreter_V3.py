import numpy as np
import inspect
import ast


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

def get_num_args(func, hp):
    if func == IDENTITY_TOKEN:
        return 1
    
    prim_name = hp.inverse_lookup(func-NUM_SPECIAL_TOKENS)
    num_args = hp.get_num_args(prim_name)
    return num_args

def get_num_lambda_func_args(lambda_func):
    import inspect

    def count_lambda_args(func):
        if not callable(func):
            return 0
        sig = inspect.signature(func)
        return len(sig.parameters)

    count = 0
    while callable(lambda_func):
        count += count_lambda_args(lambda_func)
        if count == 0:
            break
        # Try to get the next nested function without invoking the lambda
        try:
            lambda_func = lambda_func.__code__.co_consts[0]
        except (AttributeError, IndexError):
            break
    return count


def generate_syntax_trees(token_seq, hp):

    # Find first occurrence of EOS_TOKEN
    eos_indices = np.where(token_seq == EOS_TOKEN)[0]
    if len(eos_indices) > 0:
        # Truncate sequence after first EOS token
        token_seq = token_seq[:eos_indices[0]+1]

    # Get indices of non-zero elements (functions)
    func_indices = np.nonzero(token_seq)[0]

    # Get indices of NEW_LEVEL elements (level separators)
    level_separators = np.where(token_seq == 1)[0]

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

            num_args = get_num_args(token_seq[func], hp)
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

def is_valid_program(label_seq, hp):
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
        #print("==> Program validation: Levels is empty!")
        return False

    # Validate each level
    prev_outputs = None
    for i, level in enumerate(levels):
        outputs = len(level)
        inputs = sum(get_num_args(label, hp) for label in level)

        # Validate last level has only 1 output
        if i == len(levels) - 1 and outputs != 1:
            #print("==> Program validation: last level has %i outputs!" % outputs)
            return False

        # Validate inputs match previous level's outputs (except for first level)
        if i > 0 and inputs != prev_outputs:
            #print("==> Program validation: at level %i, number of inputs (%i) not matching number of previous outputs (%i)" % (i, inputs, prev_outputs))
            return False

        prev_outputs = outputs

    return True

def convert_to_prim_index(label_seq, func_idx):
    return label_seq[func_idx]

def create_lambda(label_seq, node, hp):
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

    #print("ERROR: could not find referenced argument in lower level.")

def apply_partial_program(token_seq, input_grid_set, hp):
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


# def assemble_program(tree, label_seq, hp, is_lambda=False):
#     tree_root_node = tree[-1][0]

#     lambda_func = create_lambda(label_seq, tree_root_node, hp)
#     is_color_prim = False
#     is_apply_to_grid = False
#     is_for_each = False
#     prim_index = convert_to_prim_index(label_seq, tree_root_node.func)
#     if prim_index > IDENTITY_TOKEN:
#         prim_name = hp.inverse_lookup(prim_index - NUM_SPECIAL_TOKENS)
#         is_color_prim = prim_name.startswith('color_')
#         is_apply_to_grid = prim_name == 'apply_to_grid'
#         is_for_each = prim_name == 'for_each'

#     print("==> Processing prim_index %i (is_apply_for_grid = %s, is_for_each = %s, is_lambda=%s)" % (prim_index, is_apply_to_grid, is_for_each, is_lambda))

#     if len(tree_root_node.args) == 2:
#         if len(tree) <= 1:
#             return lambda_func

#         sub_tree_arg1 = get_sub_tree(tree, tree_root_node.args[0])
#         sub_tree_arg2 = get_sub_tree(tree, tree_root_node.args[1])

#         if is_color_prim:
#             if is_lambda:
#                 return lambda x, c1, c2: lambda_func(assemble_program(sub_tree_arg1, label_seq, hp, is_lambda=True)(x))(assemble_program(sub_tree_arg2, label_seq, hp, is_lambda=True))(c1)(c2)
#             else:
#                 return lambda x, c1, c2: lambda_func(assemble_program(sub_tree_arg1, label_seq, hp)(x))(assemble_program(sub_tree_arg2, label_seq, hp)(x))(c1)(c2)
#         elif is_for_each:
#             return lambda x: lambda_func(assemble_program(sub_tree_arg1, label_seq, hp)(x))(lambda y: assemble_program(sub_tree_arg2, label_seq, hp, is_lambda=True)(y))
#         else:
#             if is_lambda:
#                 return lambda x: lambda_func(assemble_program(sub_tree_arg1, label_seq, hp, is_lambda=True)(x))(assemble_program(sub_tree_arg2, label_seq, hp, is_lambda=True))
#             else:
#                 return lambda x: lambda_func(assemble_program(sub_tree_arg1, label_seq, hp)(x))(assemble_program(sub_tree_arg2, label_seq, hp)(x))

#     elif len(tree_root_node.args) == 1:
#         if len(tree) == 1:
#             print("prim_index = %i, len(tree) == 1" % prim_index)
#             lambda_func = create_lambda(label_seq, tree_root_node, hp)
#             if is_lambda:
#                 return lambda_func
#             if is_color_prim:
#                 return lambda x, c1, c2: lambda_func(x)(c1)(c2)
#             elif is_apply_to_grid:
#                 return lambda f, x: lambda_func(f)(x)
#             else:
#                 return lambda x: lambda_func(x)
#         else:   
#             print("prim_index = %i, len(tree) != 1" % prim_index)
#             sub_tree_arg1 = get_sub_tree(tree, tree_root_node.args[0])
#             if is_lambda:
#                 return lambda x: lambda_func(lambda y: assemble_program(sub_tree_arg1, label_seq, hp, is_lambda)(y))(x)
#             if is_color_prim:
#                 return lambda x, c1, c2: lambda_func(assemble_program(sub_tree_arg1, label_seq, hp)(x))(c1)(c2)
#             elif is_apply_to_grid:
#                 return lambda f, x: lambda_func(f)(assemble_program(sub_tree_arg1, label_seq, hp)(x))
#             else:
#                 return lambda x: lambda_func(assemble_program(sub_tree_arg1, label_seq, hp)(x))
#     else:
#         print("ERROR: Invalid number of arguments for root node.")
#         print("Root node: ", tree_root_node)
#         if is_lambda:
#             return lambda y: lambda_func(y)
#         else:
#             return lambda_func()

def write_program(tree, label_seq, hp):
    def check_color_primitive(label_seq, hp):
        for label in label_seq:
            if label > NUM_SPECIAL_TOKENS:
                prim_name = hp.inverse_lookup(label - NUM_SPECIAL_TOKENS)
                if prim_name.startswith('color_'):
                    return True
        return False

    has_color_primitive = check_color_primitive(label_seq, hp)

    if has_color_primitive:
        return f'lambda x, c1, c2: {assemble_program_string(tree, label_seq, hp)}'
    else:
        return f'lambda x: {assemble_program_string(tree, label_seq, hp)}'

def assemble_program_string(tree, label_seq, hp, is_lambda=False):
    tree_root_node = tree[-1][0]

    is_color_prim = False
    is_apply_to_grid = False
    is_for_each = False
    prim_index = convert_to_prim_index(label_seq, tree_root_node.func)

    if prim_index != 2:  
        prim_name = hp.inverse_lookup(prim_index - NUM_SPECIAL_TOKENS)

        is_color_prim = prim_name.startswith('color_')
        is_apply_to_grid = prim_name == 'apply_to_grid'
        is_for_each = prim_name == 'for_each'
    else:
        if tree_root_node.args[0] == -1:
            if is_lambda:
                return 'y'
            else:
                return 'x'
        else:
            sub_tree_arg1 = get_sub_tree(tree, tree_root_node.args[0])
            return assemble_program_string(sub_tree_arg1, label_seq, hp, is_lambda=is_lambda)
    
    if len(tree_root_node.args) == 2:
        if len(tree) <= 1:
            return prim_name

        sub_tree_arg1 = get_sub_tree(tree, tree_root_node.args[0])
        sub_tree_arg2 = get_sub_tree(tree, tree_root_node.args[1])

        if is_color_prim:
            if is_lambda:
                return f"lambda x, c1, c2: {prim_name}({assemble_program_string(sub_tree_arg1, label_seq, hp, is_lambda=True)}(x))({assemble_program_string(sub_tree_arg2, label_seq, hp, is_lambda=True)})(c1)(c2)"
            else:
                return f"lambda x, c1, c2: {prim_name}({assemble_program_string(sub_tree_arg1, label_seq, hp)}(x))({assemble_program_string(sub_tree_arg2, label_seq, hp)}(x))(c1)(c2)"
        elif is_for_each:
            return f"for_each({assemble_program_string(sub_tree_arg1, label_seq, hp)})(lambda y: {assemble_program_string(sub_tree_arg2, label_seq, hp, is_lambda=True)})"
        else:
            return f"{prim_name}({assemble_program_string(sub_tree_arg1, label_seq, hp, is_lambda=is_lambda)})({assemble_program_string(sub_tree_arg2, label_seq, hp, is_lambda=is_lambda)})"

    elif len(tree_root_node.args) == 1:
        if is_apply_to_grid:
            sub_tree_arg1 = get_sub_tree(tree, tree_root_node.args[0])
            return f'apply_to_grid(x)({assemble_program_string(sub_tree_arg1, label_seq, hp)})'

        elif tree_root_node.args[0] == -1:
            if is_lambda:
                if is_color_prim:
                    return f'{prim_name}(y)(c1)(c2)'
                else:
                    return f'{prim_name}(y)'
            else:
                if is_color_prim:
                    return f'{prim_name}(x)(c1)(c2)'
                else:
                    return f'{prim_name}(x)'
        else:
            sub_tree_arg1 = get_sub_tree(tree, tree_root_node.args[0])
            if is_color_prim:
                return f"{prim_name}({assemble_program_string(sub_tree_arg1, label_seq, hp, is_lambda=is_lambda)})(c1)(c2)"
            else:
                return f"{prim_name}({assemble_program_string(sub_tree_arg1, label_seq, hp, is_lambda=is_lambda)})"
    else:
        print("ERROR: Invalid number of arguments for root node.")
        print("Root node: ", tree_root_node)
        if is_lambda:
            return f"lambda y: {prim_name}(y)"
        else:
            return str(prim_name)

def compile_program(description, primitives):
    tree = ast.parse(description)
    # Check if the program uses color parameters
    has_color_params = 'c1' in description and 'c2' in description
    if has_color_params:
        # Add color parameters to primitives if needed
        primitives = {**primitives, 'c1': None, 'c2': None}
    return eval_ast(tree.body[0].value, primitives)

def eval_ast(node, primitives, env=None):
    if env is None:
        env = {}
    if isinstance(node, ast.Lambda):
        # Create a closure that captures color parameters if they exist
        def lambda_wrapper(*args):
            local_env = {**env}
            # Map arguments to parameter names
            for arg, val in zip(node.args.args, args):
                local_env[arg.arg] = val
            return eval_ast(node.body, primitives, local_env)
        return lambda_wrapper
    elif isinstance(node, ast.Call):
        func = eval_ast(node.func, primitives, env)
        args = [eval_ast(arg, primitives, env) for arg in node.args]
        return func(*args)
    elif isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        elif node.id in primitives:
            return primitives[node.id]
        else:
            raise NameError(f"Name '{node.id}' is not defined")
    elif isinstance(node, ast.Attribute):
        obj = eval_ast(node.value, primitives, env)
        return getattr(obj, node.attr)
    else:
        return ast.literal_eval(node)
