import re


def decompose_nested_calls(s):
    result = []
    stack = []

    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                # Find the function name start
                func_start = start
                while func_start > 0 and s[func_start - 1].isalnum() or s[func_start - 1] == '_':
                    func_start -= 1
                result.append(s[func_start:i + 1])

    # Add individual variables (excluding function names)
    func_names = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?=\()', s))
    variables = set(re.findall(r'\b[a-zA-Z0-9_]+\b', s)) - func_names
    result.extend(variables)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(result))

def extract_arguments(function_call):
    args = []
    current_arg = ""
    parentheses_count = 0
    in_argument = False

    for char in function_call:
        if char == '(' and not in_argument:
            in_argument = True
            continue
        elif char == ')' and parentheses_count == 0 and in_argument:
            args.append(current_arg.strip())
            break

        if in_argument:
            if char == '(':
                parentheses_count += 1
            elif char == ')':
                parentheses_count -= 1
            elif char == ',' and parentheses_count == 0:
                args.append(current_arg.strip())
                current_arg = ""
                continue

            current_arg += char

    return args


def subpath_match(a_name, subpath, list_of_subpaths):

    for tmp_subpath in list_of_subpaths:
        if tmp_subpath.startswith(a_name):
            str_args = extract_arguments(tmp_subpath)
            for str_arg in str_args:
                if subpath == str_arg:
                    return tmp_subpath
    return None


def get_depth(path):
    d = 0.
    max_d = 0.
    for c in path:
        if c == '(':
            d += 1.

        if c == ')':
            d -= 1.

        if d > max_d:
            max_d = d

    return max_d

def get_path_from_perm(perm):
    path_combo = '['
    for idx, arg in enumerate(perm):
        path_combo += arg.generating_path

        if idx < len(perm) - 1:
            path_combo += ','
        else:
            path_combo += ']'

    return path_combo

def get_path_from_tuple(perm):
    path_combo = '['
    for idx, arg in enumerate(perm):
        path_combo += arg[-1]

        if idx < len(perm) - 1:
            path_combo += ','
        else:
            path_combo += ']'

    return path_combo

def get_eval_path(prim_name, perm):
    path_combo = '('
    for idx, arg in enumerate(perm):
        path_combo += arg.generating_path

        if idx < len(perm) - 1:
            path_combo += ','
        else:
            path_combo += ')'

    return "%s%s" % (prim_name, path_combo)

def get_paths_for_depth(path, depth):
    subpath_list = decompose_nested_calls(path)

    output = []
    for subpath in subpath_list:
        d = get_depth(subpath)

        if d == depth:
            output.append(subpath)

    return output

def get_prim_name_from_subpath(subpath):
    if '(' not in subpath:
        return None

    return subpath.split('(')[0]

def lookup_input_paths(intermediate_grids, inp_paths):
    output_grids = []
    for inp_path in inp_paths:
        output_grids.append(intermediate_grids[inp_path])

    return output_grids

def permute(tmp):

    if len(tmp) == 1:
        output = []
        for t in tmp[0]:
            output.append([t])

        return output

    output = []
    for t in tmp[0]:

        perms = permute(tmp[1:])
        for p in perms:
            if isinstance(p, int):
                output.append([t, p])
            elif isinstance(p, tuple):
                output.append([t, *p])
            else:
                inner = [t]
                for p_elem in p:
                    inner.append(p_elem)

                output.append(inner)

    return output
