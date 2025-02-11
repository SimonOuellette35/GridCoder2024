import numpy as np
import ARC_gym.utils.tokenization as tok
import ARC_gym.utils.visualization as viz

def is_same(a, b):
    return np.all(a == b)

# Given an input grid, an output grid, and a color changing/swapping primitive function, find the values of the
# 2 colors that yield the output grid from the input grid, if any. Return (None, None) if none work.
def color_heuristics(grid1, grid2, prim_name, prim_func, args_composed=True):
    print("prim_name: ", prim_name)
    if prim_name == 'color_swap':
        print("Output grid: ", grid2)
        for c1 in range(10):
            for c2 in range(c1 + 1, 10):
                if args_composed:
                    intermediate = prim_func(grid1)(c1)(c2)
                else:
                    intermediate = prim_func(grid1, c1, c2)
    
                intermediate_tok = tok.tokenize_grid(intermediate, max_length=931)
                print("Intermediate grid: ", intermediate_tok)
                if is_same(intermediate_tok, np.array(grid2)):
                    return (c1, c2)

    elif prim_name == 'color_change':
        for c1 in range(10):
            for c2 in range(10):
                if c1 == c2:
                    continue

                if args_composed:
                    intermediate = prim_func(grid1)(c1)(c2)
                else:
                    intermediate = prim_func(grid1, c1, c2)

                intermediate_tok = tok.tokenize_grid(intermediate, max_length=931)
                
                if is_same(intermediate_tok, np.array(grid2)):
                    return (c1, c2)

    return (None, None)

def color_heuristics_tuples(grid1, grid2, prim_name, prim_func, args_composed=True):
    if prim_name == 'color_swap':
        for c1 in range(10):
            for c2 in range(c1 + 1, 10):
                if args_composed:
                    intermediate = prim_func(grid1)(c1)(c2)
                else:
                    intermediate = prim_func(grid1, c1, c2)
    
                if is_same(intermediate.cells, grid2):
                    return (c1, c2)

    elif prim_name == 'color_change':
        for c1 in range(10):
            for c2 in range(10):
                if c1 == c2:
                    continue

                # if args_composed:
                #     print("prim_func = ", prim_func)
                #     print("grid1 = ", grid1)
                #     intermediate = prim_func(grid1)(c1)(c2)
                # else:
                intermediate = prim_func(grid1, c1, c2)

                if is_same(intermediate.get_shifted_cells(), grid2):
                    return (c1, c2)

    return (None, None)

def color_heuristics_tuplesV3(grid1, grid2, prim_name, prim_func, args_composed=True):
    if prim_name == 'color_swap':
        for c1 in range(10):
            for c2 in range(c1 + 1, 10):
                if args_composed:
                    intermediate = prim_func(grid1)(c1)(c2)
                else:
                    intermediate = prim_func(grid1, c1, c2)
    
                if is_same(intermediate, grid2):
                    return (c1, c2)

    elif prim_name == 'color_change':
        for c1 in range(10):
            for c2 in range(10):
                if c1 == c2:
                    continue

                intermediate = prim_func(grid1, c1, c2)

                #viz.draw_grid_triple(grid1.cells, intermediate.cells, grid2.cells)
                if is_same(intermediate.get_shifted_cells(), grid2):
                    return (c1, c2)

    return (None, None)

def generate_sequence_tree(distributions, min_prob=1e-5):
    print("distributions.shape: ", distributions.shape)
    
    def calc_seq_prob(sequence):
        root_prob = 1.0
        for i in range(distributions.shape[1]):
            if sequence[i] == 3:  # EOS_TOKEN
                break
            root_prob *= distributions[0, i, sequence[i]]

        return root_prob
    
    def dfs(distributions, p_threshold):

        seq_indices = []
        for i in range(distributions.shape[1]):
            valid_indices = []
            for idx, p in enumerate(distributions[0, i]):
                if p > p_threshold:
                    valid_indices.append((idx, p))

            valid_indices.sort(key=lambda x: x[1], reverse=True)
            seq_indices.append(valid_indices)

        return seq_indices

    # TODO: iterate on this function to keep generating lower probability programs
    def generate_possible_programs(seq_indices):

        def get_copy(seq_indices):
            return [x for x in seq_indices]

        programs = []
        for token_idx in range(len(seq_indices)):
            if len(seq_indices[token_idx]) <= 1:
                continue
            
            prog = get_copy(seq_indices)
            prog[token_idx] = seq_indices[token_idx][1:]

            joint_prob = 1.0
            for token_prob_list in prog:
                prob = token_prob_list[0][1]
                token_idx = token_prob_list[0][0]
                joint_prob *= prob

                if token_idx == 3:
                    break

            programs.append((prog, joint_prob))

        return programs

    root_sequence = tuple(np.argmax(distributions[0, i]) for i in range(distributions.shape[1]))
    print("root_sequence: ", root_sequence)
    root_prob = calc_seq_prob(root_sequence)
    print("root_prob: ", root_prob)

    seq_indices = dfs(distributions, p_threshold=0.1)
    programs = generate_possible_programs(seq_indices)
    return programs
