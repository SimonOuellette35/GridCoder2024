import math
import search.program_interpreter as pi
import numpy as np
import utils.heuristics as heur
import Hodel_primitives_atomicV2 as hp
import torch
import utils.grid_utils as g
import ARC_gym.utils.tokenization as tok
import ARC_gym.utils.visualization as viz
import time

VERBOSE = False
VIZ = False
EOS_TOKEN = 3
NUM_SPECIAL_TOKENS = 4

last_valid_path = []

class Node:
    def __init__(self, token_idx=-1, prob=1):
        self.token_idx = token_idx                  # -1 refers to the root node, should not be part of label sequence.
        self.token_prob_dist = []       # Probability for each child node
        self.children = []  # Child nodes (possible next token sequences)

        self.prob = prob
        self.visits = 0  # Number of times this node has been visited      
        self.is_expanded = False  # Whether the node has been fully expanded

        if token_idx == EOS_TOKEN:
            self.is_terminal = True
        else:
            self.is_terminal = False

    def update_probability_dist(self, token_prob_dist):
        self.is_expanded = True
        self.token_prob_dist = token_prob_dist

    def __str__(self):
        return f"Node(token_idx=[{self.token_idx}], expanded={self.is_expanded}, terminal={self.is_terminal})"

    def __repr__(self):
        return self.__str__()

# To avoid re-executing the same program multiple times, only pick terminal nodes that have not been visited yet
def criterion(child_node, total_visits, depth):

    if child_node.is_terminal and child_node.visits > 0:
        return 0
    else:
        score = child_node.prob
        
        if child_node.token_idx == 1:
            prim_name = '<NEW LEVEL>'
        elif child_node.token_idx == 2:
            prim_name = '<IDENTITY>'
        elif child_node.token_idx == 3:
            prim_name = '<EOS>'
        else:
            prim_name = hp.inverse_lookup(child_node.token_idx - NUM_SPECIAL_TOKENS)

        if VERBOSE:
            print("\tdepth = %i, child_node token: %i (%s), prob: %.4f" % (depth, child_node.token_idx, prim_name, child_node.prob))
        return score

def expand_node(node, path, example_grid_set, model, input_vocab_size=13, device='cuda'):
    # Use the transformer model to predict the next token probabilities
    X, Y = example_grid_set

    # TODO: try merging the probabilities over all grid examples

    label_seq = path_to_label_seq(path)
    shifted_label_seq = [EOS_TOKEN] + label_seq  # Remove last element to maintain fixed length

    if VERBOSE:
        print("==> shifted_label_seq = ", shifted_label_seq)
    shifted_label_seq = torch.tensor(shifted_label_seq, dtype=torch.long).to(device)

    # Add an extra dimension to shifted_label_seq
    shifted_label_seq = shifted_label_seq.unsqueeze(0)

    #start_time = time.time()
    EXAMPLE_NUM = 1
    token_probs = model.predict(X[EXAMPLE_NUM], Y[EXAMPLE_NUM], shifted_label_seq)      # TODO: ability to start from some target_seq and predict the next token's probs from there.

    node.update_probability_dist(token_probs[0].cpu().data.numpy())

    for token, prob in enumerate(token_probs[0].cpu().data.numpy()):
        if token == 0:  # Skip padding token
            continue

        child = Node(token, prob)

        # We prune children that correspond to invalid partial programs
        tmp_lbl_seq = path_to_label_seq(path + [child])
        if pi.is_valid_partial_program(tmp_lbl_seq):
            node.children.append(child)
        # else:
        #     print("==> excluded token %i due to it generating the invalid program: %s" % (child.token_idx, tmp_lbl_seq))

    node.is_expanded = True

def path_to_label_seq(path):
    label_seq = []
    for node in path:
        if node.token_idx != -1:
            label_seq.append(node.token_idx)

    return label_seq

def get_prediction(path, gridX, c1=None, c2=None, gridY=None):
    global last_valid_path

    label_seq = path_to_label_seq(path)
    
    try:
        print("Evaluating program: ", label_seq)
        if not pi.is_valid_program(label_seq):
            print("==> NOT A VALID PROGRAM!")
            return None, None, None

        last_valid_path = path

        # run the program interpreter on the task
        program_tree = pi.generate_syntax_trees(np.array(label_seq))
        program_func = pi.assemble_program(program_tree, np.array(label_seq))

        # execute the program on the input grid
        output_grids = []
        for k_idx in range(len(gridX)):
            tuple_grid_X = tok.detokenize_grid_unpadded(gridX[k_idx])

            if pi.get_num_lambda_func_args(program_func) == 1:
                output_grid = program_func(tuple_grid_X)
            else:
                if c1 is None and c2 is None:
                    tuple_grid_Y = tok.detokenize_grid_unpadded(gridY[k_idx])

                # Check if color_swap or color_change is in the label sequence
                color_primitives = ['color_swap', 'color_change']
                is_color_primitive = any(hp.inverse_lookup(label - NUM_SPECIAL_TOKENS) in color_primitives for label in label_seq if label > 1)
                
                if is_color_primitive:
                    prim_name = 'color_swap' if 'color_swap' in [hp.inverse_lookup(label - NUM_SPECIAL_TOKENS) for label in label_seq if label > 1] else 'color_change'
                else:
                    return False
                
                if c1 is None and c2 is None:
                    c1, c2 = heur.color_heuristics_tuples(tuple_grid_X, tuple_grid_Y, prim_name, program_func, args_composed=True)

                    if c1 is None or c2 is None:
                        print("==> Could not find any color combination applied to %s that could solve the problem." % prim_name)
                        output_grid = program_func(tuple_grid_X)(1)(2)
                        output_grids.append(output_grid)
                        c1 = 1
                        c2 = 2

                    print("==> Found color combination %s(%i, %i)!" % (prim_name, c1, c2))

                output_grid = program_func(tuple_grid_X)(c1)(c2)

            # if VIZ:
            #     tuple_grid_Y = tok.detokenize_grid_unpadded(gridY[k_idx])
            #     print(output_grid)
            #     viz.draw_grid_triple(tuple_grid_X, output_grid, tuple_grid_Y)
            output_grids.append(output_grid)
        
        return output_grids, c1, c2
            
    except:
        import traceback
        print("==> Invalid program, an exception occurred while running it")
        #print(traceback.format_exc())
        return None, None, None


def evaluate_program(path, example_grid_set):
    gridX, gridY = example_grid_set
    output_grids, c1, c2 = get_prediction(path, gridX, gridY=gridY)

    if output_grids is None:
        return False, None, None
    
    for k_idx in range(len(output_grids)):
        output_grid_tok = tok.tokenize_grid(output_grids[k_idx], max_length=931)
        if np.any(output_grid_tok != gridY[k_idx]):
            print("==> Program output does not match ground truth.")
            return False, None, None
            
    return True, c1, c2

def arg_max_expansion(model, path, example_grid_set_tensor, example_token_seqs, expansion_queue, depth, max_depth, score_threshold):
    root = path[0]

    while depth < max_depth:
        node = path[-1]

        if VERBOSE:
            print("==> Current node: ", node)

        # Expansion: expand the selected node if it hasn't been expanded yet
        if not node.is_expanded and not node.is_terminal:
            expand_node(node, path, example_grid_set_tensor, model, expansion_queue)

        # Selection: traverse down using a selection criterion to balance exploration and exploitation
        while node.is_expanded and not node.is_terminal:
            
            if VERBOSE:
                print("\tNode is expanded and not terminal. Selecting best child...")

            # Update the expansion queue...
            max_node = node
            max_prob = 0.
            
            for n in node.children:
                score = criterion(n, root.visits, len(path))

                if score > max_prob:
                    max_prob = score
                    max_node = n
                  
            # if the best score is 0 (or below a certain threshold), we reached a dead end and should move on to next iteration
            if max_prob < score_threshold:
                print("Best score is below threshold. Stopping current iteration...")
                return None, None, None

            for n in node.children:
                if n.token_idx != max_node.token_idx and n.prob  <= max_prob:
                    nearest_prob = n.prob
                    nearest_node = n

                    tmp_lbl_seq = path_to_label_seq(path + [n])
                    if pi.is_valid_partial_program(tmp_lbl_seq):
                        add_to_expansion_queue(path, depth, nearest_node, nearest_prob, max_prob, expansion_queue)

            if VERBOSE:
                print("\tMax child score: ", max_prob)
            node = max_node
            path.append(node)
            depth += 1

        # Update visit count
        for n in reversed(path):
            n.visits += 1

        # Simulation/Evaluation: run the program and get the binary result
        # Only makes sense to evaluate terminal nodes, since only terminal nodes correspond to a full program
        if node.is_terminal:
            result, c1, c2 = evaluate_program(path, example_token_seqs)
            print("\tResult: ", result)
            if c1 is not None:
                print("\tc1 = %i, c2 = %i" % (c1, c2))

            if result:
                return path, c1, c2
            else:
                return None, None, None
                
    # Failed to find a valid program
    return None, None, None

def get_child_node_to_expand(expansion_queue):
    # as we expand new nodes, push prob deltas to an expansion_queue. This queue
    # helps us prioritize which are the next nodes to expand.
    # print("==> Expansion queue: ")
    # for entry in expansion_queue:
    #     print("\tprob_delta = ", entry[0])
    #     lbl_seq = path_to_label_seq(entry[1])
    #     print("\tPath = ", lbl_seq)
    #     print("\t=================================================================================")
    
    expansion_entry = expansion_queue.pop(0)
    node_path = expansion_entry[1]
    node_depth = expansion_entry[2]

    return node_path, node_depth

# TODO: also consider the joint probability of parents when prob_delta is very similar...
def add_to_expansion_queue(path, depth, node, nearest_prob, max_prob, expansion_queue):
    #prob_delta = max_prob - nearest_prob
    prob_delta = 1 - (nearest_prob / max_prob)

    path_to_expand = path + [node]
    inserted = False
    for idx in range(len(expansion_queue)):
        cur_delta = expansion_queue[idx][0]
        if prob_delta < cur_delta:
            #print("Adding prob_delta %.4f for token %i @ position %i" % (prob_delta, node.token_idx, idx))
            expansion_queue.insert(idx, (prob_delta, path_to_expand, depth + 1))
            inserted = True
            break

    if not inserted:
        expansion_queue.append((prob_delta, path_to_expand, depth + 1))

# example_grid_set is an (X, Y) tuple of input grid set and target grid set
# X and Y are lists of k examples.
def search(model, example_grid_set_tensor, example_token_seqs, time_budget, max_iterations, max_depth, score_threshold=0.01):

    global last_valid_path

    expansion_queue = []
    
    print("==> Iteration 0 (arg max expansion): ")
    root = Node()
    root_path = [root]
    last_valid_path = root_path

    result, c1, c2 = arg_max_expansion(model, root_path, example_grid_set_tensor, example_token_seqs, expansion_queue, 0, max_depth, score_threshold)

    start_time = time.time()

    #print("==> Expansion queue after iteration 0: ", expansion_queue)
    # If arg max expansion yields the correct program, return immediately.
    if result is not None:
        return result, c1, c2, True

    for iteration in range(1, max_iterations):

        current_time = time.time()
        if (current_time - start_time) > time_budget:
            print("==> Timeout!")
            return last_valid_path, c1, c2, False

        print("==> Iteration: ", iteration)

        if len(expansion_queue) == 0:
            return last_valid_path, c1, c2, False

        if VERBOSE:
            print("==> Using expansion queue: ")
            for idx in range(min(len(expansion_queue), 10)):
                print(expansion_queue[idx])

        
        node_path, node_depth = get_child_node_to_expand(expansion_queue)

        lbl_seq = path_to_label_seq(node_path)

        if VERBOSE:
            print("==> Generating next program by expanding at: ", lbl_seq)

        result, c1, c2 = arg_max_expansion(model, node_path, example_grid_set_tensor, example_token_seqs, expansion_queue, node_depth, max_depth, score_threshold)

        if result is not None:
            return result, c1, c2, True

    # Failed to find a valid program
    return last_valid_path, c1, c2, False
