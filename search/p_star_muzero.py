import math
import search.program_interpreter as pi
import numpy as np
import utils.heuristics as heur
import Hodel_primitives_atomic as hp
import torch
import utils.grid_utils as g
import ARC_gym.utils.tokenization as tok
import ARC_gym.utils.visualization as viz
from model.heuristic import ManualHeuristic
import random
import csv

VERBOSE = False
VIZ = False
EOS_TOKEN = 3
NUM_SPECIAL_TOKENS = 4

man_heuristic = ManualHeuristic(hp.semantics)

class Node:
    def __init__(self, token_idx=-1, prob=1):
        self.token_idx = token_idx
        self.children = []
        self.prob = prob
        self.visits = 0
        self.value = 0
        self.is_expanded = False
        self.is_terminal = token_idx == EOS_TOKEN

    def update_probability_dist(self, token_prob_dist):
        self.is_expanded = True
        self.token_prob_dist = token_prob_dist

    def __str__(self):
        return f"Node(token_idx={self.token_idx}, expanded={self.is_expanded}, terminal={self.is_terminal}, value={self.value:.2f})"

    def __repr__(self):
        return self.__str__()

def select_node(node):
    if not node.is_expanded:
        return node

    explore_rate = 1. - (node.visits / 40.)

    if explore_rate <= 0.1:
        explore_rate = 0.1

    a = np.random.uniform()
    explore = False
    if a < explore_rate:
        explore = True
        print("--> Exploring! (node.visits = %i, explore_rate = %.2f)" % (node.visits, explore_rate))
    else:
        print("--> Exploiting (node.visits = %i, explore_rate = %.2f)" % (node.visits, explore_rate))

    scores = []
    max_child = None
    max_score = 0.
    for child in node.children:
        if child.visits == 0:
            u = child.prob
        else:
            ev = child.visits / 10
            if ev > 1:
                ev = 1.

            if explore:
                #u = (ev * child.value + (1 - ev) * child.prob) * (1 - (child.visits / node.visits))
                u = child.prob * (1 - (child.visits / node.visits))
            else:
                #u = (ev * child.value + (1 - ev) * child.prob)
                u = child.value

        scores.append(u)

        if u > max_score:
            max_score = u
            max_child = child

        if VERBOSE:
            if child.visits > 0 or child.prob > 0.1:
                print("Child %i (%s) with prob %.2f, value %.4f, visits: %i, score = %.4f" % (
                    child.token_idx,
                    hp.inverse_lookup(child.token_idx - NUM_SPECIAL_TOKENS),
                    child.prob,
                    child.value,
                    child.visits,
                    u
                ))


    if explore:
        # Convert scores to probabilities
        scores = np.array(scores)
        probabilities = scores / np.sum(scores)

        # Randomly sample a child based on the calculated probabilities
        selected_child = random.choices(node.children, weights=probabilities, k=1)[0]
    else:
        selected_child = max_child

    if VERBOSE:
        print("Selecting ", selected_child.token_idx)
    return selected_child

def expand_node(node, path, example_grid_set, model, input_vocab_size=13, device='cuda'):
    X, Y = example_grid_set
    label_seq = path_to_label_seq(path)
    shifted_label_seq = [EOS_TOKEN] + label_seq

    if VERBOSE:
        print("==> shifted_label_seq = ", shifted_label_seq)
    shifted_label_seq = torch.tensor(shifted_label_seq, dtype=torch.long).unsqueeze(0).to(device)

    EXAMPLE_NUM = 1
    token_probs = model.predict(X[EXAMPLE_NUM], Y[EXAMPLE_NUM], shifted_label_seq)
    node.update_probability_dist(token_probs[0].cpu().data.numpy())

    for token, prob in enumerate(token_probs[0].cpu().data.numpy()):
        if token == 0:  # Skip padding token
            continue

        child = Node(token, prob)

        tmp_lbl_seq = path_to_label_seq(path + [child])
        if pi.is_valid_partial_program(tmp_lbl_seq):
            node.children.append(child)

    node.is_expanded = True

def path_to_label_seq(path):
    return [node.token_idx for node in path if node.token_idx != -1]

def evaluate_program(path, example_grid_set):
    label_seq = path_to_label_seq(path)
    
    try:
        print("Evaluating program: ", label_seq)
        if not pi.is_valid_program(label_seq):
            print("==> NOT A VALID PROGRAM!")
            return 0.0

        program_tree = pi.generate_syntax_trees(np.array(label_seq))
        program_func = pi.assemble_program(program_tree, np.array(label_seq))

        gridX, gridY = example_grid_set
        correct_count = 0

        sims = []
        for k_idx in range(len(gridX)):
            tuple_grid_X = tok.detokenize_grid_unpadded(gridX[k_idx])
            tuple_grid_Y = tok.detokenize_grid_unpadded(gridY[k_idx])

            if pi.get_num_lambda_func_args(program_func) == 1:
                output_grid = program_func(tuple_grid_X)
            else:
                color_primitives = ['color_swap', 'color_change']
                is_color_primitive = any(hp.inverse_lookup(label - NUM_SPECIAL_TOKENS) in color_primitives for label in label_seq if label > 1)
                
                if not is_color_primitive:
                    return 0.0
                
                prim_name = 'color_swap' if 'color_swap' in [hp.inverse_lookup(label - NUM_SPECIAL_TOKENS) for label in label_seq if label > 1] else 'color_change'
                c1, c2 = heur.color_heuristics_tuples(tuple_grid_X, tuple_grid_Y, prim_name, program_func, args_composed=False)
                
                if c1 is None or c2 is None:
                    print("==> Could not find any color combination applied to %s that could solve the problem." % prim_name)
                    return 0.0

                print("==> Found color combination %s(%i, %i)!" % (prim_name, c1, c2))
                output_grid = program_func(tuple_grid_X, c1, c2)

            if VIZ:
                viz.draw_grid_triple(tuple_grid_X, output_grid, tuple_grid_Y)

            output_grid_tok = tok.tokenize_grid(output_grid, max_length=931)
            sim = man_heuristic.get_similarity(output_grid_tok, gridY[k_idx])
            if np.all(output_grid_tok == gridY[k_idx]):
                sim = 1.
            sims.append(sim)

        return np.median(sims)
    except:
        print("==> Invalid program, an exception occurred while running it.")
        return 0.0

def search(model, example_grid_set_tensor, example_token_seqs, time_budget, max_iterations, max_depth):
    root = Node()

    for iteration in range(max_iterations):
        print(f"==> Iteration: {iteration}")
        
        path = [root]
        node = root

        depth = 0
        while depth < max_depth:
            depth += 1

            label_seq = path_to_label_seq(path)
            # Selection
            while node.is_expanded and not node.is_terminal:
                node = select_node(node)
                path.append(node)

            # Expansion
            if not node.is_terminal:
                expand_node(node, path, example_grid_set_tensor, model)

            # Evaluation
            if node.is_terminal:
                value = evaluate_program(path, example_token_seqs)
                print("\tReturned value = ", value)

                # Backpropagation
                for node in reversed(path):
                    node.visits += 1
                    node.value = (node.value * (node.visits - 1) + value) / node.visits

                if value == 1.0:
                    return path

                break


    return None
