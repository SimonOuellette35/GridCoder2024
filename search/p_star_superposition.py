import math
import search.program_interpreter_V3 as pi
import numpy as np
import utils.heuristics as heur
import Hodel_primitives_atomicV3 as hp
import torch
import utils.grid_utils as g
import ARC_gym.utils.tokenization as tok
import ARC_gym.utils.visualization as viz
import time
import copy


DET_SEED = 12345

np.random.seed(DET_SEED)


VERBOSE = False
VIZ = False
EOS_TOKEN = 3
NUM_SPECIAL_TOKENS = 4

def get_label_seq_str(label_seq):

    label_seq_str = []
    for lbl in label_seq:
        if lbl < 4:
            if lbl == 0:
                label_seq_str.append("<PAD>")
            elif lbl == 1:
                label_seq_str.append("<NEW LEVEL>")
            elif lbl == 2:
                label_seq_str.append("<IDENTITY>")
            elif lbl == 3:
                label_seq_str.append("<EOS>")
                break
        else:
            label_seq_str.append(hp.inverse_lookup(lbl-NUM_SPECIAL_TOKENS))

    return label_seq_str

def get_prediction(label_seq, gridX, c1=None, c2=None, gridY=None, verbose=False):
   
    #print("Getting prediction for label_seq: ", label_seq)
    try:
        if verbose:
            label_seq_str = get_label_seq_str(label_seq)
            print("Evaluating program: ", label_seq_str)

        if not pi.is_valid_program(label_seq, hp):
            if verbose:
                print("==> NOT A VALID PROGRAM!")
            return None, None, None

        # run the program interpreter on the task
        program_tree = pi.generate_syntax_trees(np.array(label_seq), hp)
        if verbose:
            print("program_tree = ", program_tree)

        program_string = pi.write_program(program_tree, np.array(label_seq), hp)
        
        if verbose:
            print("Program string: ", program_string)

        program_func = pi.compile_program(program_string, hp.semantics)

        # execute the program on the input grid
        output_grids = []
        for k_idx in range(len(gridX)):
            tuple_grid_X = tok.detokenize_grid_unpadded(gridX[k_idx])

            input_grid = hp.Grid(tuple_grid_X)
            num_func_args = pi.get_num_lambda_func_args(program_func)

            if num_func_args == 1 and "color_change" not in program_string:
                output_grid = program_func(input_grid)
                if isinstance(output_grid, list):
                    output_grid = output_grid[0]
            else:
                if c1 is None and c2 is None:
                    tuple_grid_Y = tok.detokenize_grid_unpadded(gridY[k_idx])

                # Check if color_swap or color_change is in the label sequence
                color_primitives = ['color_swap', 'color_change']
                is_color_primitive = any(hp.inverse_lookup(label - NUM_SPECIAL_TOKENS) in color_primitives for label in label_seq if label > 3)
                
                if is_color_primitive:
                    prim_name = 'color_swap' if 'color_swap' in [hp.inverse_lookup(label - NUM_SPECIAL_TOKENS) for label in label_seq if label > 3] else 'color_change'
                else:
                    return False
                
                if c1 is None and c2 is None:
                    c1, c2 = heur.color_heuristics_tuplesV3(input_grid, tuple_grid_Y, prim_name, program_func, args_composed=True)

                    if c1 is None or c2 is None:
                        output_grid = program_func(input_grid, 1, 2)

                        if isinstance(output_grid, list):
                            output_grid = output_grid[0]
                        output_grids.append(output_grid)
                        c1 = 1
                        c2 = 2

                output_grid = program_func(input_grid, c1, c2)
                if isinstance(output_grid, list):
                  output_grid = output_grid[0]

            # if VIZ:
            #     tuple_grid_Y = tok.detokenize_grid_unpadded(gridY[k_idx])
            #     print(output_grid)
            #     viz.draw_grid_triple(tuple_grid_X, output_grid, tuple_grid_Y)
            output_grids.append(output_grid)
        
        return output_grids, c1, c2
            
    except:
        if verbose:
            import traceback
            print("==> Invalid program, an exception occurred while running it")
            print(traceback.format_exc())

    return None, None, None


def evaluate_program(label_seq, example_grid_set, verbose=False):
    gridX, gridY = example_grid_set
    output_grids, c1, c2 = get_prediction(label_seq, gridX, gridY=gridY, verbose=verbose)

    if output_grids is None:
        return False, None, None
    
    try:
        for k_idx in range(len(output_grids)):
            #output_grid_tok = tok.tokenize_grid(output_grids[k_idx].get_shifted_cells(), max_length=931)
            output_grid_tok = tok.tokenize_grid(output_grids[k_idx].cells, max_length=931)

            if verbose:
                print("output_grid_tok = ", output_grid_tok)
                print("gridY[k_idx] = ", gridY[k_idx])

                grid_output_viz = tok.detokenize_grid_unpadded(gridY[k_idx])
                viz.draw_grid_pair(output_grids[k_idx].cells, grid_output_viz)

            if np.any(output_grid_tok != gridY[k_idx]):
                if verbose:
                    print("==> Program output does not match ground truth.")
                return False, None, None
    except:
        if verbose:
            print("==> Exception occurred while evaluating program.")

        return False, None, None

    return True, c1, c2

def get_probability_space(model, example_grid_set, starting_seq = [], example_num = 0, device='cuda', max_seq_len=40, THRESH=0.01):
    # Use the transformer model to predict the next token probabilities
    prob_dist = []
    X, Y = example_grid_set

    shifted_label_seq = [EOS_TOKEN] + starting_seq
    shifted_label_seq = torch.tensor(shifted_label_seq, dtype=torch.long).to(device)

    # Add an extra dimension to shifted_label_seq
    shifted_label_seq = shifted_label_seq.unsqueeze(0)

    done = False
    seq_len = 0

    while not done and seq_len < max_seq_len:
        token_probs = model.predict(X[example_num], Y[example_num], shifted_label_seq[:40])
        prob_dist.append(token_probs[0].cpu().data.numpy())
        best_token = np.argmax(token_probs[0].cpu().data.numpy())

        if best_token == EOS_TOKEN:
            # Filter out probabilities below threshold
            token_probs_np = token_probs[0].cpu().data.numpy()
            prob_count = 0
            max_alternative_token = EOS_TOKEN
            max_alternative_prob = 0
            for arg_idx, p in enumerate(token_probs_np):
                if p > THRESH:
                    prob_count += 1
                    if arg_idx != EOS_TOKEN:
                        if p > max_alternative_prob:
                            max_alternative_token = arg_idx
                            max_alternative_prob = p
            
            if prob_count <= 1:
                done = True
            else:
                best_token = max_alternative_token

        # Create new tensor with additional token
        new_seq = torch.cat([shifted_label_seq, torch.tensor([[best_token]], device=shifted_label_seq.device)], dim=1)
        shifted_label_seq = new_seq
        seq_len += 1

    return prob_dist, shifted_label_seq

# example_grid_set is an (X, Y) tuple of input grid set and target grid set
# X and Y are lists of k examples.
def search(model, example_grid_set_tensor, example_token_seqs, time_budget, max_iterations, max_depth, score_threshold=0.01):

    start_time = time.time()

    THRESH = 0.01
    probability_dist, arg_max_seq = get_probability_space(model, example_grid_set_tensor, THRESH=THRESH)

    # for token_idx, prob_dist in enumerate(probability_dist):
    #     print("==> Probabilities at token %i" % (token_idx))
    #     for prob_idx, prob in enumerate(prob_dist):
    #         if prob > THRESH:
    #             if prob_idx < 4:
    #                 print("\t(%i) ==> %.2f" % (prob_idx, prob))
    #             else:
    #                 print("\t%s ==> %.2f" % (hp.inverse_lookup(prob_idx-NUM_SPECIAL_TOKENS), prob))

    probable_tokens = []
    for prob_list in probability_dist:
        token_list = []
        for prob_idx, prob in enumerate(prob_list):
            if prob > THRESH:
                token_list.append(prob_idx)

        probable_tokens.append(token_list)

    # Bootstrap probabilities by trying a K random starting tokens and example pairs
    K = 5
    prob_counts = np.ones(len(probability_dist))
    tested_start_seqs = []
    for _ in range(K):
        num = np.random.choice(np.arange(len(example_grid_set_tensor[0])))
        found = False
        for pt in probable_tokens[0]:
            start_seq = [pt]
            if start_seq not in tested_start_seqs:
                found = True
                break
        
        if not found:
            token1 = np.random.choice(probable_tokens[0])
            token2 = np.random.choice(probable_tokens[1])
            start_seq = [token1, token2]

        tmp_prob_dist, _ = get_probability_space(model, example_grid_set_tensor, starting_seq=start_seq, example_num=num, THRESH=THRESH)
        tested_start_seqs.append(start_seq)

        #print("Using start_seq = ", start_seq)
        # for token_idx, prob_dist in enumerate(tmp_prob_dist):
        #     print("==> Probabilities at token %i" % (token_idx))
        #     for prob_idx, prob in enumerate(prob_dist):
        #         if prob > THRESH:
        #             if prob_idx < 4:
        #                 print("\t(%i) ==> %.2f" % (prob_idx, prob))
        #             else:
        #                 print("\t%s ==> %.2f" % (hp.inverse_lookup(prob_idx-NUM_SPECIAL_TOKENS), prob))

        for token_idx, prob_list in enumerate(tmp_prob_dist):
            idx = token_idx + len(start_seq)
            if idx < len(probability_dist):
                prob_counts[idx] += 1
                for prob_idx, p in enumerate(prob_list):    
                    probability_dist[idx][prob_idx] += p
            else:
                probability_dist.append(prob_list)
                prob_counts = np.append(prob_counts, 1)

    for token_idx, prob_list in enumerate(probability_dist):
        for prob_idx, p in enumerate(prob_list):
            probability_dist[token_idx][prob_idx] /= float(prob_counts[token_idx])

    print("======> Final probability distribution after bootstrapping:")
    for token_idx, prob_dist in enumerate(probability_dist):
        print("==> Probabilities at token %i" % (token_idx))
        for prob_idx, prob in enumerate(prob_dist):
            if prob > THRESH:
                if prob_idx < 4:
                    print("\t(%i) ==> %.2f" % (prob_idx, prob))
                else:
                    print("\t(%i) %s ==> %.2f" % (prob_idx, hp.inverse_lookup(prob_idx-NUM_SPECIAL_TOKENS), prob))

    def enumerate_sequences(prob_dist, max_length):
        num_permutations = np.inf

        thresholds = np.ones(len(prob_dist)) * THRESH

        last_probable_tokens = []
        while num_permutations > 1000000 and (time.time() - start_time) < (time_budget - 60):

            probable_tokens = []
            for token_idx, prob_list in enumerate(prob_dist):
                token_list = []
                for prob_idx, prob in enumerate(prob_list):
                    if prob > thresholds[token_idx]:
                        token_list.append(prob_idx)

                if len(token_list) > 0:
                    probable_tokens.append(token_list)

            # Check if EOS_TOKEN appears in any of the token lists
            has_eos = False
            for tokens in probable_tokens:
                if EOS_TOKEN in tokens:
                    has_eos = True
                    break

            if has_eos:
                last_probable_tokens = copy.deepcopy(probable_tokens)
            else:
                break

            # Get index of last element whose list contains EOS_TOKEN
            last_eos_idx = -1
            for i, tokens in enumerate(probable_tokens):
                if EOS_TOKEN in tokens:
                    last_eos_idx = i
            if last_eos_idx != -1:
                probable_tokens = probable_tokens[:last_eos_idx+1]

            # Count unique permutations up to first EOS_TOKEN
            num_permutations = 0
            for i in range(len(probable_tokens)):
                # For position i, count permutations that end at i
                if EOS_TOKEN in probable_tokens[i]:
                    perms_to_i = 1
                    for j in range(i):
                        perms_to_i *= len(probable_tokens[j])
                    num_permutations += perms_to_i
            
            # If no EOS found, count all full permutations
            if num_permutations == 0:
                num_permutations = 1
                for t in probable_tokens:
                    num_permutations *= len(t)

            #print("==> There is a total of %i permutations with current thresholding." % (num_permutations))
            # Increment thresholds proportionally to index
            n = len(thresholds)
            if n > 1:
                total_increment = 0.1
                decay = 0.5
                # Create exponentially decaying increments from right to left
                weights = np.array([decay ** (n-1-i) for i in range(n)])
                thresholds += weights * total_increment

        probable_tokens = last_probable_tokens
        #print("probable_tokens = ", probable_tokens)

        def generate_permutations(probable_tokens, max_length, i = 0):

            if i == len(probable_tokens):   # TODO: potentially expand by further decoding?
                return [[EOS_TOKEN]]

            if len(probable_tokens[i]) == 1 and probable_tokens[i][0] == EOS_TOKEN:
                return [[EOS_TOKEN]]

            all_perms = []
            for token in probable_tokens[i]:
                if token == EOS_TOKEN:
                    all_perms.append([EOS_TOKEN])
                else:
                    perms_list = generate_permutations(probable_tokens, max_length, i+1)

                    perms_with_token = []
                    for perm in perms_list:
                        # Prepend token to the list perm
                        perm = [token] + perm
                        perms_with_token.append(perm)
                    
                    all_perms.extend(perms_with_token)
                    
            return all_perms
        
        return generate_permutations(probable_tokens, max_length)

    def calculate_probs(prob_dist, programs):
        programs_with_probs = []
        for prog in programs:
            prob = 1
            for pos_idx, token in enumerate(prog):
                if pos_idx >= len(prob_dist):
                    break

                tmp_prob = prob_dist[pos_idx][token]
                prob = prob + math.log(tmp_prob)

                if token == EOS_TOKEN:
                    break

            programs_with_probs.append((prob, prog))

        return programs_with_probs

    programs = enumerate_sequences(probability_dist, len(arg_max_seq))
    programs_with_probs = calculate_probs(probability_dist, programs)
    
    # Sort programs by probability in decreasing order
    sorted_programs = sorted(programs_with_probs, key=lambda x: x[0], reverse=True)
    print("Len(sorted_programs) = ", len(sorted_programs))

    # run the programs, eval them until you time out of find the right one.
    n = 0

    for prog in sorted_programs:
        if n % 100 == 0:
            if (time.time() - start_time) > time_budget:
                return prog[1], None, None, False
            
        n += 1

        # evaluate the program and stop if it succeeds.
        verbose = False
        
        result, c1, c2 = evaluate_program(prog[1], example_token_seqs, verbose=verbose)
        #print("\tIteration %i: Result: %s" % (n, result))

        # if c1 is not None:
        #     print("\tc1 = %i, c2 = %i" % (c1, c2))

        # if prog[1][:4] == [13, 1, 50, 3]:
        #     print("==> Trying the correct program!")
        #     return prog[1], c1, c2, True

        if result:
            print("Success! Iteration %i, Time elapsed: %.2f" % (n, time.time() - start_time))
            return prog[1], c1, c2, result

        # if n >= 10000:
        #     return prog[1], c1, c2, False

    return prog[1], c1, c2, False