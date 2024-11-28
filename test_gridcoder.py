import argparse
import os
import sys
import torch
import json
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Command line arguments for GridCoder")

    parser.add_argument("--task", type=str, help="Task to load for eval or training ARC set: the name of the filename for the task")
    parser.add_argument("--dataset", type=str, default="eval", help="Task to load ('synthetic' for synthetically generated tasks, 'eval' for ARC evaluation dataset, 'train' for ARC training dataset)")
    parser.add_argument("--heuristic", type=str, default="Transformer", help="possible choices: [Transformer, Pixelwise]")
    parser.add_argument("--time_budget", type=int, default=300, help="Time budget per task in seconds")
     
    args = parser.parse_args()
    return args

args = parse_arguments()

if args.task == 'Kaggle':
    os.chdir('/kaggle/working/GridCoder/')
    sys.path.append('/kaggle/working/GridCoder/')
    print("==> Current working directory: ", os.getcwd())

from ARC_gym.arc_evaluation_dataset import ARCEvaluationDataset
from datasets.similarity_dataset_p_star_atomic import ARCInspiredHodelSimilarity
from torch.utils.data import DataLoader
from ARC_gym.utils.batching import make_gridcoder_batch
import ARC_gym.utils.tokenization as tok
import Hodel_primitives_atomicV3 as Hodel_atomic
import utils.sequence_utils as seq_utils
from model.LVM import LVM
#import search.p_star as p_star
import search.p_star_superposition as p_star
import utils.grid_utils as g

# ================================================================== Dataset ==================================================================

if args.task == 'Kaggle':
    # Load and parse the JSON file
    with open('/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json', 'r') as f:
    #with open('/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json', 'r') as f:
        data = json.load(f)
    
        # Store each task object in a list
        test_tasks = []
        for task_id, task_data in data.items():
            test_tasks.append({task_id: task_data})

        print("Loaded %i test tasks!" % len(test_tasks))
else:
    if args.dataset == 'synthetic':
        ds = ARCInspiredHodelSimilarity()
    elif args.dataset == 'eval':
        print("Testing on ARC eval dataset.")
        ds = ARCEvaluationDataset()
        eval_loader = DataLoader(ds,
                                batch_size=1,
                                collate_fn=lambda x: make_gridcoder_batch(x),
                                shuffle=False)
    
# TODO: implement ARC training dataset as well

# ================================================================== Heuristic ==================================================================

NUM_SPECIAL_TOKENS = 4
DSL_size = len(Hodel_atomic.semantics) + NUM_SPECIAL_TOKENS             # + 4 for the IDENTITY, NEW_LEVEL, EOS, and PADDING special tokens
DSL_size2 = max(list(Hodel_atomic.prim_indices.values())) + NUM_SPECIAL_TOKENS + 1
MAX_SEQ_LENGTH = 40

if DSL_size != DSL_size2:
    print("ERROR: there are %i entries in semantics, but the maximum prim_indices index is %i" % (
        DSL_size, max(list(Hodel_atomic.prim_indices.values()))
    ))
    exit(-1)

print("DSL size = %i" % DSL_size)

input_vocab_size = 13
output_vocab_size = DSL_size

# Load the model
EMBED_DIM = 512
model = LVM(input_vocab_size, output_vocab_size, emb_dim=EMBED_DIM, max_seq_length=MAX_SEQ_LENGTH)
if args.task == 'Kaggle':
    model.load_state_dict(torch.load('/kaggle/working/GridCoder/model_full.pth'))
else:
    model.load_state_dict(torch.load('model_full.pth'))

# Ensure the model is on the correct device
model.to('cuda')

model.train()      # For some reason if performs worse with model.eval()!
#model.eval()

# ================================================================== Tasks ==================================================================

device = 'cuda'

def save_submissions(submissions):
    # Convert submissions dictionary to JSON-serializable format
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj

    json_safe_submissions = convert_to_json_serializable(submissions)
    
    # Save submissions dictionary to JSON file
    with open('/kaggle/working/submission.json', 'w') as f:
        json.dump(json_safe_submissions, f)

def process_task(model, X_tensor, Y_tensor, X_token_seq, Y_token_seq):
    if args.heuristic == 'Transformer':
        max_iterations = 10000
        max_depth = 40

        try:
            result, c1, c2, success = p_star.search(model, (X_tensor, Y_tensor), (X_token_seq, Y_token_seq), args.time_budget, max_iterations, max_depth)
            if success:
                print("Success! Program found: ", result)
        except:
            import traceback
            print("Exception occurred during search:")
            traceback.print_exc()
            return None, None, None, False


    return result, c1, c2, success

def preprocess_kaggle_data(examples):

    def process_grid_pair(input_grid, output_grid):
        # Convert input_grid and output_grid lists into tuples of tuples
        cells_x = tuple(tuple(row) for row in input_grid)
        cells_y = tuple(tuple(row) for row in output_grid)

        support_x = tok.tokenize_grid(cells_x, max_length=931)
        support_y = tok.tokenize_grid(cells_y, max_length=931)

        GRID_LENGTH = (31 * 30) + 1              # 931
        x_token_seq = support_x[:GRID_LENGTH]
        y_token_seq = support_y[:GRID_LENGTH]

        x_tensor = torch.unsqueeze(torch.from_numpy(g.one_hot_encode(g.gridify(x_token_seq), input_vocab_size)).to(device).float(), dim=0)
        y_tensor = torch.unsqueeze(torch.from_numpy(g.one_hot_encode(g.gridify(y_token_seq), input_vocab_size)).to(device).float(), dim=0)

        return x_tensor, x_token_seq, y_tensor, y_token_seq
    
    X_tensors = []
    Y_tensors = []
    X_token_seqs = []
    Y_token_seqs = []

    for example in examples:
        input_grid = example['input']

        if 'output' in example:
            output_grid = example['output']
        else:
            output_grid = input_grid

        x_tensor, x_token_seq, y_tensor, y_token_seq = process_grid_pair(input_grid, output_grid)

        X_tensors.append(x_tensor)
        X_token_seqs.append(x_token_seq)
        Y_tensors.append(y_tensor)
        Y_token_seqs.append(y_token_seq)

    return X_tensors, Y_tensors, X_token_seqs, Y_token_seqs

if args.task == 'Kaggle':

    # Expected submission json format:
    # {"00576224": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
    #  "009d5c81": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
    #  "12997ef3": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]},
    #               {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
    #  ...
    # }
    submissions = {}
    print("Found %i test tasks!" % len(test_tasks))
    task_counter = 0
    for test_task in test_tasks:
        key = list(test_task.keys())[0]
        task_counter += 1
        #print(key)
        #if key == '32e9702f':
        print("Processing task #%i: %s" % (task_counter,  key))
        train_examples = test_task[key]['train']

        X_tensor, Y_tensor, X_token_seq, Y_token_seq = preprocess_kaggle_data(train_examples)
        
        solution, c1, c2, success = process_task(model, X_tensor, Y_tensor, X_token_seq, Y_token_seq)

        if success:
            print("==> Success ! Solution: ", solution)
        else:
            print("Failed to find a solution.")

        test_examples = test_task[key]['test']
        _, _, X_token_seq, Y_token_seq = preprocess_kaggle_data(test_examples)

        submission_list = []
        print("There are %i test examples!" % (len(test_examples)))

        for test_idx in range(len(test_examples)):

            if solution is None:
                result = X_token_seq[test_idx]

                result_grid = tok.detokenize_grid_unpadded(result)
                # Convert tuple of tuples to list of lists
                result_list = []
                for row in result_grid:
                    result_list.append(list(row))
            else:
                try:
                    # handle color_change case
                    if c1 is not None:
                        result, _, _ = p_star.get_prediction(solution, [X_token_seq[test_idx]], c1, c2)
                    else:
                        result, _, _ = p_star.get_prediction(solution, [X_token_seq[test_idx]])

                    # Convert tuple of tuples to list of lists
                    result_list = []
                    for row in result[0].cells:
                        result_list.append(list(row))
                except:
                    print("Exception occurred during prediction:")
                    import traceback
                    traceback.print_exc()
                    result = X_token_seq[test_idx]

                    result_grid = tok.detokenize_grid_unpadded(result)
                    # Convert tuple of tuples to list of lists
                    result_list = []
                    for row in result_grid:
                        result_list.append(list(row))

            attempt_json = {
                'attempt_1': result_list,
                'attempt_2': result_list
            }
            submission_list.append(attempt_json)

        print("Adding key %s to submissions." % key)
        submissions[key] = submission_list

        save_submissions(submissions)

else:
    for task_idx, eval_task in enumerate(eval_loader):
        
        if eval_task['task_desc'][0] == args.task:
            print("Task description: ", eval_task['task_desc'])

            X_tensor = []
            Y_tensor = []

            X_token_seq = []
            Y_token_seq = []

            # input, output pairs must be provided
            support_x = eval_task['xs'][0]
            support_y = eval_task['ys'][0]

            tmp_input_seq = seq_utils.gen_in_context_seq_full(support_x, support_y)

            for k_idx in range(tmp_input_seq.shape[0]):
                input_grid = tmp_input_seq[k_idx, :931]
                output_grid = tmp_input_seq[k_idx, 931:]

                X_token_seq.append(input_grid)
                Y_token_seq.append(output_grid)

                input1 = torch.unsqueeze(torch.from_numpy(g.one_hot_encode(g.gridify(input_grid), input_vocab_size)).to(device).float(), dim=0)
                input2 = torch.unsqueeze(torch.from_numpy(g.one_hot_encode(g.gridify(output_grid), input_vocab_size)).to(device).float(), dim=0)

                X_tensor.append(input1)
                Y_tensor.append(input2)

            process_task(model, X_tensor, Y_tensor, X_token_seq, Y_token_seq)

