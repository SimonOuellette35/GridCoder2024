from datasets.similarity_dataset_p_star_atomic import ARCInspiredHodelSimilarity as ARCInspiredHodelSimilarityAtomic
import numpy as np
from tqdm import tqdm
import utils.sequence_utils as seq_utils
import ARC_gym.utils.tokenization as tok
import ARC_gym.utils.visualization as viz
import utils.heuristics as h
import search.program_interpreter_V3 as pi
import torch.nn.functional as F
import csv

DSL = 'atomic'

training_dataset = ARCInspiredHodelSimilarityAtomic()
validation_dataset = ARCInspiredHodelSimilarityAtomic(validation=True)
MAX_SEQ_LENGTH = 40
training_filename = 'training_data_atomic.csv'
validation_filename = 'validation_data_atomic.csv'

# In dataset, each iteration or entry is a task. Each task contains multiple distinct samples (each node along the
#  tree is an opportunity for a sample).
def generate_data_batch(dataset, N):

    samples = []

    for next_idx in range(N):
        S = dataset[next_idx]

        desc = S['task_desc']

        #print("==> Generated task description: ", desc)
        start_grids = S['xs']
        end_grids = S['ys']
        label_seq = S['label_seq']

        samples.append((start_grids, end_grids, label_seq, desc))

    return samples, desc

# Define new vocabulary sizes
input_vocab_size = 13           # 10 colors + 3 special tokens

def preprocess_data(samples):
    label_seqs = []
    input_seqs = []
    EOS_TOKEN = 2

    for idx in range(len(samples)):
        sample = samples[idx]
        xs, ys, label_seq, desc = sample

        # print("==> Generated start grids: ", xs)
        # print("==> Generated end grids: ", ys)
        # print("==> Generated label sequence: ", label_seq)
        # print("==> Generated task description: ", desc)

        tmp_input_seq = seq_utils.gen_in_context_seq_full(xs, ys)
        input_seqs.append(tmp_input_seq)
        
        # Add EOS token to the label sequence
        label_seq_with_eos = label_seq + [EOS_TOKEN]
        label_seqs.append(label_seq_with_eos)

    # Pad label sequences
    padded_label_seqs = []
    for seq in label_seqs:
        seq = list(np.array(seq) + 1)
        padded_seq = seq + [0] * (MAX_SEQ_LENGTH - len(seq))
        padded_label_seqs.append(padded_seq)

    return np.array(input_seqs), np.array(padded_label_seqs)

def one_hot_encode_batch(grid_batch):
    output_batch = np.zeros((grid_batch.shape[0], input_vocab_size, 30, 30))
    for b_idx in range(grid_batch.shape[0]):
        for i in range(grid_batch.shape[1]):
            for j in range(grid_batch.shape[2]):
                token = int(grid_batch[b_idx, i, j])
                output_batch[b_idx, token, i, j] = 1

    return output_batch

def one_hot_encode(grid_example):

    output_example = np.zeros((input_vocab_size, 30, 30))
    for i in range(grid_example.shape[0]):
        for j in range(grid_example.shape[1]):
            token = int(grid_example[i, j])
            output_example[token, i, j] = 1

    return output_example

def gridify_batch(token_seq_batch):
    output_batch = np.zeros((token_seq_batch.shape[0], 30, 30))

    for b_idx, token_seq in enumerate(token_seq_batch):
        x = 0
        y = 1
        for idx in range(len(token_seq)):
            if token_seq[idx] == 0:
                continue

            if token_seq[idx] >= 3:
                output_batch[b_idx, 30 - y, x] = token_seq[idx]

            x += 1

            if token_seq[idx] == 2:
                break

            if token_seq[idx] == 1:
                y += 1
                x = 0

    return output_batch

def gridify(token_seq):
    output_grid = np.zeros((30, 30))

    x = 0
    y = 1
    for idx in range(len(token_seq)):
        if token_seq[idx] == 0:
            continue

        if token_seq[idx] >= 3:
            output_grid[30 - y][x] = token_seq[idx]

        x += 1

        if token_seq[idx] == 2:
            return output_grid

        if token_seq[idx] == 1:
            y += 1
            x = 0

    return output_grid


print("Generating training data...")

# Training data generation loop
# For DSL 2:
N = 1000000

with open(training_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['input_grid', 'output_grid', 'target_seq'])  # Write header

    for _ in tqdm(range(N)):
        current_samples, task_desc = generate_data_batch(training_dataset, 1)
        input_seqs, label_seqs = preprocess_data(current_samples)

        target_seq = label_seqs[0]
        print("==> task_desc = ", task_desc)
        print("\ttarget_seq = ", target_seq)
        target_list = target_seq.tolist()

        for k_idx in range(input_seqs.shape[1]):
            input_grid = input_seqs[0, k_idx, :931]
            output_grid = input_seqs[0, k_idx, 931:]

            inp = tok.detokenize_grid_unpadded(input_grid)
            outp = tok.detokenize_grid_unpadded(output_grid)
            viz.draw_grid_pair(inp, outp)

            # Convert numpy arrays to lists for CSV writing
            input_list = input_grid.flatten().tolist()
            output_list = output_grid.flatten().tolist()

            # Write the data to the CSV file
            writer.writerow([input_list, output_list, target_list])

print("==> Training data task ratios: ", training_dataset.task_ratios)

print("Training data generation complete.")
print("Generating validation data...")

# Validation data generation loop
# For DSL 2:
N = 50000

with open(validation_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['input_grid', 'output_grid', 'target_seq'])  # Write header

    for _ in tqdm(range(N)):
        current_samples, task_desc = generate_data_batch(validation_dataset, 1)
        input_seqs, label_seqs = preprocess_data(current_samples)

        # For now only use the first example...
        target_seq = label_seqs[0]
        target_list = target_seq.tolist()

        for k_idx in range(input_seqs.shape[1]):
            input_grid = input_seqs[0, k_idx, :931]
            output_grid = input_seqs[0, k_idx, 931:]
            
            # Convert numpy arrays to lists for CSV writing
            input_list = input_grid.flatten().tolist()
            output_list = output_grid.flatten().tolist()
            
            # Write the data to the CSV file
            writer.writerow([input_list, output_list, target_list])

print("==> Validation data task ratios: ", validation_dataset.task_ratios)