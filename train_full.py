from torch import nn, optim
import csv
import Hodel_primitives_atomicV3 as Hodel_atomic
import numpy as np
import torch
from tqdm import tqdm
import utils.grid_utils as g
import torch.nn.functional as F
import random
from model.LVM import LVM

# In this version, we're using the full <Start of sentence>, primitive token, <End of sentence>, and <Pad> tokens representation.
VERBOSE = False
RESUME = 0
TRAIN_MODEL = True
DSL = 'atomic'
#DSL = '2deep'

NUM_SPECIAL_TOKENS = 4
print("len(Hodel_atomic.semantics) = ", len(Hodel_atomic.semantics))
print("len(Hodel_atomic.prim_indices) = ", len(Hodel_atomic.prim_indices))
print("max(list(Hodel_atomic.prim_indices.values())) = ", max(list(Hodel_atomic.prim_indices.values())))

DSL_size = len(Hodel_atomic.semantics) + NUM_SPECIAL_TOKENS             # + 4 for the IDENTITY, NEW_LEVEL, EOS, and PADDING special tokens
DSL_size2 = max(list(Hodel_atomic.prim_indices.values())) + NUM_SPECIAL_TOKENS + 1
training_filename = 'training_data_atomic.csv'
validation_filename = 'validation_data_atomic.csv'
MAX_SEQ_LENGTH = 40

if DSL_size != DSL_size2:
    print("ERROR: there are %i entries in semantics, but the maximum prim_indices index is %i" % (
        DSL_size, DSL_size2
    ))
    exit(-1)

print("DSL size = ", DSL_size)

# Define new vocabulary sizes
input_vocab_size = 13           # 10 colors + 3 special tokens
output_vocab_size = DSL_size    # exact number of DSL primitives
device = 'cuda'
batch_size = 500
EMBED_DIM = 512
EOS_TOKEN = 3


def custom_loss_function(pred_logits, target_actions, verbose=False):
    pred_seq = torch.argmax(pred_logits, dim=-1)
    
    if verbose:
        pred_seq_np = pred_seq.cpu().data.numpy()[0]
        target_actions_np = target_actions.cpu().data.numpy()

        target_eos_index = np.where(target_actions_np == 3)[0][0]
        
        print("predicted sequence: ", pred_seq_np[:target_eos_index + 1])
        print("target sequence: ", target_actions_np[:target_eos_index + 1])
    
    # Create a mask for non-zero tokens in target_actions
    mask = (target_actions != 0)

    # Flatten pred_logits and target_actions for easier processing
    pred_logits_flat = pred_logits.view(-1, pred_logits.size(-1))
    target_actions_flat = target_actions.view(-1)
    mask_flat = mask.view(-1)
    
    # Apply the mask to both pred_logits and target_actions
    pred_logits_masked = pred_logits_flat[mask_flat]
    target_actions_masked = target_actions_flat[mask_flat]
    
    # Calculate cross-entropy loss on non-zero tokens
    loss = F.cross_entropy(pred_logits_masked, target_actions_masked)
    
    return loss

best_loss = np.inf

def train():

    print("Training model.")
    
    # Initialize the model
    model = LVM(input_vocab_size, output_vocab_size, EMBED_DIM, MAX_SEQ_LENGTH)

    if RESUME > 0:
        # Load the pre-trained model
        model.load_state_dict(torch.load('model_full.pth'))
        print("Loaded pre-trained model from model_full.pth")

    # Ensure the model is on the correct device
    model.to(device)

    model_size = sum(p.numel() for p in model.parameters())
    print("Model parameter count: ", model_size)

    # Prepare optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.005)

    # Read validation data into memory
    print("Reading validation data into memory...")
    with open(validation_filename, 'r', newline='') as val_csvfile:
        val_reader = csv.reader(val_csvfile)
        next(val_reader)  # Skip the header row
        val_data = list(val_reader)

    print("Starting training...")
    # Training loop
    epochs = 10
    best_loss = np.inf
    with open(training_filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row

        batch = []
        batch_num = 1
        for row in reader:
            batch.append(row)
            if len(batch) == batch_size:
                if batch_num >= RESUME:
                    model.train()
                    train_loss, train_acc = train_batch(batch, model, optimizer, device)
                    batch = []
    
                    val_batch = random.sample(val_data, min(1000, len(val_data)))
    
                    #model.eval()
                    val_loss, val_acc = validate_batch(val_batch, model, device)
    
                    print(f"Batch #{batch_num} ==> Train. Loss: {train_loss}, Val. Loss: {val_loss} (Train. acc = {train_acc}, Val. acc. = {val_acc})")
                    batch_num += 1
    
                    # save best model, with num_args in the filename
                    if TRAIN_MODEL:
                        if val_loss < best_loss:
                            best_loss = val_loss
                            print("Saving best model.")
                            torch.save(model.state_dict(), 'model_full.pth')
                else:
                    batch_num += 1
                    batch = []
                    print("Skipping batch_num %i -- resuming training at %i" % (batch_num, RESUME))

        # Process any remaining rows
        if batch:
            train_batch(batch, model, optimizer, device)

def accuracy(pred_sequence, target_sequence):
    # Create a mask for non-zero tokens in target_sequence
    mask = (target_sequence != 0)
    
    # Apply the mask to both sequences
    pred_sequence_masked = pred_sequence[mask]
    target_sequence_masked = target_sequence[mask]
    
    # Calculate the number of correct predictions
    correct_predictions = (pred_sequence_masked == target_sequence_masked).sum().item()
    
    # Calculate total number of non-zero tokens
    total_tokens = mask.sum().item()
    
    # Calculate accuracy
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
    
    return accuracy

def validate_batch(batch, model, device):
    val_accuracy = 0.
    mean_val_loss = 0

    with torch.no_grad():

        for row in tqdm(batch, 'Validation'):

            input_grid = eval(row[0])
            output_grid = eval(row[1])
            label_seq = eval(row[2])

            # Pad the sequence to a fixed length
            label_seq = label_seq[:MAX_SEQ_LENGTH] + [0] * (MAX_SEQ_LENGTH - len(label_seq))

            # Prepend token 3 (EOS_TOKEN) to label_seq
            shifted_label_seq = [EOS_TOKEN] + label_seq[:-1]  # Remove last element to maintain fixed length
            shifted_label_seq = torch.tensor(shifted_label_seq, dtype=torch.long).to(device)

            input1 = torch.unsqueeze(torch.from_numpy(g.one_hot_encode(g.gridify(input_grid), input_vocab_size)).to(device).float(), dim=0)
            input2 = torch.unsqueeze(torch.from_numpy(g.one_hot_encode(g.gridify(output_grid), input_vocab_size)).to(device).float(), dim=0)
        
            logits = model(input1, input2)

            label_seq = torch.tensor(label_seq, dtype=torch.long).to(device)
            val_loss = custom_loss_function(logits, label_seq, verbose=VERBOSE)

            # Generate pred_sequence from logits
            pred_sequence = torch.argmax(logits, dim=-1)
            
            # Calculate accuracy
            val_accuracy += accuracy(pred_sequence[0].cpu().data.numpy(), label_seq.cpu().data.numpy())
        
            mean_val_loss += val_loss.item()

    mean_val_loss /= float(len(batch))
    val_accuracy /= float(len(batch))

    return mean_val_loss, val_accuracy


def train_batch(batch, model, optimizer, device):

    mean_loss = 0
    optimizer.zero_grad()

    train_accuracy = 0.
    loss = 0

    for row in tqdm(batch, 'Training'):

        input_grid = eval(row[0])
        output_grid = eval(row[1])
        label_seq = eval(row[2])

        # Pad the sequence to a fixed length
        label_seq = label_seq[:MAX_SEQ_LENGTH] + [0] * (MAX_SEQ_LENGTH - len(label_seq))
        # Prepend token 3 (EOS_TOKEN) to label_seq
        shifted_label_seq = [EOS_TOKEN] + label_seq[:-1]  # Remove last element to maintain fixed length
        shifted_label_seq = torch.tensor(shifted_label_seq, dtype=torch.long).to(device)

        input1 = torch.unsqueeze(torch.from_numpy(g.one_hot_encode(g.gridify(input_grid), input_vocab_size)).to(device).float(), dim=0)
        input2 = torch.unsqueeze(torch.from_numpy(g.one_hot_encode(g.gridify(output_grid), input_vocab_size)).to(device).float(), dim=0)
    
        logits = model(input1, input2)

        label_seq = torch.tensor(label_seq, dtype=torch.long).to(device)
        loss = custom_loss_function(logits, label_seq)

        # Generate pred_sequence from logits
        pred_sequence = torch.argmax(logits, dim=-1)
        
        # Calculate accuracy
        train_accuracy += accuracy(pred_sequence[0].cpu().data.numpy(), label_seq.cpu().data.numpy())
       
        mean_loss += loss.item()
        loss /= float(len(batch))
        loss.backward()
        
    optimizer.step()
    mean_loss /= float(len(batch))
    train_accuracy /= float(len(batch))

    return mean_loss, train_accuracy

train()
