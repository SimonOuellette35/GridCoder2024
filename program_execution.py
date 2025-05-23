import json
import random
import sys
import Hodel_primitives_atomicV3 as Hodel_atomic
import search.program_interpreter_V3 as pi
import numpy as np
import ARC_gym.utils.visualization as viz
import ARC_gym.utils.tokenization as tok


def load_random_object(file_path):
    """
    Load a random object from a JSON file containing a list of objects.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: A randomly selected object from the JSON list
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file isn't valid JSON
        ValueError: If the JSON file doesn't contain a list
        IndexError: If the list is empty
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError(f"The JSON file {file_path} does not contain a list")
            
        if not data:
            raise IndexError(f"The list in {file_path} is empty")
            
        return random.choice(data)
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not valid JSON")
        sys.exit(1)
    except (ValueError, IndexError) as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

NUM_SPECIAL_TOKENS = 4
def convert_to_label_seq(program):
    label_seq = []
    for token in program:
        if token == 'NEW_LEVEL':
            token_idx = 1
        elif token == 'IDENTITY':
            token_idx = 2
        elif token == 'EOS':
            token_idx = 3
        else:
            token_idx = Hodel_atomic.prim_indices[token] + NUM_SPECIAL_TOKENS

        label_seq.append(token_idx)

    return label_seq

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python program_execution.py <json_file_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    sample = load_random_object(file_path)
    
    tmp_grid_sequence = sample['input_sequence']
    
    input_grid = tmp_grid_sequence[:931]
    output_grid = tmp_grid_sequence[931:]    
    label_seq = convert_to_label_seq(sample['prog'])

    print("Program: ", sample['prog'])

    # TODO: execute the program to get the output grid
    input_grid_cells = tok.detokenize_grid_unpadded(input_grid)
    output_grid_cells = tok.detokenize_grid_unpadded(output_grid)

    input_DSL_grid = Hodel_atomic.Grid(input_grid_cells)

    # run the program interpreter on the task
    program_tree = pi.generate_syntax_trees(np.array(label_seq), Hodel_atomic)
    task_desc = pi.write_program(program_tree, np.array(label_seq), Hodel_atomic)
    program_func = pi.compile_program(task_desc, Hodel_atomic.semantics)

    pred_output = program_func(input_DSL_grid)

    if isinstance(pred_output, list):
        pred_output = pred_output[0]

    viz.draw_grid_triple(input_grid_cells, pred_output.get_shifted_cells(), output_grid_cells)