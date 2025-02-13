import numpy as np
import Hodel_primitives_atomic as hp
import search.program_interpreter as pi

class ManualHeuristic:

    def __init__(self, prim_dict):
        self.prim_dict = prim_dict

    def get_binary_similarity(self, grid1, grid2):
        def recode(flat_list):
            for idx, v in enumerate(flat_list):
                if v > 3:
                    flat_list[idx] = 4

        flat_grid1 = np.copy(np.reshape(grid1, (-1,)))
        flat_grid2 = np.copy(np.reshape(grid2, (-1,)))
        recode(flat_grid1)
        recode(flat_grid2)

        count_correct = np.count_nonzero(flat_grid1 == flat_grid2)
        return count_correct / float(len(flat_grid1))

    # input_grid_set and output_gridset have shape (num_examples, 30, 30)
    def forward(self, input_grid_set, output_grid_set, tgt_sequence):
        
        # the input will be tensors, so we need to convert them to numpy arrays
        input_grid_set = input_grid_set.cpu().numpy()
        output_grid_set = output_grid_set.cpu().numpy()
        tgt_sequence = tgt_sequence.cpu().numpy()

        # intermediate_grid_sets has shape [num_examples, num_available_grids, 30, 30]
        intermediate_grid_sets = pi.apply_partial_prog(tgt_sequence, input_grid_set)
        
        primitive_scores = []
        
        for prim_name, lambda_func in self.prim_dict.items():
            num_args = hp.get_num_args(prim_name)
            
            best_primitive_score = 0
            
            for i in range(len(intermediate_grid_sets[0]) - num_args + 1):
                if num_args == 1:
                    input_grid_set = intermediate_grid_sets[:, i:i+1]
                elif num_args == 2:
                    input_grid_set = intermediate_grid_sets[:, i:i+2]
                else:
                    raise ValueError(f"Unsupported number of arguments for primitive {prim_name}")
                
                similarities = []
                for k in range(len(input_grid_set)):
                    if num_args == 1:
                        transformed_grid = lambda_func(input_grid_set[k][0])
                    else:
                        transformed_grid = lambda_func(input_grid_set[k][0], input_grid_set[k][1])
                    
                    similarity = self.get_binary_similarity(transformed_grid, output_grid_set[k])
                    similarities.append(similarity)
                
                primitive_score = np.median(similarities)
                best_primitive_score = max(best_primitive_score, primitive_score)
            
            primitive_scores.append(best_primitive_score)
        
        total_score = sum(primitive_scores)
        normalized_scores = [score / total_score for score in primitive_scores] if total_score > 0 else [1.0 / len(primitive_scores)] * len(primitive_scores)
        
        return normalized_scores
    def get_similarity(self, grid1, grid2, verbose=False):
        flat_grid1 = np.reshape(grid1, (-1,))
        flat_grid2 = np.reshape(grid2, (-1,))

        max_len = np.where(flat_grid2 == 0)[0][0] if 0 in flat_grid2 else len(flat_grid2)
        count_correct = np.count_nonzero((flat_grid1 == flat_grid2)[:max_len])

        RGB_similarity = count_correct / max_len
        return RGB_similarity
        #bin_similarity = self.get_binary_similarity(grid1, grid2)

        #return (RGB_similarity + bin_similarity) / 2.

    def get_batched_similarity(self, grid_batch1, grid_batch2):
        flat_grids1 = np.reshape(grid_batch1, (len(grid_batch1), 1, -1))
        flat_grids2 = np.reshape(grid_batch2, (len(grid_batch2), 1, -1))

        stacked_grids = np.concatenate((flat_grids1, flat_grids2), axis=1)
        count_correct = np.count_nonzero(stacked_grids[:, 0, :] == stacked_grids[:, 1, :], axis=-1)

        return count_correct / float(flat_grids1.shape[-1])
