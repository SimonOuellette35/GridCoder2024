This is the code for the GridCoder algorithm mentioned in my ARC Prize 2024 paper submission:
- [Towards Efficient Neurally-Guided Program Induction for ARC-AGI](https://arxiv.org/abs/2411.17708)

# Setup
- First, clone [ARC_gym](https://github.com/SimonOuellette35/ARC_gym/)
- From the ARC_gym local repo folder, pip install -e .
- Then, clone [ARC-AGI](https://github.com/fchollet/ARC-AGI) into the GridCoder2024 root folder, because you will need ARC-AGI grid examples for training/validation data generation.
- Rename the ARC-AGI folder to ARC
- You can get pre-trained weights from my Kaggle model: [https://www.kaggle.com/models/simonouellette/gridcoder-2024/](https://www.kaggle.com/models/simonouellette/gridcoder-2024/) -- download the model_full.pth file under Kaggle_code/

# General Usage
- Training the model: train_full.py (model/LVM.py is the model architecture itself).
- Testing the solution: test_gridcoder.py (see Kaggle notebook for details on using it in the context of a Kaggle competition).
- Generating training data: generate_training_data_full.py
- The training data generation code is under the datasets/ folder.
- The main search algorithm itself is search/p_star_superposition.py (Note: I originally called the algorithm P* as a play on A*, but in the paper I refer to this as GridCoder).
- P_star.py is "GridCoder cond." in the paper.
- P_star_muzero.py is the MCTS-based variant that was evaluated in the paper.

# Reproducing the paper experiments

Assuming you have downloaded the pretrained model from Kaggle, you can use the script test_gridcoder.py to reproduce the experiment results.

For the sake of time efficiency, I only run the script on the individual tasks that are theoretically solvable in the given DSL. The script uses the *--task (task ID)* command-line argument to specify which eval task to attempt to solve.
In other words, it does not loop over all the eval tasks. In fact, if you don't specify that argument, the script will silently fail without attempting to solve any task (this is obviously not ideal from a user-friendliness perspective, but keep in mind this is raw "proof-of-concept" code written in a hurry to meet the competition deadline).

Example command-line usage:

$ python test_gridcoder.py --task 1990f7a8.json

The tasks that are potentially solvable are as follows:

**[Trivial "objectness" type of tasks]**
- 1990f7a8.json
- 67636eac.json
- 25094a63.json
- 45737921.json
- 73182012.json
- 423a55dc.json
- 7c9b52a0.json
- 1c56ad9f.json
- 0a1d4ef5.json
- af24b4cc.json
- 0bb8deee.json
- 8fbca751.json
- 1c0d0a4b.json

**[Object Selector type of tasks]**
- 1a6449f1.json
- 2c0b0aff.json
- d56f2372.json
- 73ccf9c2.json
- 3194b014.json
- 54db823b.json

**[Misc. objectness type of tasks]**
- e7dd8335.json
- 7ee1c6ea.json

**[split-merge tasks]**
- 195ba7dc.json
- 5d2a5c43.json
- 3d31c5b3.json
- e99362f0.json
- e133d23d.json
- e345f17b.json
- ea9794b1.json
- 31d5ba1a.json
- d19f7514.json
- 66f2d22f.json
- 506d28a5.json
- 6a11f6da.json
- 34b99a2b.json
- 281123b4.json
- 0c9aba6e.json

**[tiling tasks]**
- 59341089.json 
- c48954c1.json 
- 7953d61e.json 
- 0c786b71.json 
- 833dafe3.json 
- 00576224.json 
- 48131b3c.json
- ed98d772.json 
- bc4146bd.json 

**[n-deep tasks]**
- 66e6c45b.json 
- 32e9702f.json
- 60c09cac.json
- e633a9e5.json

Note that if you want to try the alternative search algorithm discussed in the paper, you can change the line 35 import in the test_gridcoder.py script to search.p_star as p_star ("GridCoder cond") or p_star_muzero.py for the MCTS version.
