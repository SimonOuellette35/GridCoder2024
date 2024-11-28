This is the code for the GridCoder algorithm mentioned in my ARC Prize 2024 paper submission:
- [Towards Efficient Neurally-Guided Program Induction for ARC-AGI](https://arxiv.org/abs/2411.17708)

# Setup
- First, clone [ARC_gym](https://github.com/SimonOuellette35/ARC_gym/)
- From the ARC_gym local repo folder, pip -e install .
- Then, clone [ARC-AGI](https://github.com/fchollet/ARC-AGI) into the GridCoder2024 root folder, because you will need ARC-AGI grid examples for training/validation data generation.
- Rename the ARC-AGI folder to ARC
- You can get a pre-trained from my Kaggle model: [https://www.kaggle.com/models/simonouellette/gridcoder-2024/](https://www.kaggle.com/models/simonouellette/gridcoder-2024/) -- download the model_full.pth file under Kaggle_code/

# Usage
- Training the model: train_full.py (model/LVM.py is the model architecture itself).
- Testing the solution: test_gridcoder.py (see Kaggle notebook for details on using it in the context of a Kaggle competition).
- Generating training data: generate_training_data_full.py
- The training data generation code is under the datasets/ folder.
- The main search algorithm itself is search/p_star_superposition.py (Note: I originally called the algorithm P* as a play on A*, but in the paper I refer to this as GridCoder).
- P_star.py is "GridCoder cond." in the paper.
- P_star_muzero.py is the MCTS-based variant that was evaluated in the paper.
