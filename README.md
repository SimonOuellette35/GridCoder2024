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

**[Trivial "objectness" tasks] (8/13)**
- 1990f7a8.json: SUCCESS
- 67636eac.json: SUCCESS
- 25094a63.json: FAILURE
- 45737921.json: SUCCESS
- 73182012.json: SUCCESS
- 423a55dc.json: SUCCESS
- 7c9b52a0.json: SUCCESS
- 1c56ad9f.json: SUCCESS
- 0a1d4ef5.json: FAILURE
- af24b4cc.json: FAILURE
- 0bb8deee.json: SUCCESS
- 8fbca751.json: FAILURE
- 1c0d0a4b.json: FAILURE

**[Object Selector] (1/6)**
- 1a6449f1.json: SUCCESS
- 2c0b0aff.json: FAILURE
- d56f2372.json: FAILURE
- 73ccf9c2.json: FAILURE
- 3194b014.json: FAILURE
- 54db823b.json: FAILURE

**[Misc. objectness] (1/2)**
- e7dd8335.json: SUCCESS
- 7ee1c6ea.json: FAILURE

**[split-merge tasks] (15/15)**
- 195ba7dc.json: SUCCESS
- 5d2a5c43.json: SUCCESS
- 3d31c5b3.json: SUCCESS
- e99362f0.json: SUCCESS
- e133d23d.json: SUCCESS
- e345f17b.json: SUCCESS
- ea9794b1.json: SUCCESS
- 31d5ba1a.json: SUCCESS
- d19f7514.json: SUCCESS
- 66f2d22f.json: SUCCESS
- 506d28a5.json: SUCCESS
- 6a11f6da.json: SUCCESS
- 34b99a2b.json: SUCCESS
- 281123b4.json: SUCCESS
- 0c9aba6e.json: SUCCESS

**[tiling tasks] (5/9)**
- 59341089.json: SUCCESS
- c48954c1.json: FAILURE		* Solution actually in non-zero probability space, but search times out before reaching it.
- 7953d61e.json: SUCCESS
- 0c786b71.json: SUCCESS
- 833dafe3.json: SUCCESS
- 00576224.json: FAILURE		* Solution actually in non-zero probability space, but search times out before reaching it.
- 48131b3c.json: FAILURE
- ed98d772.json: SUCCESS
- bc4146bd.json: FAILURE

**[n-deep tasks] (3/4)**
- 66e6c45b.json: SUCCESS
- 32e9702f.json: SUCCESS
- 60c09cac.json: SUCCESS
- e633a9e5.json: FAILURE

In most cases except the two failures identified as such, the failures are due to model inaccuracy rather that search inefficiency. That is, in most failure cases the solution is predicted to have a near-zero probability according to the model. Presumably a better model architecture, better training data and/or more training can resolve these issues. Note that also that there is some randomness in results due to the "probability bootstrapping" phase which involves some random sampling -- hence the deterministic seed to ensure reproducibility.

Note that if you want to try the alternative search algorithm discussed in the paper, you can change the line 35 import in the test_gridcoder.py script to search.p_star as p_star ("GridCoder cond") or p_star_muzero.py for the MCTS version.
