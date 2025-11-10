# Get Started

1. download [dataset](https://www.kaggle.com/datasets/tianmin/lichess-chess-puzzle-dataset) to ./data/ 
2. unzip dataset to filename `./data/lichess-chess-puzzle-transformed.csv`
3. transform dataset with `python data/transform_data.py` (takes about ~20min)
4. train/test/save a model with `python -m chess_ml.train.Immitation` (takes more than ~20min)


# Filestructure

```
.
├── chess_ml
│   ├── data
│   │   └── Puzzles.py
│   ├── env
│   │   ├── Environment.py: env handling board and rewards
│   │   └── Rewards.py: reward functions
│   ├── model
│   │   ├── ChessNN.py: base class with wrappers for RL and legal move masking
│   │   └── FeedForward.py: feed forward implementation of base class
│   ├── test.py
│   └── train
│       ├── Immitation.py: training routine immitation learning 
│       └── Reinforcement.py: training routine reinforcement learning
└── data
    ├── lichess_puzzle_labeled.csv
    ├── lichess_puzzle_transformed.csv
    ├── lichess_transformed.csv
    └── transform_data.py: transforms kaggle dataset to labeled dataset
```

# TODO

- [ ] RL loop 
- [ ] Immitation training
