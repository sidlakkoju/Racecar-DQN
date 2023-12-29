# Racecar DQN!!! 🏎️💨


## Click to watch the Demo!!!
[![alt text](https://raw.githubusercontent.com/sidlakkoju/Racecar-DQN/main/test_files/thumbnail.jpg)](https://www.youtube.com/watch?v=z0XigG7Qflo)
^^Youtube^^

## Training

Training parameters are in `parameters.py`. Adjust as needed.

Run `train_model.py` to train the model.


## Trained Models

Found in the `models` directory. 
- `model_weights_mps.pth` (**Best Performer**) was trained on apple silicon with MPS. 
- `model_weights_cuda.pth` was trained on an Nvidia GPU with CUDA. Cuda trains much faster than cpu and mps.

Use `test_model.py` with appropriate weights path in `agent.load_model()` to test the model.



## Resources Used
- Good introduction to RL and DQN's: [Professor Nguyen Slides](https://docs.google.com/presentation/d/1QgRoOgJw7rv9_xBMv9nscA57Pv8pOysDDSerHF7P7Nk/edit#slide=id.ga283940d24_0_188)

- [Pytorch Cartpole DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

- Original Deepmind DQN Atari paper: [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)


## Drive Yourself 🏎️💨

- Run `manual_drive.py` to drive the car yourself using the arrow keys. Can you beat the DQN? 🤔
