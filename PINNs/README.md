# Physics Informed Neural Networks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heitornolla/Physics-Inspired-ML/blob/main/PINNs/notebook/pinn_for_ball_trajectory.ipynb)

This code is written based on VizuaraAI's PINN tutorial. The goal of this code is to model the trajectory of a ball with a Nueral Network and, mainly, to understand how PINNs work.

The main idea of PINNs is that, aside from the standard data-based losses (MSE, MAE) we have additional loss terms based on physics-constraints in our problem. In this case, for example, we know that the trajectory of a ball is meant to obey a certain equation. As such, we can enforce our neural network's predictions to also obey this equation through an additional loss term.

However, few problems perfectly fit physics constraints. Aside from different behaviors due to an initial stage (which in this case would be the height where the ball is thrown), we may have external sources of noise which are not modelled (e.g. ball format, shape, wind, strength of thrower...). As such, the beauty in this example comes from setting good lambda values so that our model does not overfit neither to the data and to the ODE defined.

Experiment with the code! Change the lambda parameters in the losses, the amount of noise in the data, the dataset generated, to see how it affects the result.
