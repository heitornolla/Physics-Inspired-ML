# Universal Differential Equations (UDE) for Damped Spring-Mass

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heitornolla/Physics-Inspired-ML/blob/main/UDEs/notebook/UDE_problem_spring_mass.ipynb)

A Universal Differential Equation is a framework where part of the differential equation is defined by known physical laws, and other parts are represented by universal approximators, such as Neural Networks.

Mathematically:
$$\frac{du}{dt} = f(u, t, \theta) + NN(u, t, \phi)$$
where $f$ is the known physics and $NN$ is the neural network learning the unknown dynamics.

## The Specific Problem

We simulate a classic damped spring-mass system:
$$m \ddot{x} + c \dot{x} + kx = 0$$

In this implementation:

1. We assume the mass ($m$) and spring constant ($k$) are **known**.
2. We assume the damping force ($c \dot{x}$) is **unknown**.
3. A Neural Network is embedded inside the ODE solver to learn the damping behavior based purely on noisy observations of displacement and velocity.
