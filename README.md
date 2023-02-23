# Official implementation of Bilinear Value Networks

[Paper link](https://openreview.net/pdf?id=LedObtLmCjS)

Abstract: Universal value functions are a core component of off-policy multi-goal reinforcement learning. 
The de-facto paradigm is to approximate Q(s, a, g) using monolithic neural networks which lack inductive biases to produce complex interactions between the state s and the goal g. In this work, we propose a bilinear decomposition that represents the Q-value via a low-rank approximation in the form of a dot product between two vector fields. The first vector field, f(s, a), captures the environment's local dynamics at the state s; whereas the second component, ϕ(s, g), captures the global relationship between the current state and the goal.
We show that our bilinear decomposition scheme improves sample efficiency over the original monolithic value approximators, and transfer better to unseen goals. We demonstrate significant learning speed-up over a variety of tasks on a simulated robot arm, and the challenging task of dexterous manipulation with a Shadow hand.

# Installation (conda)
```
conda create -n bvn python=3.8
pip3 install -r requirements.txt
```

# Fetch
See `fetch`

# ShadowHand
See `shadow_hand`
