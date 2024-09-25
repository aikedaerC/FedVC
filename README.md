<h1 align="center">FedVC: Virtual Clients for Federated Autonomous Driving with Imbalanced Label Distribution</h1>


We are in an early-release beta. Expect some adventures and rough edges.

## Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)


## Introduction

Federated learning in autonomous driving ensures the privacy of individual data silos during collaborative training without sharing raw local data. Each data silo typically suffers from imbalanced label distribution, and the resulting federated learning model is often safety-critical.
Most existing solutions for addressing data imbalance in federated learning focus on classification tasks. However, the research on prediction tasks specific to autonomous driving, such as steering angle and trajectory prediction, remains largely unexplored.
In this work, we have proposed a novel peer-to-peer federated learning framework, abbreviated as FedVC, which introduces virtual clients to make the collaborative training process to perceive the global data distribution. It creates a dynamic dataset by selecting the proper local data portions which should be utilized in a single training round and manages the execution time of backpropagation process for each virtual client. Specifically, without sharing any synthetic dataset, the global view is just constructed from a few metadata of each data silo. This design preserves data privacy while mitigating the issue of data imbalance in autonomous driving.
The experimental results show that FedVC outperforms classical FedAvg and the most recent methods at three steering angle prediction datasets with different levels of imbalanced label distribution. 

## Motivation

The approach to dividing data in federated learning is diverse. Assuming that there is an optimal data division, $\mathcal{D}^{\star}$, which can achieve the best convergence results (i.e., the highest accuracy and best generalization), the goal is to approximate the effect of $\mathcal{D}^{\star}$ in all client training processes. It can be considered that the training process of $\mathcal{D}^{\star}$ achieving good convergence is static, while in another division $\mathcal{D}$, we can try dynamic training methods to approximate the effect of $\mathcal{D}^{\star}$. Further discussion is needed to fully understand dynamic training and its relationship with the $\mathcal{D}^{\star}$ algorithm.


