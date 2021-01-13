---
title: "Learning to Extrapolate Knowledge: Transductive Few-shot Out-of-Graph Link Prediction (NeurIPS 2020)"
date: 2020-12-06
description: "Summary of the paper: Learning to Extrapolate Knowledge: Transductive Few-shot Out-of-Graph Link Prediction, which is accepted to NeurIPS 2020."
categories:
 - Research
tags: 
    - Research
    - Deep Learning
    - Graph Extrapolation Network
    - Extrapolation
    - Knowledge Graph
    - Meta Learning
    - Few-shot Learning
    - Graph Neural Network
    - Link Prediction
    - Out of Graph
---

# About

* **Title**: Learning to Extrapolate Knowledge: Transductive Few-shot Out-of-Graph Link Prediction.
* **Author**: Jinheon Baek, Dong Bok Lee, and Sung Ju Hwang.
* **Conference**:  In Proceedings of Advances in Neural Information Processing System 33 proceedings (NeurIPS 2020).
* **Paper**: https://arxiv.org/abs/2006.06648
* **Code**: https://github.com/JinheonBaek/GEN

---

# Abstract

Many practical graph problems, such as knowledge graph construction and drug-drug interaction prediction, require to handle multi-relational graphs. However, handling real-world multi-relational graphs with Graph Neural Networks (GNNs) is often challenging due to their evolving nature, where new entities (nodes) can emerge over time. Moreover, newly emerged entities often have few links, which makes the learning even more difficult. Motivated by this challenge, we introduce a realistic problem of few-shot out-of-graph link prediction, where we not only predict the links between the seen and unseen nodes as in a conventional out-of-knowledge link prediction but also between the unseen nodes, with only few edges per node. We tackle this problem with a novel transductive meta-learning framework which we refer to as Graph Extrapolation Networks (GEN). GEN meta-learns both the node embedding network for inductive inference (seen-to-unseen) and the link prediction network for transductive inference (unseen-to-unseen). For transductive link prediction, we further propose a stochastic embedding layer to model uncertainty in the link prediction between unseen entities. We validate our model on multiple benchmark datasets for knowledge graph completion and drug-drug interaction prediction. The results show that our model significantly outperforms relevant baselines for out-of-graph link prediction tasks.

---

# Introduction

## Motivation

![Figure1](/assets/images/GEN/Figure1.png "Figure 1. Predicting missing links." ){: width="50%" height="50%"}

While graphs contain a huge amount of knowledge, they are highly incomplete. Thus, automatic graph completion, which is known as link prediction (See Figure 1), is a practically important problem. However, despite recent successes on inferring missing links by learning embeddings of entities and their relations, the link prediction task for real-world graphs remains challenging (See Figure 2):

* **Evolving nature**: new entities can emerge over time.
* **Long-tail distribution**: most entities have few triplets.

![Figure2](/assets/images/GEN/Figure2.png "Figure 2. (Left): Evolving nature. (Right): Long-tail distribution." ){: width="90%" height="90%"}

## Problem Definition

To deal with challenges on graphs, we propose few-shot out-of-graph (OOG) link prediction (See Figure 3), whose goal is to predict links between seen and unseen, and even among unseen entities (evolving nature), with few links per entity (long-tail).

![Figure3](/assets/images/GEN/Figure3.png "Figure 3. An illustration of Out-of-Graph link prediction for emerging entities. Blue dotted arrows denote inferred relationships between seen and unseen entities, Red dotted arrows denote inferred relationships between unseen entities." ){: width="60%" height="60%"}

---

# Method

## Mete-Learning Framework

To tackle OOG link prediction, we propose a novel meta-learning framework. It meta-learns the node embeddings for unseen entities by simulating the unseen entities during training (See Figure 4, left), which is impossible for conventional learning, and extrapolates learned knowledge to real unseen entities.

![Figure4](/assets/images/GEN/Figure4.png "Figure 4. Mete-learning makes the model generalize on real unseen entities." ){: width="75%" height="75%"}

## Graph Extrapolation Network

We train the proposed Graph Extrapolation Network (GEN) under the our meta-learning framework, with an inductive scheme for predicting seen to unseen relations and a transductive scheme one top of the inductive for predicting both seen to unseen and unseen to unseen relations (Figure 5, right).

![Figure5](/assets/images/GEN/Figure5.png "Figure 5. (Left) Overall algorithm. (Right) Meta-learned Graph Extrapolation Network." ){: width="90%" height="90%"}

While our meta-learning framework with GEN generalizes to the unseen entities, due to the intrinsic unreliability of few-shot OOG link prediction, there could be high uncertainties on unseen entities. To this end, we model the stochasticity on the unseen entity embeddings by approximating their posterior distribution:
![Figure5.1](/assets/images/GEN/Figure5-1.png){: width="40%" height="40%"}
where mean and variance is obtained from transductive GENs (Please see the main paper for details).

---

# Results

## Main Results

Transductive-GEN (T-GEN) outperforms all baselines on out-of-graph link prediction tasks (See Table1, Table 2).

![Table1](/assets/images/GEN/Table1.png "Table 1. OOG link prediction results on knowledge graph completion." ){: width="75%" height="75%"}

![Table2](/assets/images/GEN/Table2.png "Table 2. OOG link prediction results on Drug-Drug Interaction prediction." ){: width="75%" height="75%"}

## Seen to Unseen and Unseen to Unseen Results

Transductive-GEN (T-GEN) obtains significant performance gain on the unseen to unseen link prediction case (Figure 6).

![Figure6](/assets/images/GEN/Figure6.png "Figure 6. Results of seen to unseen(S/U), unseen to unseen(U/U), and total link prediction of Inductive GEN (I-GEN) and Transductive GEN (T-GEN)." ){: width="75%" height="75%"}

## Visualization Results

The reason why GEN generalizes well to the link prediction with unseen entities is because GEN embeds the unseen entities on the manifold of seen entities while baseline embeds off-manifold (Figure 7).

![Figure7](/assets/images/GEN/Figure7.png "Figure 7. (Left): Seen-to-Unseen baseline (LAN). (Center): Seen-to-Seen baseline, retrained from scratch (TransE). (Right) Ours (T-GEN)." ){: width="75%" height="75%"}

---

# Conclusion

We formally defined a realistic problem of the few-shot out-of-graph (OOG) link prediction task, which considers link prediction not only between seen to unseen (or emerging) entities but also between unseen entities for multi-relational graphs, where each entity comes with only few associative triplets to train. To this end, we proposed a novel meta-learning framework for OOG link prediction, which we refer to as Graph Extrapolation Network (GEN). Under the defined K-shot learning setting, GENs learn to extrapolate the knowledge of a given graph to unseen entities, with a stochastic transductive layer to further propagate the knowledge between the unseen entities and to model uncertainty in the link prediction. We validated the OOG link prediction performance of GENs on five benchmark datasets, on which proposed model largely outperformed the relevant baselines.