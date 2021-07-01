---
title: "Accurate Learning of Graph Representations with Graph Multiset Pooling (ICLR 2021)"
date: 2021-07-01
description: "Summary of the paper: Accurate Learning of Graph Representations with Graph Multiset Pooling, which is accepted to ICLR 2021."
categories:
 - Research
tags: 
    - Research
    - Deep Learning
    - Graph Pooling
    - Graph Multiset Pooling
    - Graph Transformer
    - Graph Multiset Transformer
    - Graph Representation Learning
    - Graph Classification
    - Graph Reconstruction
    - Graph Generation
use_math: true
---

# About

* **Title**: Accurate Learning of Graph Representations with Graph Multiset Pooling.
* **Author**: Jinheon Baek\*, Minki Kang\*, and Sung Ju Hwang. (*: equal contribution)
* **Conference**:  International Conference on Learning Representations (ICLR 2021).
* **Paper**: https://openreview.net/forum?id=JHcqXGaqiGn
* **Code**: https://github.com/JinheonBaek/GMT
* **TL;DR**: We propose a novel graph pooling method for graph representation learning, which considers a multiset scheme with attention-based transformer operations.

---

# Abstract

Graph neural networks have been widely used on modeling graph data, achieving impressive results on node classification and link prediction tasks. Yet, obtaining an accurate representation for a graph further requires a pooling function that maps a set of node representations into a compact form. A simple sum or average over all node representations considers all node features equally without consideration of their task relevance, and any structural dependencies among them. Recently proposed hierarchical graph pooling methods, on the other hand, may yield the same representation for two different graphs that are distinguished by the Weisfeiler-Lehman test, as they suboptimally preserve information from the node features. To tackle these limitations of existing graph pooling methods, we first formulate the graph pooling problem as a multiset encoding problem with auxiliary information about the graph structure, and propose a Graph Multiset Transformer (GMT) which is a multi-head attention based global pooling layer that captures the interaction between nodes according to their structural dependencies. We show that GMT satisfies both injectiveness and permutation invariance, such that it is at most as powerful as the Weisfeiler-Lehman graph isomorphism test. Moreover, our methods can be easily extended to the previous node clustering approaches for hierarchical graph pooling. Our experimental results show that GMT significantly outperforms state-of-the-art graph pooling methods on graph classification benchmarks with high memory and time efficiency, and obtains even larger performance gain on graph reconstruction and generation tasks.

---

# Introduction

## Problem Definition

![Figure1](/assets/images/GMT/Figure1.png "Figure 1. (Left): Conceptual comparison of graph pooling methods. (Right): Illustration of Set, Multiset, and Graph Multiset encoding." ){: width="90%" height="50%"}

Graph pooling is important to represent a whole graph into a compact representation. However, previous graph pooling approaches have obvious drawbacks defined as follows:
* Simple sum pooling can not consider relative importance among nodes (See B. of Figure 1, Left).
* Node drop discards some nodes at pooling, leading to information loss on those discarded nodes (See C. of Figure 1, Left). 
* Node clustering computes the dense cluster matrix, which leads to high computational complexity (See D. of Figure 1, Left).
* Most graph pooling studies overlook the graph isomorphism test except for a few (See green check icon in Figure 1, Left).

## Motivation

To obtain accurate representations of graphs, we need a graph pooling function, which satisfties following properties:
* **(WL Test)** To obtain accurate representations of graphs, we first note that a graph pooling function should be as powerful as the WL (Weisfeiler-Lehman) graph isomorphism test in distinguishing two different graphs.
* **(Multiset Encoding)** We focus on that graph representation learning should be regarded as multiset encoding (See B. Multiset of Figure 1, Right).
* **(Graph Multiset Encoding)** Furthermore, we define a graph multiset encoding (See C. of Figure 1, Right), whose goal is to encode two different graphs into two unique embeddings, by utilizing graph-structured attention units. 

---

# Method

## Graph Multiset Transformer

![Figure2](/assets/images/GMT/Figure2.PNG "Figure 2. Overall architecture of Graph Multiset Transformer." ){: width="95%" height="95%"}

### Graph Multi-head Attention (GMH)

To consider dependencies among nodes of a graph, we use multi-head attention units as a basic component in our pooling scheme:
$\$ \text{MH}(Q, K, V) = \left[ O_1, ..., O_h \right] W^O, \\\ O_i = \text{Att}(QW^Q_i, KW^K_i, VW^V_i), $\$
where $\text{Att}(Q, K, V) = w(Q K^T)V$ with an activation function $w$. The attention function computes the dot product of the query with all keys, to put more weights on the relevant values, namely nodes.

Furthermore, to explicitly leverage the graph structure, we modify the multi-head attention function by constructing the key and value layers using GNNs:
$\$ \text{GMH}(Q, H, A) = \left[ O_1, ..., O_h \right] W^O, \\\ O_i = \text{Att}(QW^Q_i, \text{GNN}^K_i(H, A), \text{GNN}^V_i(H, A)). $\$


### Graph Multiset Pooling (GMPool)

Based on the GMH, we propose a graph pooling function that compresses the $n$ nodes into $k$ typical nodes with a parameterized seed matrix $S$, while taking the graph structure into account:
$\$ \text{GMPool}_{k}(H, A) = \text{LN}(Z + \text{rFF}(Z)), \\\ Z = \text{LN}(S + \text{GMH}(S, H, A)), $\$
where $\text{rFF}$ is any row-wise feedforward layer that processes each individual row independently and identically, and $\text{LN}$ is a layer normalization. Note that the GMH function considers interactions between $k$ seed vectors (queries) in $S$ and $n$ nodes (keys) in $H$, to compress $n$ nodes into $k$ clusters with their attention similarities between queries and keys.

### Self-Attention (SelfAtt)

The GMPool does not consider the relationships between nodes. To tackle this limitation, we propose a Self-Attention function:
$\$ \text{SelfAtt}(H) = \text{LN}(Z + \text{rFF}(Z)), \\\ Z = \text{LN}(H + \text{MH}(H, H, H)), $\$
where, compared to GMH in the above equation that considers interactions between $k$ vectors and $n$ nodes, SelfAtt captures inter-relationships among $n$ nodes by putting node embeddings $H$ on both query and key locations in MH.

### Overall Architecture

A full structure of Graph Multiset Transformer (GMT) consisting of GNNs and pooling layers using ingredients above is depicted in Figure 2.

## Connection with Weisfeiler-Lehman Graph Isomorphism Test

Weisfeiler-Lehman (WL) test is known for its ability to efficiently distinguish two different graphs. Building on previous powerful GNNs, if our graph pooling function is injective, then our overall architecture can be at most as powerful as the WL test, which is formalized in Theorem 1, Lemma 2, and Proposition 3.

**Theorem 1 (Non-isomorphic Graphs to Different Embeddings).** Let $\mathcal{A}: G \rightarrow \mathbb{R}^d$ be a GNN, and Weisfeiler-Lehman test decides two graphs $G_1 \in \mathcal{G}$ and $G_2 \in \mathcal{G}$ as non-isomorphic. Then, $\mathcal{A}$ maps two different graphs $G_1$ and $G_2$ to distinct vectors if node aggregation and update functions are injective, and graph-level readout, which operates on a multiset of node features ${ H_i }$, is injective.

**Lemma 2 (Injectiveness on Graph Multiset Pooling).** Assume the input feature space $\mathcal{H}$ is a countable set. Then the output of $\text{GMPool}_k^i(H, A)$ with $\text{GMH}(S_i, H, A)$ for a seed vector $S_i$ can be unique for each multiset $H \subset \mathcal{H}$ of bounded size. Further, the output of full $\text{GMPool}_k(H, A)$ constructs a multiset with k elements, which are also unique on the input multiset $H$.

**Proposition 3 (Injectiveness on Pooling Function).** The overall Graph Multiset Transformer with multiple GMPool and SelfAtt can map two different graphs $G_1$ and $G_2$ to distinct embedding spaces, such that the resulting GNN with proposed pooling functions can be as powerful as the WL test.

## Connection with Node Clustering Approaches

Node clustering is widely used for coarsening a graph in a hierarchical manner, and also our architecture can be further approximated to the node clustering methods by manipulating an adjacency matrix as formalized in Proposition 5, whereas requiring minimal space complexity as formalized in Theorem 4.

**Theorem 4 (Space Complexity of Graph Multiset Pooling).** Graph Multiset Pooling condsense a graph with $n$ nodes to $k$ nodes in $\mathcal{O}(nk)$ space complexity, which can be further optimized to $\mathcal{O}(n)$.

**Proposition 5 (Approximation to Node Clustering).** Graph Multiset Pooling $\text{GMPool}_k$ can perform hierarchical node clustering with learnable $k$ cluster centroids by Seed Vector $S$.

---

# Results

## Graph Classification

Graph Multiset Transformer (GMT) outperforms all baselines by a large margin on various classification datasets (See Table 1).

![Table1](/assets/images/GMT/classification.PNG "Table 1. Graph classification results on test sets." ){: width="95%" height="95%"}

## Graph Reconstruction

Graph Multiset Pooling (GMPool) obtains significant performance gains on both the synthetic graph and molecule graph reconstruction tasks (Figure 3).

![Figure3](/assets/images/GMT/reconstruction.PNG "Figure 3. Graph classification results on synthetic ring and graph graphs (left) and ZINC molecular graphs (right)." ){: width="95%" height="95%"}

## Graph Generation

Using GMT, instead of simple pooling, results in more stable molecule generations on the QM9 dataset with a MolGAN architecture (Figure 4).

![Figure4](/assets/images/GMT/generation.PNG "Figure 4. Validity curve for molecule generation." ){: width="50%" height="50%"}

## Efficiency

GMT is efficient in terms of both memory and time complexity compared to existing baselines (Figure 5).

![Figure5](/assets/images/GMT/efficiency.PNG "Figure 5. Memory efficiency (left) and Time efficiency (right) of GMT." ){: width="75%" height="75%"}

---

# Conclusion

In this work, we pointed out that existing graph pooling approaches either do not consider the task relevance of each node (sum or mean) or may not satisfy the injectiveness (node drop and clustering methods). To overcome such limitations, we proposed a novel graph pooling method, \emph{Graph Multiset Transformer} (GMT), which not only encodes the given set of node embeddings as a multiset to uniquely embed two different graphs into two distinct embeddings, but also considers both the global structure of the graph and their task relevance in compressing the node features. We theoretically justified that the proposed pooling function is as powerful as the WL test, and can be extended to the node clustering schemes. We validated the proposed GMT on 10 graph classification datasets, and our method outperformed state-of-the-art graph pooling models on most of them. We further showed that our method is superior to the existing graph pooling approaches on graph reconstruction and generation tasks, which require more accurate representations of the graph than classification tasks. We strongly believe that the proposed pooling method will bring substantial practical impact, as it is generally applicable to many graph-learning tasks that are becoming increasingly important. 