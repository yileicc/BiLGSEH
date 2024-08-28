# BiLGSEH
title: Bi-Direction Label-Guided Semantic Enhancement for Cross-Modal Hashing

## Introduction
Supervised cross-modal hashing has garnered considerable interest owing to its lower storage and computation costs, as well as the rich semantics it provides. Although pioneers have made tremendous efforts to produce compact binary codes, little attention has been paid to two limitations: (1) Labels have not been fully utilized for multi-grained semantics mining and fusion; (2) Cross-modal interaction lacks reliability as it fails to fully consider multi-grained semantics and accurate sample relationships. To alleviate such limitations, this paper puts forward a novel Bi-direction Label-Guided Semantic Enhancement cross-modal Hashing method, abbreviated as BiLGSEH. Specifically, to solve the first issue, a new designed label-guided semantic fusion strategy is introduced to excavate and fuse multi-grained semantic features guided by multi-labels. To address the second problem, a novel semantic-enhanced relation aggregation strategy is used to construct and aggregate abundant multi-modal relation information via bi-direction similarity. In addition, CLIP features are involved to enhance alignment between multi-modal contents and complex semantics. In a nutshell, BiLGSEH could produce discriminative hash codes by effectively aligning both semantics distribution and relation structure. This superiority is verified by performance comparisons with 14 competitive solutions.

## datasets
The CLIP dataset used in this paper is sourced from the paper titled "When CLIP Meets Cross-modal Hashing Retrieval: A New Strong Baseline." You can visit the paper's page to download the corresponding dataset.

## Demo
Taking NUS-WIDE as an example, our model can be trained and verified by the following command: python main_nus.py
