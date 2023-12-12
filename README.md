# Papers_for_graph_representation_learning_with_OOD_generalization
papers with out-of-distribution generalization are categorized based on machine learning methods and OOD detection tasks
   
  
|            Model           | Year | Architecture |                             Method                            |                                 Dataset                                 |
|:--------------------------:|:----:|:------------:|:-------------------------------------------------------------:|:-----------------------------------------------------------------------:|
|          CIGA[25]          | 2022 |      GNN     |              Causal invariance, Contrast learning             |                      SPMotif、DrugOOD、ColoredMNIST                     |
|        GNNsafe [57]        | 2023 |      GNN     |                       Energy-based model                      |                          Cora、Amazon、Coauthor                         |
|          EERM [24]         | 2022 |      GNN     |  Autograph induction, Breadth first search, Causal invariance |     Amazon-Photo、Twitch-explicit、Facebook-100、Elliptic、OGB-Arxiv    |
|           DIR[29]          | 2022 |      GNN     |                       Causal invariance                       |              Spurious-Motif、MNIST-75sp、Graph-SST2、Molhiv             |
|           BUP[73]          | 2022 | Bayesian GNN |                            Bayesian                           |                              Cora、Citeseer                             |
|           GDA[9]           | 2023 |      GNN     |                Pair learning, Causal invariance               |  CMNIST-color、Cora-word、Twitch-language、WebKB-universit、CBAS-color  |
| Stable Learning on GNN[10] | 2021 |      GNN     |                        Domain adaptive                        |                   Citeseer 、OGB-Arxiv、Recommendation                  |
|          DGNN [35]         | 2022 |      GNN     |                       Causal invariance                       |                       Cora、Citeseer、Pubmed、NELL                      |
|        StableGNN[36]       | 2021 |      GNN     |                       Causal invariance                       | Molbace、Molbbbp、Molhiv、MUTAG、Molclintox、Moltox21、Molesol、Mollipo |
|          CSBM [74]         | 2021 |      GNN     |         Standard random block, Gaussian mixture model         |                          Cora、PubMed、Wiki.Net                         |
|          GKDE [75]         | 2020 |      GNN     |               Dirichlet distributions, Bayesian               |  Cora、Citeseer、Pubmed、Amazon Photo、Amazon Compute、Coauthor Physic  |
|         OSSNC [55]         | 2022 |     GCNII    |            Internal and external  dual optimization           |                    Cora、Citeseer、Pubmed、ogbn-arXiv                   |
|         SR-GNN [76]        | 2021 |      GNN     |                        Domain adaption                        |                  Cora、Citeseer、PubMed、ogban、Reddit                  |
|           DSE[77]          | 2022 |      GNN     | Conditional  ariational Grap, Auto-Encoder, Causal invariance |                        TR3、MNISTsup、Graph-SST2                        |
|         OOD-GNN[78]        | 2022 |      GNN     |                Random Fourier, Characteristics                |              TRIANGLES、MNIST-75SP、COLLAB、 PROTEINS、 D&D             |
|        Graphde [14]        | 2022 |      GNN     |                     Variational inference                     |               Spurious-Motif、MNIST-75sp 、Collab、DrugOOD              |
|        MoleOOD [18]        | 2022 |      GNN     |                       Invariant learning                      |                       BACE、BBBP、SIDER、HIV、OGB                       |
|           SFP[39]          | 2022 |      GNN     |         Model sparsity, singular, Value decomposition         |                             PLACE365、MSCOCO                            |
|           GIL[28]          | 2022 |      GNN     |                       Invariant learning                      |       SP-Motif、MNIST-75sp、Graph-SST2、Open Graph Benchmark (OGB)      |
|          LiSA[23]          | 2023 |      GNN     |           Invariant learning, Subgraph augmentation           |                  Spurious-Motif、MUTAG、D&D、MNIST-75sp                 |
|          MOOD[79]          | 2022 |      GCN     |             OOD-controlled Reverse-time diffusion             |                    TR3、MNIST superpixels、Graph-SST2                   |
|           MOG[80]          | 2021 |      GCN     |               Improved Langevin, Dynamics method              |                              ZINC250k、QM9                              |
|          DSIL[38]          | 2023 |      GNN     |                        Domain invariant                       |                            Drugbank、Twosides                           |
|         IS-GIB[11]         | 2023 |      GNN     |                     Information Bottleneck                    |   Cora、Citeseer、Twitch-explicit、Facebook100、Gossipcop、Politifact   |
|         GOOD-D [12]        | 2023 |      GNN     |             Hierarchical graph, Contrast learning             |                                 TU、OGB                                 |
## Methods of machine learning
### 1.1 Supervised learning
(1)DIR:  Discovering invariant rationales for graph neural networks  
      YX Wu, X Wang, A Zhang, X He, TS Chuag  
( [paper](https://arxiv.org/abs/2201.12872)  [code](https://github.com/Wuyxin/DIR-GNN) )  
  
(2)CIGA:  Learning causally invariant representations for out-of-distribution generalization on graphs  
   Y Chen, Y Zhang, Y Bian, H Yang, MA Kaili, B Xie, T Liu, B Han, J Cheng  
 ([paper](https://arxiv.org/abs/2202.05441)  [code](https://github.com/LFhase/CIGA) ) 
   
(3)GNNSAFE:  Energy-based out-of-distribution detection for graph neural networks  
   Q Wu, Y Chen, C Yang, J Yan  
 ([paper](https://arxiv.org/abs/2302.02914)  [code](https://github.com/qitianwu/GraphOOD-GNNSafe) )  
   
(4)BUP:  Uncertainty Propagation in Node Classification   
   Z Xu, C Lawrence, A Shaker, R Siarheyeu  
([paper](https://www.computer.org/csdl/proceedings-article/icdm/2022/509900b275/1KpCBAulk2Y)   )    
  
(5)GDA:  Graph Structure and Feature Extrapolation for Out-of-Distribution Generalization  
   X Li, S Gui, Y Luo, S Ji  
([paper](https://doi.org/10.48550/arXiv.2306.08076)  )      
  
(6)STABLE learning on Graph:  Stable prediction on graphs with agnostic distribution shift  
   S Zhang, K Kuang, J Qiu, J Yu, Z Zhao, H Yang et al.  
([paper](https://arxiv.org/abs/2110.03865)   )      
  
(7)EERM:  Handling distribution shifts on graphs: An invariance perspective  
   Q Wu, H Zhang, J Yan, D Wipf  
([paper](https://arxiv.org/abs/2202.02466)  [code](https://github.com/qitianwu/GraphOOD-EERM) )  
  
### 1.2 Semi-Supervised learning
(1)DGNN:  Debiased graph neural networks with agnostic label selection bias  
   S Fan, X Wang, C Shi, K Kuang, N Liu, B Wang  
([paper](https://ieeexplore.ieee.org/document/9698407) )      
  
(2)StableGNN:  Generalizing graph neural networks on out-of-distribution graphs  
   S Fan, X Wang, C Shi, P Cui, B Wang  
([paper](https://ieeexplore.ieee.org/document/10268633) )   
  
(3)CSBM:  Graph convolution for semi-supervised classification: Improved linear separability and out-of-distribution generalization  
   A Baranwal, K Fountoulakis, A Jagannath  
([paper](https://arxiv.org/abs/2102.06966v1) )      
  
(4)GKDE:  Uncertainty aware semi-supervised learning on graph data  
   X Zhao, F Chen, S Hu, JH Cho  
([paper](https://www.semanticscholar.org/reader/71a35aa42cd1ed6f213e58122154739dfd6340e8)  [code](https://github.com/zxj32/uncertainty-GNN) )      
  
(5)OSSNC:  End-to-end open-set semi-supervised node classification with out-of-distribution detection  
   S Fan, X Wang, C Shi, K Kuang, N Liu, B Wang  
([paper](https://ink.library.smu.edu.sg/sis_research/7479) )      
  
(6)SR-GNN:  Shift-robust gnns: Overcoming the limitations of localized graph training data  
   Q Zhu, N Ponomareva, J Han, B Perozzi  
([paper](https://doi.org/10.48550/arXiv.2108.01099)  [code](https://github.com/GentleZhu/Shift-Robust-GNNs) )    
  
### 1.3 Unsupervised learning
(1)DSE:  Deconfounding to explanation evaluation in graph neural networks  
   YX Wu, X Wang, A Zhang, X Hu, F Feng, X He, TS Chua  
([paper](https://arxiv.org/abs/2201.08802)  [code](https://anonymous.4open.science/r/DSE-24BC) )      
  
(2)OOD-GNN:  Ood-gnn: Out-of-distribution generalized graph neural network  
   H Li, X Wang, Z Zhang, W Zhu  
([paper](https://arxiv.org/abs/2112.03806) )      
  
(3)graphde:  Graphde: A generative framework for debiased learning and out-of-distribution detection on graphs  
   Li Zenan, Wu Qitian, Nie Fan, Yan Junchi  
([paper](https://github.com/Emiyalzn/GraphDE)  [code](https://github.com/Emiyalzn/GraphDE) )    
  
(4)MoleOOD:  Learning substructure invariance for out-of-distribution molecular representations  
   N Yang, K Zeng, Q Wu, X Jia, J Yan  
([paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/547108084f0c2af39b956f8eadb75d1b-Abstract-Conference.html )  [code](https://github.com/yangnianzu0515/MoleOOD) )   
  
(5)SFP:  Exploring optimal substructure for out-of-distribution generalization via feature-targeted model pruning  
   Y Wang, J Guo, S Guo, W Zhang, J Zhang  
([paper](https://arxiv.org/abs/2212.09458) )  
(6)GIL:  Learning invariant graph representations for out-of-distribution generalization  
   H Li, Z Zhang, X Wang, W Zhu  
([paper](https://openreview.net/forum?id=acKK8MQe2xc) )  
  
(7)LiSA:  Mind the Label Shift of Augmentation-based Graph OOD Generalization  
   J Yu, J Liang, R He  
([paper](https://arxiv.org/abs/2303.14859)  [code](https://github.com/Samyu0304/LiSA) )      
  
(8)MOOD:  Exploring chemical space with score-based out-of-distribution generation  
   S Lee, J Jo, SJ Hwang  
([paper](https://arxiv.org/abs/2206.07632)  [code](https: //github.com/SeulLee05/MOOD) )      
  
(9)MOG:  MOG: Molecular Out-of-distribution Generation with Energy-based Models  
   S Lee, DB Lee, SJ Hwang  
([paper](https://openreview.net/forum?id=qkTEaJ9orc1) ) 
  
(10)DSIL:  DSIL-DDI: A Domain-Invariant Substructure Interaction Learning for Generalizable Drug–Drug Interaction Prediction  
   Z Tang, G Chen, H Yang, W Zhong, CYC Chen  
([paper](https://ieeexplore.ieee.org/document/10044475) )      
  
(11)IS-GIB:  Individual and Structural Graph Information Bottlenecks for Out-of-Distribution Generalization  
   L Yang, J Zheng, H Wang, Z Liu, Z Huang, S Hong, W Zhang, B Cui  
([paper](https://arxiv.org/abs/2306.15902)  [code](https://github.com/YangLing0818/GraphOOD) )    
  
### 1.4 Self-Supervised learning
(1)GOOD-D:  GOOD-D: On Unsupervised Graph Out-Of-Distribution Detection  
   Y Liu, K Ding, H Liu, S Pan  
([paper](https://doi.org/10.1145/3539597.3570446)  )  



     
## OOD Detection Tasks 
### 2.1 Graph-Level Tasks 
(1)CSBM  ([paper](https://arxiv.org/abs/2102.06966v1) )      
Graph convolution for semi-supervised classification: Improved linear separability and out-of-distribution generalization  
   A Baranwal, K Fountoulakis, A Jagannath  
  

(2) DIR ( [paper](https://arxiv.org/abs/2201.12872)  [code](https://github.com/Wuyxin/DIR-GNN) )      
Discovering invariant rationales for graph neural networks  
      YX Wu, X Wang, A Zhang, X He, TS Chuag  
  

(2)MOG  ([paper](https://openreview.net/forum?id=qkTEaJ9orc1) )      
MOG: Molecular Out-of-distribution Generation with Energy-based Models  
   S Lee, DB Lee, SJ Hwang  
  

(4) GDA ([paper](https://doi.org/10.48550/arXiv.2306.08076)  )      
Graph Structure and Feature Extrapolation for Out-of-Distribution Generalization  
   X Li, S Gui, Y Luo, S Ji  
  

(5)graphde  ([paper](https://github.com/Emiyalzn/GraphDE)  [code](https://github.com/Emiyalzn/GraphDE) )      
Graphde: A generative framework for debiased learning and out-of-distribution detection on graphs  
   Li Zenan, Wu Qitian, Nie Fan, Yan Junchi  
  

(6) OOD-GNN  ([paper](https://arxiv.org/abs/2112.03806) )      
Ood-gnn: Out-of-distribution generalized graph neural network  
   H Li, X Wang, Z Zhang, W Zhu  
  

(6)GOOD-D  ([paper](https://doi.org/10.1145/3539597.3570446)  )      
GOOD-D: On Unsupervised Graph Out-Of-Distribution Detection  
   Y Liu, K Ding, H Liu, S Pan  
  

(8) IS-GIB  ([paper](https://arxiv.org/abs/2306.15902)  [code](https://github.com/YangLing0818/GraphOOD) )      
Individual and Structural Graph Information Bottlenecks for Out-of-Distribution Generalization  
   L Yang, J Zheng, H Wang, Z Liu, Z Huang, S Hong, W Zhang, B Cui  
  

(9)  MOOD  ([paper](https://arxiv.org/abs/2206.07632)  [code](https: //github.com/SeulLee05/MOOD) )      
Exploring chemical space with score-based out-of-distribution generation  
   S Lee, J Jo, SJ Hwang  
  

(10) STABLE learning on Graph ([paper](https://arxiv.org/abs/2110.03865)   )      
Stable prediction on graphs with agnostic distribution shift  
   S Zhang, K Kuang, J Qiu, J Yu, Z Zhao, H Yang et al.  
  

### 2.2 Subgraph-level Tasks 
(1)GDA ([paper](https://doi.org/10.48550/arXiv.2306.08076)  )     
Graph Structure and Feature Extrapolation for Out-of-Distribution Generalization  
   X Li, S Gui, Y Luo, S Ji  
   

(2) CIGA ([paper](https://arxiv.org/abs/2202.05441)  [code](https://github.com/LFhase/CIGA) )      
Learning causally invariant representations for out-of-distribution generalization on graphs  
   Y Chen, Y Zhang, Y Bian, H Yang, MA Kaili, B Xie, T Liu, B Han, J Cheng  
 ([paper](https://arxiv.org/abs/2202.05441)  
  

(3) DIR ( [paper](https://arxiv.org/abs/2201.12872)  [code](https://github.com/Wuyxin/DIR-GNN) )     
Discovering invariant rationales for graph neural networks  
      YX Wu, X Wang, A Zhang, X He, TS Chuag   
  

(4)LiSA  ([paper](https://arxiv.org/abs/2303.14859)  [code](https://github.com/Samyu0304/LiSA) )      
Mind the Label Shift of Augmentation-based Graph OOD Generalization  
   J Yu, J Liang, R He  
  

(5) GIL  ([paper](https://openreview.net/forum?id=acKK8MQe2xc) )    
Learning invariant graph representations for out-of-distribution generalization  
   H Li, Z Zhang, X Wang, W Zhu  
    

(6) SFP  ([paper](https://arxiv.org/abs/2212.09458) )      
Exploring optimal substructure for out-of-distribution generalization via feature-targeted model pruning  
   Y Wang, J Guo, S Guo, W Zhang, J Zhang  
  

(7) MoleOOD  ([paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/547108084f0c2af39b956f8eadb75d1b-Abstract-Conference.html )  [code](https://github.com/yangnianzu0515/MoleOOD) )     
 Learning substructure invariance for out-of-distribution molecular representations  
   

(8) OOD-GNN  ([paper](https://arxiv.org/abs/2112.03806) )      
Ood-gnn: Out-of-distribution generalized graph neural network  
   H Li, X Wang, Z Zhang, W Zhu  
  

(9)DSIL  ([paper](https://ieeexplore.ieee.org/document/10044475) )      
 DSIL-DDI: A Domain-Invariant Substructure Interaction Learning for Generalizable Drug–Drug Interaction Prediction  
   Z Tang, G Chen, H Yang, W Zhong, CYC Chen  
  

(10)GOOD-D  ([paper](https://doi.org/10.1145/3539597.3570446)  )   
GOOD-D: On Unsupervised Graph Out-Of-Distribution Detection  
   Y Liu, K Ding, H Liu, S Pan  
     

(11) StableGNN  ([paper](https://ieeexplore.ieee.org/document/10268633) )      
Generalizing graph neural networks on out-of-distribution graphs  
   S Fan, X Wang, C Shi, P Cui, B Wang  
  

### 2.3 Node-Level Tasks
(1)GKDE  ([paper](https://www.semanticscholar.org/reader/71a35aa42cd1ed6f213e58122154739dfd6340e8)  [code](https://github.com/zxj32/uncertainty-GNN) )     
Uncertainty aware semi-supervised learning on graph data  
   X Zhao, F Chen, S Hu, JH Cho  
   

(2) DGNN  ([paper](https://ieeexplore.ieee.org/document/9698407) )      
Debiased graph neural networks with agnostic label selection bias  
   S Fan, X Wang, C Shi, K Kuang, N Liu, B Wang  
  

(3) GNNSAFE ([paper](https://arxiv.org/abs/2302.02914)  [code](https://github.com/qitianwu/GraphOOD-GNNSafe) )      
Energy-based out-of-distribution detection for graph neural networks  
   Q Wu, Y Chen, C Yang, J Yan  
  

(4)EERM ([paper](https://arxiv.org/abs/2202.02466)  [code](https://github.com/qitianwu/GraphOOD-EERM) )      
Handling distribution shifts on graphs: An invariance perspective  
   Q Wu, H Zhang, J Yan, D Wipf  
  

(5) BUP ([paper](https://www.computer.org/csdl/proceedings-article/icdm/2022/509900b275/1KpCBAulk2Y)   )      
Uncertainty Propagation in Node Classification   
   Z Xu, C Lawrence, A Shaker, R Siarheyeu  
  
(6)GDA ([paper](https://doi.org/10.48550/arXiv.2306.08076)  )      
Graph Structure and Feature Extrapolation for Out-of-Distribution Generalization  
   X Li, S Gui, Y Luo, S Ji  
   
(7)GOOD-D  ([paper](https://doi.org/10.1145/3539597.3570446)  )    
GOOD-D: On Unsupervised Graph Out-Of-Distribution Detection  
   Y Liu, K Ding, H Liu, S Pan  
## 1 Methods of machine learning  

  
1.1 Supervised learning

(1) DIR ( [paper](https://arxiv.org/abs/2201.12872)  [code](https://github.com/Wuyxin/DIR-GNN) )      
(2) CIGA ([paper](https://arxiv.org/abs/2202.05441)  [code](https://github.com/LFhase/CIGA) )      
(3) GNNSAFE ([paper](https://arxiv.org/abs/2302.02914)  [code](https://github.com/qitianwu/GraphOOD-GNNSafe) )      
(4) BUP ([paper](https://www.computer.org/csdl/proceedings-article/icdm/2022/509900b275/1KpCBAulk2Y)   )      
(5) GDA ([paper](https://doi.org/10.48550/arXiv.2306.08076)  )      
(6) STABLE learning on Graph ([paper](https://arxiv.org/abs/2110.03865)   )      
(7) EERM  learning on Graph ([paper](https://arxiv.org/abs/2202.02466)  [code](https://github.com/qitianwu/GraphOOD-EERM) )


1.2 Semi-Supervised learning

(1) DGNN  ([paper](https://ieeexplore.ieee.org/document/9698407) )      
(2) StableGNN  ([paper](https://ieeexplore.ieee.org/document/10268633) )      
(3) CSBM  ([paper](https://arxiv.org/abs/2102.06966v1) )      
(4) GKDE  ([paper](https://www.semanticscholar.org/reader/71a35aa42cd1ed6f213e58122154739dfd6340e8)  [code](https://github.com/zxj32/uncertainty-GNN) )      
(5) OSSNC  ([paper](https://ink.library.smu.edu.sg/sis_research/7479) )      
(6) SR-GNN  ([paper](https://doi.org/10.48550/arXiv.2108.01099)  [code](https://github.com/GentleZhu/Shift-Robust-GNNs) )      


1.3 Unsupervised learning

(1) DSE  ([paper](https://arxiv.org/abs/2201.08802)  [code](https://anonymous.4open.science/r/DSE-24BC) )      
(2) OOD-GNN  ([paper](https://arxiv.org/abs/2112.03806) )      
(3) graphde  ([paper](https://github.com/Emiyalzn/GraphDE)  [code](https://github.com/Emiyalzn/GraphDE) )      
(4) MoleOOD  ([paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/547108084f0c2af39b956f8eadb75d1b-Abstract-Conference.html )  [code](https://github.com/yangnianzu0515/MoleOOD) )      
(5) SFP  ([paper](https://arxiv.org/abs/2212.09458) )      
(6) GIL  ([paper](https://openreview.net/forum?id=acKK8MQe2xc) )      
(7) LiSA  ([paper](https://arxiv.org/abs/2303.14859)  [code](https://github.com/Samyu0304/LiSA) )      
(8) MOOD  ([paper](https://arxiv.org/abs/2206.07632)  [code](https://github.com/SeulLee05/MOOD) )      
(9) MOG  ([paper](https://openreview.net/forum?id=qkTEaJ9orc1) )      
(10) DSIL  ([paper](https://ieeexplore.ieee.org/document/10044475) )      
(11) IS-GIB  ([paper](https://arxiv.org/abs/2306.15902)  [code](https://github.com/YangLing0818/GraphOOD) )      


1.4 Self-Supervised learning

(1)GOOD-D  ([paper](https://doi.org/10.1145/3539597.3570446)  )      
  
                                                                  
                                                                    
                                                    
## 2 OOD Detection Tasks 
### 2.1 Graph-Level Tasks 

(1) CSBM  ([paper](https://arxiv.org/abs/2102.06966v1) )      
(2) DIR ( [paper](https://arxiv.org/abs/2201.12872)  [code](https://github.com/Wuyxin/DIR-GNN) )      
(3) MOG  ([paper](https://openreview.net/forum?id=qkTEaJ9orc1) )      
(4) GDA ([paper](https://doi.org/10.48550/arXiv.2306.08076)  )      
(5) graphde  ([paper](https://github.com/Emiyalzn/GraphDE)  [code](https://github.com/Emiyalzn/GraphDE) )      
(6) OOD-GNN  ([paper](https://arxiv.org/abs/2112.03806) )      
(7) GOOD-D  ([paper](https://doi.org/10.1145/3539597.3570446)  )      
(8) IS-GIB  ([paper](https://arxiv.org/abs/2306.15902)  [code](https://github.com/YangLing0818/GraphOOD) )      
(9)  MOOD  ([paper](https://arxiv.org/abs/2206.07632)  [code](https://github.com/SeulLee05/MOOD) )      
(10) STABLE learning on Graph ([paper](https://arxiv.org/abs/2110.03865)   )      


### 2.2 Subgraph-level Tasks 

(1) GDA ([paper](https://doi.org/10.48550/arXiv.2306.08076)  )      
(2) CIGA ([paper](https://arxiv.org/abs/2202.05441)  [code](https://github.com/LFhase/CIGA) )      
(3) DIR ( [paper](https://arxiv.org/abs/2201.12872)  [code](https://github.com/Wuyxin/DIR-GNN) )      
(4) LiSA  ([paper](https://arxiv.org/abs/2303.14859)  [code](https://github.com/Samyu0304/LiSA) )      
(5) GIL  ([paper](https://openreview.net/forum?id=acKK8MQe2xc) )      
(6) SFP  ([paper](https://arxiv.org/abs/2212.09458) )      
(7) MoleOOD  ([paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/547108084f0c2af39b956f8eadb75d1b-Abstract-Conference.html )  [code](https://github.com/yangnianzu0515/MoleOOD) )      
(8) OOD-GNN  ([paper](https://arxiv.org/abs/2112.03806) )      
(9) DSIL  ([paper](https://ieeexplore.ieee.org/document/10044475) )      
(10) GOOD-D  ([paper](https://doi.org/10.1145/3539597.3570446)  )      
(11) StableGNN  ([paper](https://ieeexplore.ieee.org/document/10268633) )      


### 2.3 Node-Level Tasks

(1) GKDE  ([paper](https://www.semanticscholar.org/reader/71a35aa42cd1ed6f213e58122154739dfd6340e8)  [code](https://github.com/zxj32/uncertainty-GNN) )      
(2) DGNN  ([paper](https://ieeexplore.ieee.org/document/9698407) )      
(3) GNNSAFE ([paper](https://arxiv.org/abs/2302.02914)  [code](https://github.com/qitianwu/GraphOOD-GNNSafe) )      
(4) EERM ([paper](https://arxiv.org/abs/2202.02466)  [code](https://github.com/qitianwu/GraphOOD-EERM) )      
(5) BUP ([paper](https://www.computer.org/csdl/proceedings-article/icdm/2022/509900b275/1KpCBAulk2Y)   )      
(6) GDA ([paper](https://doi.org/10.48550/arXiv.2306.08076)  )      
(7) GOOD-D  ([paper](https://doi.org/10.1145/3539597.3570446)  )    
