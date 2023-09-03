# Awesome Neuro-Symbolic AI
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)](https://github.com/traincheckai/awesome-neuro-symbolic-ai/pulls) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Stars](https://img.shields.io/github/stars/traincheckai/awesome-neuro-symbolic-ai?color=yellow)  ![Forks](https://img.shields.io/github/forks/traincheckai/awesome-neuro-symbolic-ai?color=blue&label=Fork)

A curated list of awesome Neuro-Symbolic AI (NSAI or NeSy) software frameworks.

If you want to contribute to this list (please do), send me a pull request or contact me [@mattfaltyn](https://mattfaltyn.github.io/).




## Table of Contents
- [NSAI Basics and Resources](#nsai-basics-and-resources)
  * [NSAI in Two Sentences](#nsai-in-two-sentences)
  * [Textbooks](#textbooks)
  * [Overview Articles](#overview-articles)
  * [Workshops](#workshops)
- [NSAI Categories](#nsai-categories)
  * [Category 1: Sequential](#category-1--sequential)
    + [Type I](#type-i)
  * [Category 2: Nested](#category-2--nested)
    + [Type II](#type-ii)
  * [Category 3: Cooperative](#category-3--cooperative)
    + [Type III](#type-iii)
  * [Category 4: Compiled](#category-4--compiled)
    + [Type IV](#type-iv)
    + [Type V](#type-v)
- [Frameworks](#frameworks)
  * [Logical Neural Network](#logical-neural-network)
    + [Software](#software)
    + [Media](#media)
      - [Academic Papers](#academic-papers)
      - [Blogs](#blogs)
  * [Logic Tensor Networks](#logic-tensor-networks)
  * [Neural Logic Machines](#neural-logic-machines)
  * [Hinge-Loss Markov Random Fields and Probabilistic Soft Logic](#hinge-loss-markov-random-fields-and-probabilistic-soft-logic)
  * [TensorLog](#tensorlog)
  * [Markov Logic Networks](#markov-logic-networks)


## NSAI Basics and Resources

### NSAI in Two Sentences
"NSAI aims to build rich computational AI models, systems and applications by combining neural and symbolic learning and reasoning. It hopes to create synergies among the strengths of neural and symbolic AI while overcoming their complementary weaknesses." - [Sixteenth International Workshop on Neural-Symbolic Learning and Reasoning](https://sites.google.com/view/nesy-2022/)

### Textbooks
- [2022 - Neuro-Symbolic Artificial Intelligence: The State of the Art](https://www.iospress.com/catalog/books/neuro-symbolic-artificial-intelligence-the-state-of-the-art)
- [2009 - Neural-Symbolic Cognitive Reasoning](https://link.springer.com/book/10.1007/978-3-540-73246-4)
- [2002 - Neural-Symbolic Learning Systems: Foundations and Applications](https://link.springer.com/book/10.1007/978-1-4471-0211-3)

### Overview Articles
- [2022 - Towards Data-and Knowledge-Driven Artificial Intelligence: A Survey on Neuro-Symbolic Computing](https://arxiv.org/abs/2210.15889)
- [2022 - Is Neuro-Symbolic AI Meeting its Promise in Natural Language Processing? A Structured Review](https://arxiv.org/abs/2202.12205)
- [2021 - Neuro-Symbolic Artificial Intelligence: Current Trends](https://arxiv.org/abs/2105.05330)
- [2021 - Neuro-Symbolic VQA: A review from the perspective of AGI desiderata](https://arxiv.org/abs/2104.06365)
- [2017 - Neural-Symbolic Learning and Reasoning: A Survey and Interpretation](https://arxiv.org/abs/1711.03902)
- [2015 - Neural-Symbolic Learning and Reasoning: Contributions and Challenges](https://openaccess.city.ac.uk/id/eprint/11838/)
- [2009 - The Facets of Artificial Intelligence: A Framework to Track the Evolution of AI](https://www.ijcai.org/proceedings/2018/0718.pdf)
- [2005 - Dimensions of Neural-symbolic Integration - A Structured Survey](https://arxiv.org/abs/cs/0511042)
- [2004 - The Integration of Connectionism and First-Order Knowledge Representation and Reasoning as a Challenge for Artificial Intelligence](https://ui.adsabs.harvard.edu/abs/2004cs........8069B/abstract)

### Workshops
- [IBM Neuro-Symbolic AI Summer School [August 8-9, 2022]](https://video.ibm.com/playlist/655332)
- [IBM Neuro-Symbolic AI Workshop - Jan 2022](https://video.ibm.com/playlist/649216)
- [NeuroSymbolic Artificial Intelligence Course](https://cedar.buffalo.edu/~srihari/CSE701/index.html)



## NSAI Categories
[Henry Kautz's](https://en.wikipedia.org/wiki/Henry_Kautz) taxonomy from his Robert S. Englemore Memorial Lecture in 2020 at the Thirty-Fourth AAAI Conference on Artificial Intelligence (slides [here](https://ai.ntu.edu.tw/mlss2021/wp-content/uploads/2021/08/0804-Henry-Kautz.pdf)) is informal standard for categorizing neuro-symbolic architectures. [Hamilton et al (2022)](https://arxiv.org/abs/2202.12205) reframed Kautz's taxonomy into four categories to make it more intuitive. We omit Kautz's Type VI as no architectures currently exist under that category. 


### Category 1: Sequential 

####  Type I

A Type I (symbolic Neuro symbolic) system is standard deep learning. This class is included in the taxonomy as the input and output of a neural network can be symbols (such as words in language translation) that are vectorized within the model. Some Type I architectures include:
- [QDGAT](https://arxiv.org/abs/2009.07448)
- [TBox](https://www.researchgate.net/profile/Yu-Gu-24/publication/359972607_Local_ABox_Consistency_Prediction_with_Transparent_TBoxes_Using_Gated_Graph_Neural_Networks/links/62591a8c709c5c2adb7d16f5/Local-ABox-Consistency-Prediction-with-Transparent-TBoxes-Using-Gated-Graph-Neural-Networks.pdf)
- [Sememes-Based Framework for KG Embeddings](https://link.springer.com/chapter/10.1007/978-3-030-82147-0_34)
- [D4-Gumbel](https://aclanthology.org/P19-1026/)
- [Cowen-Rivers et al (2019)](https://arxiv.org/abs/1906.04985)
- [NTF-IDF](https://www.emerald.com/insight/content/doi/10.1108/IJWIS-11-2020-0067/full/html)
- [FDLC](https://ieeexplore.ieee.org/document/9333555)
- [RE-ILP](https://aclanthology.org/R19-1076/)
- [Hierarchical Graph Transformer](https://ieeexplore.ieee.org/document/8988213)
- [DialogueCRN](https://arxiv.org/abs/2106.01978)
- [Hybrid LOINC Mapping](https://aclanthology.org/2021.naloma-1.2.pdf)
- [The CoRg Project](https://west.uni-koblenz.de/pdf/2e08f404b02ea69db9511b7ba86e3dc6.pdf)
- [Skip-thought + MaLSTM](https://ieeexplore.ieee.org/document/8995321)
- [Dependency Tree-LSTM](https://arxiv.org/pdf/2103.03755.pdf)
- [Abstractive Text Summarization](https://aclanthology.org/2021.cl-4.27.pdf)
- [NMT+Word2Vec](https://ieeexplore.ieee.org/document/8873551)


### Category 2: Nested 

#### Type II

A Type II (Symbolic[Neuro]) system is a hybrid system in which a symbolic solver utilizes neural networks as subroutines to solve one or more tasks. Some Type II frameworks include:
- AlphaGo


### Category 3: Cooperative 

#### Type III

A Type III (Neuro; Symbolic) system is a hybrid system where a neural network solves one task and interacts via its input and output with a symbolic system that solves a different task. Some Type III frameworks include:
- Neural-Concept Learner


### Category 4: Compiled 

#### Type IV

Type IV (Neuro: Symbolic → Neuro) is a system in which the symbolic knowledge is compiled into the training set of a neural network. Some Type IV frameworks include:
- [2020 - Logical Neural Networks](#logical-neural-network)

#### Type V

A Type V (Neuro_Symbolic) system is a tightly-coupled but distributed neuro-symbolic systems where a symbolic logic rule is mapped onto an embedding which acts as a soft-constraint on the network’s loss function. These systems are often tensorized in some manner. Some Type V frameworks include:
- [2020 - Logic Tensor Networks](#logic-tensor-networks)
- [2019 - Neural Logic Machines](#neural-logic-machines)
- 2017 - Hinge-Loss Markov Random Fields and Probabilistic Soft Logic
- 2016 - TensorLog
- 2006 - Markov Logic Networks





## Frameworks 

In this section, we aim to provide the most comprehensive NSAI frameworks to date.

### Logical Neural Network
A `Neural = Symbolic` framework for sound and complete weighted real-value logic created by IBM Research. 

#### Software
- See the [IBM NSAI Toolkit](https://ibm.github.io/neuro-symbolic-ai/toolkit) for a full list of associated repositories. 

#### Media

##### Academic Papers
- [2022 - Foundations of Reasoning with Uncertainty via Real-valued Logics](https://arxiv.org/abs/2008.02429)
- [2022 - Extending Logical Neural Networks Using First-Order Theories](https://arxiv.org/pdf/2207.02978.pdf)
- [2021 - Neuro-Symbolic Inductive Logic Programming with Logical Neural Networks](https://arxiv.org/abs/2112.03324)
- [2021 - Training Logical Neural Networks by Primal–Dual Methods for Neuro-Symbolic Reasoning](https://ieeexplore.ieee.org/document/9415044)
- [2021 - Proof Extraction for Logical Neural Networks](https://openreview.net/forum?id=Xw3kb6UyA31) 
- [2021 - Neuro-Symbolic Approaches for Text-Based Policy Learning](https://aclanthology.org/2021.emnlp-main.245/)
- [2021 - Neuro-Symbolic Reinforcement Learning with First-Order Logic](https://aclanthology.org/2021.emnlp-main.283/)
- [2021 - LOA: Logical Optimal Actions for Text-based Interaction Games](https://aclanthology.org/2021.acl-demo.27/)
- [2021 - Reinforcement Learning with External Knowledge by using Logical Neural Networks](https://arxiv.org/abs/2103.02363)
- [2021 - LNN-EL: A Neuro-Symbolic Approach to Short-text Entity Linking](https://aclanthology.org/2021.acl-long.64/)
- [2021 - Leveraging Abstract Meaning Representation for Knowledge Base Question Answering](https://aclanthology.org/2021.findings-acl.339/)
- [2021 - Logic Embeddings for Complex Query Answering](https://arxiv.org/abs/2103.00418)
- [2020 - Logical Neural Networks](https://arxiv.org/abs/2006.13155)

##### Blogs
- [2021 - IBM, MIT and Harvard release “Common Sense AI” dataset at ICML 2021](https://research.ibm.com/blog/icml-darpa-agent)
- [2021 - AI, you have a lot of explaining to do](https://research.ibm.com/blog/explaining-commonsense-ai)
- [2020 - Logical Neural Networks](https://skirmilitor.medium.com/logical-neural-networks-31498d1aa9be)
- [2020 - NSQA: Neuro-Symbolic Question Answering](https://towardsdatascience.com/nsqa-neuro-symbolic-question-answering-6d14d98e88f3)
- [2020 - Semantic Parsing using Abstract Meaning Representation](https://medium.com/@sroukos/semantic-parsing-using-abstract-meaning-representation-95242518a380)
- [2020 - A Semantic Parsing-based Relation Linking approach for Knowledge Base Question Answering](https://medium.com/@sroukos/a-semantic-parsing-based-relation-linking-approach-for-knowledge-base-question-answering-93c14d7931c1)
- [2020 - Getting AI to Reason: Using Logical Neural Networks for Knowledge-Based Question Answering](https://medium.com/swlh/getting-ai-to-reason-using-logical-neural-networks-for-knowledge-based-question-answering-60456654f5fa)
- [2020 - Neurosymbolic AI to Give Us Machines With True Common Sense](https://medium.com/swlh/neurosymbolic-ai-to-give-us-machines-with-true-common-sense-9c133b78ab13)




### Logic Tensor Networks
Sony's Logic Tensor Networks (LTN) is a neurosymbolic framework that supports querying, learning, and reasoning with both rich data and abstract knowledge about the world. LTN introduces a fully differentiable logical language, called Real Logic, whereby the elements of a first-order logic signature are grounded onto data using neural computational graphs and first-order fuzzy logic semantics.




### Neural Logic Machines
Google's Neural Logic Machine (NLM) is a neural-symbolic architecture for both inductive learning and logic reasoning. NLMs use tensors to represent logic predicates.



### Hinge-Loss Markov Random Fields and Probabilistic Soft Logic 
Bach et al's Hinge-Loss Markov Random Fields (HL-MRFs) are a new kind of probabilistic graphical model that generalizes different approaches to convex inference. We unite three approaches from the randomized algorithms, probabilistic graphical models, and fuzzy logic communities, showing that all three lead to the same inference objective. We then define HL-MRFs by generalizing this unified objective. The second new formalism, probabilistic soft logic (PSL), is a probabilistic programming language that makes HL-MRFs easy to define using a syntax based on first-order logic.



### TensorLog
Cohen's TensorLog is a probabilistic deductive database in which reasoning uses a differentiable process. In TensorLog, each clause in a logical theory is first converted into certain type of factor graph. Then, for each type of query to the factor graph, the message-passing steps required to perform belief propagation (BP) are “unrolled” into a function, which is differentiable.


### Markov Logic Networks
Matthew Richardson's and Pedro Domingos' Markov Logic Networks (MLNs) are a first-order knowledge base with a weight attached to each formula (or clause). Together with a set of constants representing objects in the domain, it specifies a ground Markov network containing one feature for each possible grounding of a first-order formula in the KB, with the corresponding weight.
