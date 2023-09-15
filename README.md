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

---
### [DeepProbLog](https://github.com/ML-KULeuven/deepproblog)

#### Description & Software Languages
- DeepProbLog is a library that extends ProbLog by integrating Probabilistic Logic Programming with deep learning, using neural predicates. 
- These neural predicates represent probabilistic facts parameterized by neural networks.
- The primary programming language is Python.

#### Features
- The library's key features include a combination of Probabilistic Logic Programming and deep learning, utilizing neural predicates for representing probabilistic facts and providing functionalities for both exact and approximate inference.

#### Docs & Tutorials Quality
- DeepProbLog's documentation primarily comes from in-code comments and the associated research papers. 
- The absence of detailed user-friendly tutorials or step-by-step guides might pose a challenge for new or inexperienced users. 

#### Popularity & Community
- stars: >100 

- forks: >30 
- contributors: 2 

#### Creation Date
- June 20, 2021

#### Maintenance
- Not Active

#### License
- Apache License 2.0

---
### [LNN](https://github.com/IBM/LNN)

#### Description & Software Languages
- Logical Neural Networks (LNNs) provide a framework that merges the learning capabilities of neural networks with the knowledge and reasoning power of symbolic logic. 
- This creates a robust and interpretable disentangled representation.
- The primary programming language is Python.

#### Features
- Meaningful neurons: Each neuron represents a component of a formula in a weighted real-valued logic for interpretable disentangled representation.
- Omnidirectional inference: The system does not focus on predefined target variables, providing the capacity for logical reasoning, including classical first-order logic theorem proving.
- End-to-end differentiability: The model is completely differentiable, minimizing a unique loss function that captures logical contradictions, which improves resilience to inconsistent knowledge.
- Open-world assumption: The system maintains bounds on truth values, allowing for probabilistic semantics and resilience to incomplete knowledge.

#### Docs & Tutorials Quality
- provides a variety of resources for users to learn about and understand the framework, including academic papers, getting started guides, API overviews, and specific modules on Neuro-Symbolic AI. 
- This comprehensive documentation could support a wide range of users, from beginners to advanced. 

#### Popularity & Community
- stars: >100 
- forks: >300  
- contributors: 10 

#### Creation Date
- Nov 21, 2021

#### Maintenance
- Not Active

#### License
- Apache License 2.0

---

### [LTN](https://github.com/logictensornetworks/logictensornetworks)

#### Description & Software Languages
- Logic Tensor Network (LTN) is a neurosymbolic framework that supports querying, learning, and reasoning with rich data and abstract knowledge about the world. 
- LTN uses Real Logic, a differentiable first-order logic language, to incorporate data and logic. 
- LTN converts Real Logic formulas into TensorFlow computational graphs.
- The primary programming language is Python.

#### Features
- Use of Real Logic: LTN uses a differentiable first-order logic language, enabling complex queries about data, constraints during learning, and logical proofs.
- Neural-symbolic Integration: Merges deep learning and reasoning, representing many-valued logics.
- Versatility: LTN can be applied to a range of tasks such as classification, regression, clustering, and link prediction.
- TensorFlow Integration: Real Logic sentences are represented as TensorFlow computational graphs, promoting efficient computation and compatibility with TensorFlow's ecosystem.

#### Docs & Tutorials Quality
- LTN provides extensive resources for learning, including a series of tutorials that cover various aspects of LTN (from grounding to learning and reasoning), a variety of examples demonstrating how to use LTN for different tasks, and jupyter notebooks to aid learning.
- It also includes a comprehensive collection of useful functions and a set of tests, which is beneficial for users seeking to understand the system more deeply or contribute to its development. 
- Overall, the documentation and tutorials are thorough and well-structured, aiding users from beginners to advanced in understanding and using LTN.


#### Popularity & Community
- stars: >200 
- forks: >40  
- contributors: 4  

#### Creation Date
- June 1, 2018

#### Maintenance
- Not Active

#### License
- MIT

---
### [NLM (Archived)](https://github.com/google/neural-logic-machines)

#### Description & Software Languages
- The Neural Logic Machine (NLM) is a neural-symbolic architecture that facilitates inductive learning and logical reasoning. - NLMs use tensors to represent logical predicates, which are then grounded as True or False over a fixed set of objects.
- Rules are implemented as neural operators that can be applied over the premise tensors to generate conclusion tensors.
- The primary programming language is Python.

#### Features
- Neural-Symbolic Integration: Combines the flexibility of neural networks with the reasoning capabilities of symbolic logic, facilitating both inductive learning and logic reasoning.
- Tensor-Based Predicate Representation: Encodes logical predicates as tensors, which are binary-valued (True or False) over a set of objects.
- Neural Operators: Implements rules as neural operators that transform premise tensors into conclusion tensors.
- Versatility: This repository contains multiple tasks related to graph-related reasoning (using supervised learning) and decision-making (using reinforcement learning).

#### Docs & Tutorials Quality
- The repository provides adequate information on installation and usage, including how to train and test the model using the included scripts. However, compared to the previous libraries, the documentation is somewhat less comprehensive, lacking in-depth tutorials or examples for users to follow along. 
- Nonetheless, for users familiar with neural-symbolic systems and with a solid grounding in PyTorch, the current documentation should suffice. It includes information on various command-line options, prerequisite packages, and references to the original research paper.

#### Popularity & Community
- stars: >200 
- forks: >50  
- contributors: 1 

#### Creation Date
- May 6, 2019

#### Maintenance
- Not Active

#### License
- MIT


--- 
### [NTP](https://github.com/uclmr/ntp) 

#### Description & Software Languages
- The End-to-End Differentiable Proving project is an implementation of the same-titled paper. 
- This high-level, highly-experimental research code is designed for reasoning and proving in a differentiable manner. 
- The system learns logical reasoning tasks by training end-to-end with a Prolog-like syntax.
- The primary programming language is newLISP.

#### Features
- The system includes the Neural Theorem Prover (NTP) for integrating symbolic logic into neural networks, end-to-end differentiable proving for logical reasoning, and a data format that follows Prolog syntax for easy representation of facts and rules.

#### Docs & Tutorials Quality
- The repository contains basic explanations and some instructions on how to run the code, however, it's explicitly marked as not well-documented and no longer maintained.
- The repository doesn't provide explicit installation instructions. Users may need to explore the code and its dependencies to determine the correct installation process.
- The main components of the project are located in the 'ntp' directory. Key modules include unify, tiling of batched representations, multiplexing, and the Kmax heuristic.
- Tests are included and can be run using a command-line tool `nosetests`

#### Popularity & Community
- stars: >200 
- forks: >10 
- contributors: 1  

#### Creation Date
- Feb 2, 2018

#### Maintenance
- Not active

#### License
- Apache License 2.0

---
### [PyNeuraLogic](https://github.com/LukasZahradnik/PyNeuraLogic)

#### Description & Software Languages
- PyNeuraLogic is a Python library that allows users to write Differentiable Logic Programs. 
- It utilizes its NeuraLogic backend to make the inference process of logic programming differentiable. 
- This provides an interface akin to deep learning where numeric parameters associated with rules can be learned, similar to learning weights in neural networks.
- The primary programming language is Python.

#### Features
- The library allows logic programming in Python, providing a declarative coding paradigm. 
- It also bridges the gap between logic programming and neural networks, offering the potential to create models with capabilities beyond common Graph Neural Networks (GNNs).
- Unlike other GNN frameworks, PyNeuraLogic is not limited to GNN models and allows users to design their own deep relational learning models. 
- PyNeuraLogic claims to provide superior performance on a range of common GNN models and applications.

#### Docs & Tutorials Quality
- PyNeuraLogic has an assortment of tutorial notebooks to help users get started with various tasks and applications. 
- It also links to several papers and articles for a more in-depth understanding of the concepts it employs. 

#### Popularity & Community
- stars: >200 
- forks: >10  
- contributors: 3  

#### Creation Date
- Dec 6, 2020

#### Maintenance
- Active

#### License
- MIT

---
### [TensorLog](https://github.com/TeamCohen/TensorLog)

#### Description & Software Languages
- TensorLog is a unique software library that combines the characteristics of probabilistic first-order logic with neural networks.
- In this library, queries are compiled into differentiable functions within a neural network infrastructure such as TensorFlow or Theano. 
- This integration of probabilistic logical reasoning with deep learning infrastructure allows for the tuning of the parameters of a probabilistic logic through deep learning frameworks. 
- TensorLog is designed to scale to large problems, handling hundreds of thousands of knowledge-base triples and tens of thousands of examples. 
- The primary programming language is Python.

#### Features
- TensorLog offers a unique integration of probabilistic first-order logic and deep learning infrastructure, allowing for the seamless use of high-performance deep learning frameworks for tuning probabilistic logic parameters.
- TensorLog is capable of scaling to large problems, handling hundreds of thousands of knowledge-base triples and tens of thousands of examples.
- The library allows queries to be compiled into differentiable functions, supporting integration with neural network infrastructures such as TensorFlow or Theano.

#### Docs & Tutorials Quality
- TensorLog provides a range of resources for learning more about the library, including papers, information about the TensorLog database, and details on running TensorLog programs and experiments. 
- The documentation also offers guides for using TensorLog with TensorFlow and running queries interactively from Python. 
- The 'further reading' resources, and the detailed guides on using TensorLog in various scenarios, suggest that the library is well-documented and beginner-friendly.

#### Popularity & Community
- stars: >100 
- forks: >20  
- contributors: 4 

#### Creation Date
- Feb 14, 2016

- #### Maintenance
Not active

#### License
Apache License 2.0



---
---
### [Logical Neural Network](https://github.com/IBM/LNN)
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
