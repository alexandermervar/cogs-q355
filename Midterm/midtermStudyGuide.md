# COGS-Q 355 Midterm Study Guide
## Alexander Mervar 2.28.2022

### Week 1
**Brain Diagram**
[Use this link for brain diagrams](https://github.com/alexandermervar/cogs-q355/blob/master/Midterm/Midterm%20Slides/1%20-%20Intro.pdf)

### Week 2
**Explain what is long-term potentiation, and how does it happen biochemically?**
- In neuroscience, long-term potentiation (LTP) is a persistent strengthening of synapses based on recent patterns of activity. These are patterns of synaptic activity that produce a long-lasting increase in signal transmission between two neurons. The opposite of LTP is long-term depression, which produces a long-lasting decrease in synaptic strength.
It is one of several phenomena underlying synaptic plasticity, the ability of chemical synapses to change their strength. As memories are thought to be encoded by modification of synaptic strength, LTP is widely considered one of the major cellular mechanisms that underlies learning and memory.

**In the Hodgkin Huxley equations, give the one equation for total current flow across the cell membrane, and explain what each variable and term means.**
$$
I_k = g_k (V-E_k)
$$

**What is an integrate and fire model neuron?  In qualitative terms, how does it work?**
- An integrate an fire fire neuron is a model to see if an action potential happened or not. There is little detail on how the action potential came about. We check to see if the neuronal membrane reaches $V_thresh$.

**What is a rate coded model?**
- A rate coded model measures the frequency of action potential generation.

### Week 3
**What does it mean to say that a matrix is not invertible?**
To be invertible, the matrix must be
- NxN in dimension(s)
- The given matrix A must be row equivalent to $I_n$

**What is the L1 norm?  L2 norm?**
- L1 Norm: City Block Norm (The direction in the x direction + the direction in the y direction)
- L2 Norm: Euclidean Norm ($a^2 + b^2 = c^2$ where c equals the euclidean norm)

**What is a matrix transpose?**
- Matrix transposition is when every row in the matrix becomes a column. (I.e. $R_1$ becomes $C_1$ and so forth...)

**Know the formula relating the dot product to the cosine of the angle between two vectors and how to use it**
- $Angle = arcCos dotProduct / (The product of the squared sums of the individual vectors) [see link](https://github.com/alexandermervar/cogs-q355/blob/master/Assignments/Assignment%203/3%20-%20Linear%20Algebra.ipynb)

**What are eigenvectors and eigenvalues useful for?**
- Eigenvectors are the direction along which the data has maximum variance
- Eigenvalues can help find eigenvectors

**What is the difference between a McCullouch-Pitts neuron and a Perceptron?**
- McC&P has all or none inputs (0 or 1). A perceptron has graded inputs with weighted output units. So, some input units are more influential than others on the output units.


### Week 4
**What is a sensory homunculus?**
- A cortical homunculus is a distorted representation of the human body, based on a neurological "map" of the areas and proportions of the human brain dedicated to processing motor functions, or sensory functions, for different parts of the body. [see slide 5](https://github.com/alexandermervar/cogs-q355/blob/master/Midterm/Midterm%20Slides/6%20-%20Plasticity.pdf)

**Describe three experiments in monkeys that demonstrate cortical plasticity**
- Experiment 1: If a nerve is cut, the dorsal surface of areas still receiving a signal will expand to fill unused space
- Experiment 2: If digit is amputated, the nearby digits' cortical area expands to fill unused space
- Experiment 3: If a digit is over stimulated, that cortical area for that digit expands.

**What is unsupervised learning?**
- Without knowing what the outputs should be for a network, unsupervised learning models are able to cluster data using different algorithms/methods.

**Name three algorithms/methods of unsupervised learning**
- K means
- Autoencoders
- Blind Source Separation

**Describe the k means algorithm**
- Start with N training vectors and K desired clusters. Then, assign each point to the nearest K cluster centroid. Then, recompute the cluster centroid as the mean of all vectors nearest the cluster centroid.

### Week 5
**What is supervised learning?**
- You know the desired values of the output units, so you compute the error term and update the network to better produce the desired outputs using a test set.

**What does the XOR problem reveal about supervised learning methods?**
- There is no linear classifier that works for the XOR problem

**What is the delta rule?**
- 

**Consider a 2-layer perceptron, with a sum-squared error loss function of E=sum($e^2$), where e=desired-actual outputs.  Using the chain rule from calculus, and assuming gradient descent with learning rate alpha, derive the delta rule for learning.**
- 

**Describe three nonlinear signal functions. What are the advantages and disadvantages of each?**
- 

**What is the danger of having too many hidden units in a multi-layer perceptron?**
- 

**What is the advantage of having more than three layers in a deep learning network?**
- 

**What is the advantage of using a cross-entropy loss function instead of a sum squared error loss function?**
- 

**The modified learning rule “Adam” incorporates two modifications to the usual gradient descent rule. Describe them and why they are useful.**
- 

### Week 6


