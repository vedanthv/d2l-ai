## Chapter 1  : Introduction

Machine learning is the study of algorithms that can learn from experience. As a machine learning algorithm accumulates more experience, typically in the form of observational data or interactions with an environment, its performance improves.

Some Key Components of Machine Learning

The data that we can learn from.

A model of how to transform the data.

An objective function that quantifies how well (or badly) the model is doing.

An algorithm to adjust the model’s parameters to optimize the objective function.

**Objective Functions**

In order to develop a formal mathematical system of learning machines, we need to have formal measures of how good (or bad) our models are. In machine learning, and optimization more generally, we call these objective functions. By convention, we usually define objective functions so that lower is better. This is merely a convention. You can take any function for which higher is better, and turn it into a new function that is qualitatively identical but for which lower is better by flipping the sign. Because lower is better, these functions are sometimes called loss functions.


**Types of Learning**

1. Supervised Learning

<img src = "https://d2l.ai/_images/supervised-learning.svg">

1.1 Regression

What makes a problem a regression is actually the form of the target. Say that you are in the market for a new home. You might want to estimate the fair market value of a house, given some features like above. The data here might consist of historical home listings and the labels might be the observed sales prices. When labels take on arbitrary numerical values (even within some interval), we call this a regression problem. The goal is to produce a model whose predictions closely approximate the actual label values.

1.2 Classification

In classification, we want our model to look at features, e.g., the pixel values in an image, and then predict which category (sometimes called a class) among some discrete set of options, an example belongs. For handwritten digits, we might have ten classes, corresponding to the digits 0 through 9. 

The simplest form of classification is when there are only two classes, a problem which we call binary classification. 

1.3 Search

In the field of information retrieval, we often impose rankings over sets of items. Take web search for example. The goal is less to determine whether a particular page is relevant for a query, but rather, which, among a set of relevant results should be shown most prominently to a particular user. 

One possible solution might be to first assign a score to every element in the set and then to retrieve the top-rated elements. PageRank, the original secret sauce behind the Google search engine, was an early example of such a scoring system. Peculiarly, the scoring provided by PageRank did not depend on the actual query.

1.4 Recommendation Models

Recommender systems are another problem setting that is related to search and ranking. The problems are similar insofar as the goal is to display a set of relevant items to the user. The main difference is the emphasis on personalization to specific users in the context of recommender systems. 

For instance, for movie recommendations, the results page for a science fiction fan and the results page for a connoisseur of Peter Sellers comedies might differ significantly. Similar problems pop up in other recommendation settings, e.g., for retail products, music, and news recommendation.

1.5 Sequential Learning

These problems are among the most exciting applications of machine learning and they are instances of sequence learning. They require a model to either ingest sequences of inputs or to emit sequences of outputs (or both). 

Specifically, sequence-to-sequence learning considers problems where inputs and outputs both consist of variable-length sequences. Examples include machine translation and speech-to-text transcription. While it is impossible to consider all types of sequence transformations, the following special cases are worth mentioning.

Examples : 

1. Tagging and Parsing
2. Automatic Speech Recognition
3. Text to Speech
4. Machine Translation

**Unsupervised Learning**

1. Clustering : Can we find a small number of prototypes that accurately summarize the data? Given a set of photos, can we group them into landscape photos, pictures of dogs, babies, cats, and mountain peaks? Likewise, given a collection of users’ browsing activities, can we group them into users with similar behavior? This problem is typically known as clustering.

2. Can we find a small number of parameters that accurately capture the relevant properties of the data? The trajectories of a ball are well described by velocity, diameter, and mass of the ball. Tailors have developed a small number of parameters that describe human body shape fairly accurately for the purpose of fitting clothes.

3. Is there a description of the root causes of much of the data that we observe? For instance, if we have demographic data about house prices, pollution, crime, location, education, and salaries, can we discover how they are related simply based on empirical data? The fields concerned with causality and probabilistic graphical models tackle such questions.

4. Another important and exciting recent development in unsupervised learning is the advent of deep generative models. These models estimate the density of the data 
, either explicitly or implicitly. Once trained, we can use a generative model either to score examples according to how likely they are, or to sample synthetic examples from the learned distribution.

**Reinforcement Learning**

Reinforcement learning gives a very general statement of a problem, in which an agent interacts with an environment over a series of time steps. At each time step, the agent receives some observation from the environment and must choose an action that is subsequently transmitted back to the environment via some mechanism (sometimes called an actuator). Finally, the agent receives a reward from the environment.

The behaviour of an agent is governed by a policy.

Consider the game of chess. The only real reward signal comes at the end of the game when we either win, earning a reward of, say, 1, or when we lose, receiving a reward of, say, -1. So reinforcement learners must deal with the credit assignment problem: determining which actions to credit or blame for an outcome. 

The same goes for an employee who gets a promotion on October 11. That promotion likely reflects a large number of well-chosen actions over the previous year. Getting more promotions in the future requires figuring out what actions along the way led to the promotion.

## Chapter 2 : Preliminaries

**Broadcasting**

Broadcasting works according to the following two-step procedure: (i) expand one or both arrays by copying elements along axes with length 1 so that after this transformation, the two tensors have the same shape; (ii) perform an elementwise operation on the resulting arrays.

**Derivatives**

Put simply, a derivative is the rate of change in a function with respect to changes in its arguments. Derivatives can tell us how rapidly a loss function would increase or decrease were we to increase or decrease each parameter by an infinitesimally small amount.

**Partial Derivatives**

<img src = "D:\data-science\04-Deep Learning\D2L\partial_der.JPG">

**Chain Rule**

<img src = "D:\data-science\04-Deep Learning\D2L\chain.JPG">

**Jacobian**

<img src = "D:\data-science\04-Deep Learning\D2L\jacobian.JPG">

**Central Limit Theorum**

<img src = "https://d2l.ai/_images/output_probability_245b7d_78_0.svg">

In general, for xaverages of repeated events (like coin tosses), as the number of repetitions grows, our estimates are guaranteed to converge to the true underlying probabilities. The mathematical proof of this phenomenon is called the law of large numbers and the central limit theorem tells us that in many situations, as the sample size grows, these errors should go down at a rate of. Let’s get some more intuition by studying how our estimate evolves as we grow the number of tosses from 1 to 10000.

**Bayes Theorum**

![bayes](https://user-images.githubusercontent.com/44313631/217043576-c4a4ef49-7ea3-4187-acfe-6635cb58f906.JPG)

**Expectation**

![expectation](https://user-images.githubusercontent.com/44313631/217043619-b084f9f1-3378-4a8a-b21b-99fd5921b884.JPG)

**Standard Deviation**

![stddev](https://user-images.githubusercontent.com/44313631/217043651-dc6c6685-3423-473d-857e-f637967cdb7f.JPG)

**Covariance**

![covariance](https://user-images.githubusercontent.com/44313631/217043673-6c356d9e-c9e7-4bcd-8eb1-d691dba6e6f9.JPG)

### Chapter 3 : Linear Neural Networks

**Linear Regression Loss Function**

![linreg](https://user-images.githubusercontent.com/44313631/217043706-274063d4-0fc1-4bac-9faf-99424b180b26.JPG)

**Mini Batch Gradient Descent**

![minibatch](https://user-images.githubusercontent.com/44313631/217043739-096b05ca-c63a-44ae-889c-8f57bf1e34c5.JPG)

**Likelihood Estimation**

![likelihood](https://user-images.githubusercontent.com/44313631/217043776-1a9c4f6c-bbfc-44a8-aa3c-e1783bf78c4f.JPG)

