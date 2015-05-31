---
layout: post
title: Notes on machine learning theory
---

<h1>{{ page.title }}</h1>
<p class="meta">Apirl 30, 2015 - Shang Hai</p>

> The purposes of writing this article are 

> - Understand the key points of machine learning theory.
> - More important is to record the process of learning new stuff, which on the other hands could be recalled or referenced later on to reveal how human beings master new knowledge gradually, compared with how machine could potentially follow the similar learning procedure.

I am going to start with my overall feeling about machine learning or AI, and then my understand of machine learning theory from Andrew’s lecture, and finally end up with my thoughts.

In my feeling now, machine learning or artificial intelligent are the words that are over-described, and current methods or theory are all targeting to solve very specific problems, which have following characteristics,

- There is no such concrete mathematics model that can perfectly solve the issue.
- The issue itself can’t be totally random without any pattern.

So, it is actually far away from what people usually expect this discipline to achieve,
 
- Behaves “intelligent” like human beings.
- Can evolve itself beyond pre-established routines.

Roman isn’t built in one day. The real “intelligent” needs interdisciplinary study involving such as computer science, neuroscience, linguistics, philosophy, math, etc. Machine learning is just the subfield that explores the construction and study of algorithms that can “learn” from sample data and make predictions on new data. Though various machine learning methods can be used to solve lots of real worlds problems smartly, they don’t attempt to simulate how human beings think, instead they are just well designed. So, let’s see how is the theory that can build a “smart” machine on given data set as first step to AI. 

I think the process of coming up with a good learning strategy for machine is kind of similar as how human beings discover a natural law which all aims to solve some issues. The process could be considered roughly as,

- Gather a group of data that is related to the given issue, which is usually called as sample data.
- Do a series of experiments, and find patterns exhibited by the data.
- Summarize the pattern in the form of mathematics language.
The difference is that machine is just a tool that “learns” following human beings instructions, on the contrary, human beings are self-driven and curious about unknowns and can ask questions which lead to the process of finding solutions.

Then machine learning theory is the theory that aims to choose the best learning strategy based on the sample data set for a given issue. There are a few terminologies,

- **Hypothesis**, my understanding of a hypothesis is a fixed function that takes new data as input and makes prediction or gives answer as output. Such as a linear model with fixed parameters (a hypothesis) can be the best choice for a linear classifier issue.
- **Hypothesis set**, a group of hypothesis (either finite or infinite) from which the best hypothesis could be chosen. I think usually the group of hypothesis has the same definition, such as a group of linear classifiers or a group of neural networks. But it is not necessary if one learning problem is hard to identify which category could do better job, and then can think them all as a larger group of hypothesis to be chosen from.
- **Generalization error** of a hypothesis is defined as the probability that if I have a new example from the problem, this hypothesis will misclassify it. E.g., a chosen hypothesis will have 10 percent chance to make a mistake when applying it to do prediction, and this 10 percent is just the generalization error of this hypothesis. There are other two factors that are important to evaluate a hypothesis,
  - **Bias**, is defined to be the expected generalization error. If bias is big even if there is a very large training set to fit to the hypothesis, then this hypothesis fails to capture the structure exhibited by the data (this is also called under-fit.)
  - **Variance**, literally it can be thought as the variance of the predicted value or class. Most of the times, it is not a good thing even if it can do a good job on training data set, it very likely generate large generation error on new data (this is also called over-fit.)

Quote from Andrew’s lecture, “there is a tradeoff between bias and variance. If our model is too “simple” and has very few parameters, then it may have large bias (but small variance); if it is too “complex” and has very many parameters, then it may suffer from large variance (but have smaller bias)”.

One thing has to be noticed is that probability theory takes very important role in machine learning. Just thinking the real world cases that input data set from a solvable problem almost always have noise or too many random factors involved and probably follows into a distribution. There are two distributions considered by the theory, 

- One is for the data set, that we can think they are drawn IID (independent and identically distributed) from some probability distribution D. (true for both training set and predicted data)
- The other is for the misclassification of the hypothesis that will follow Bernoulli distribution. 

Then, this can gives guarantee on generalization error of a given hypothesis in two aspects,

- The training error is a reliable estimation of the generalization error of the hypothesis.
- The training error is an upper-bound on the generalization error of the hypothesis.

Coming here, I just feel amazing that I even don’t know what the definition of the hypothesis is, I could already tell how well this hypothesis can do the prediction as long as I have “enough” training set to “try” this hypothesis. The following lemma 2 explains the theory for this, and combine with the lemma 1, we can see how to pick up the best hypothesis out of a group. (Following two lemmas are quote from Andrew’s lecture),

> **Lemma 1**, the union bound. Let A1, A2, . . . Ak be k different events (that may not be independent). Then,
> 
> `P(A1 ∪···∪Ak) ≤ P(A1)+...+P(Ak)`
> 
> In probability theory, the union bound is usually stated as an axiom (and thus we won’t try to prove it), but it also makes intuitive sense: The probability of any one of k events happening is at most the sums of the probabilities of the k different events.
> 
> **Lemma 2**, Hoeffding inequality. Let Z1, . . . , Zm be m independent and identically distributed (IID) random variables drawn from a Bernoulli(φ) distribution. I.e., P(Zi =1)=φ, and P(Zi =0)=1−φ. Let φˆ=(1/m) Sum(Zi) (i from 1 to m) be the mean of these random variables, and let any γ > 0 be fixed. Then,
> 
> `P (|φ − φˆ| > γ) ≤ 2 exp(−2γ*γ*m)`
> 
> This lemma (which in learning theory is also called the Chernoff bound) says that if we take φˆ—the average of m Bernoulli (φ) random variables—to be our estimate of φ, then the probability of our being far from the true value is small, so long as m is large. Another way of saying this is that if you have a biased coin whose chance of landing on heads is φ, then if you toss it m times and calculate the fraction of times that it came up heads, that will be a good estimate of φ with high probability (if m is large).

By means of these two lemmas, the mysterious veil of the machine learning theory will be uncovered. Again, let me restate what problem this theory want to address. It is that “give a learning problem and a group of pre-defined hypothesis, how to make sure the picked hypothesis can do the best job”. Well, there are two cases, (I only state the results and some of my understanding, but leave the details out of the writing.)
For the case of finite hypothesis set, the theory is in the formula of, 

`P(∀h∈H.|ε(hi)−εˆ(hi)|≤γ) ≥ 1 − 2k exp(−2γ^2m)`

There are three quantities of interest here: m, γ and P, which are the size of training set, the far-away between training error and generalization error, and the probability of this far-away being small respectively.

There are three outcomes or statements,

- With probability at least 1−2k exp(−2γ2m), we have that the generalization error will be within γ of training error for all hypothesis in the group. This is called uniform convergence result. 
- Then, “given γ and the probability (1 – δ) (let δ = 2k exp(−2γ2m)), how large must m be before we can guarantee that with probability at least 1 − δ, training error will be within γ of generalization error?” The answer is `m ≥ (1/2γ*γ)log(2k/ δ)`This is really the power of the theory that “This bound tells us how many training examples we need in order to make a guarantee. The training set size m that a certain method or algorithm requires in order to achieve a certain level of performance is also called the algorithm’s sample complexity. The key property of the bound above is that the number of training examples needed to make this guarantee is only logarithmic in k, the number of hypotheses.”
- A theorem. Let the size of hypothesis is k, and let any m, δ be fixed. Then with probability at least 1 − δ, we have that `ε(hˆ) ≤ min ε(h) + 2sqrt(1/2m * log (2k/ δ))` The beauty of this formula quantifies bias/variance tradeoff in model selection. “Specifically, suppose we have some hypothesis class H, and are considering switching to some much larger hypothesis class H′⊇ H. If we switch to H′, then the first term min ε(h) can only decrease (since we’d then be taking a min over a larger set of functions). Hence, by learning using a larger hypothesis class, our “bias” can only decrease. However, if k increases, then the second 2sqrt term would also increase. This increase corresponds to the “variance” increasing when a larger hypothesis class is used.”

For the case of infinite hypothesis set, it has very similar result as the finite one. Usually, many hypothesis classes contain an infinite number of candidates, such as any real number parameterized hypothesis, they have infinite functions fixed by its parameters. The interesting thing is that we use computer to represent real numbers, which use 64 bits to represent a double precision floating point number. So, it is actually not infinite that totally consist of at most k = the 64 power of 2 different hypothesis. Then, we can just substitute this to the previous inequation so as to get 

`m ≥ O(1/ γ2 log(2^64*d/ δ)) = O(d/ γ2 log(1/ δ)) = O(d)`

Thus, this indicates that the number of training examples needed is at most linear in the parameters of the model.

Besides, one thing worth note is that all the above results are proved for the learning strategy that use empirical risk minimization (ERM). This is the most “basic” learning strategy and very intuitive that the hypothesis is picked with minimum training error. (There are non-ERM learning strategy, but good theory for this is still an area of active research. I may learn and dig into this later.)

At last, let me wrap all these up a little bit by three aspects: the issue, the theory and my thought.

**The issues** are often those that need human intelligence involved, such as recognize objects or make a decision, not how a planet orbit around a star, in other words, they are not like the issues that can be solved by fixed functions or algorithm. Usually, there are many random factors that can’t be described by exact mathematics model. So, probability theory takes important role in machine learning.

**The theory** is just what I talked above, which it can guide in the following steps to solve a given learning issue,

- Start with using ERM learning strategy.
- Determine the hypothesis set, and usually it is infinite and parameterized. 
- Decide the size of training set by given fixed γ and δ, and then prepare a good training set.
- Pick the only one hypothesis that has minimum training error. 

This is too high level, and I know there are many different hypothesis classes, such as linear classifier, neural network, SVM, or perceptron, etc. that are worth to understand in details so as to know which class is better for different issues. Besides, to pick the best hypothesis is not just “pick” one by one, especially if the set is infinite, it is not possible to check each of them. Instead, there are many mathematics optimization techniques that can be used to “pick” the one quickly, such as least square or gradient descent. Overall, theory is good that it make things clear, but a good understanding has to be obtained by working on real project about machine learning.

**My expectation** of the strong artificial intelligence is that it doesn’t have to be physically like human beings, but have to pass Turing test that when I talk to “it”, I almost can’t distinguish whether I talk to a machine or a human beings. I think this is the long term goal, and we are far-far away from it (even maybe it never be realized). The short term goal to make machine “intelligent” enough so as to assist human beings to increase productivity I think is more feasible and important with practical significance. 

As I can see there are no big breakthroughs towards the long term goal since AI research. Because, there are four key characteristics that human beings have but a machine doesn’t,

1. Human beings are always curious, and can ask questions to unknown. 
2. Human beings can do experiments, and find natural laws.
3. Human beings can induce and deduce from basic laws, and then can solve a class of similar issues.
4. Human beings can evolve, and all good things are encoded into its DNA.

**Closing words** at last, making machine more clever is the trend and inevitable, because kinds of machines are already part of our life.

----------

Update on 5/31/2015, [Large Scale Deep Learning](http://static.googleusercontent.com/media/research.google.com/en/us/people/jeff/CIKM-keynote-Nov2014.pdf), a good summary of latest progress of machine learning from Google, which proved that Deep Neural Network is an important tool in building intelligent system. 
It is really exciting to see such solid and encouraging improvement and progress. When having time, it is definitely worth to know all the details about DNN, but before that let me just quote the conclusion from the article, 

> Deep neural networks are very effective for wide range of tasks, 
> 
>  - By using parallelism, we can quickly train very large and effective deep neural models on very large datasets.
>  - Automatically build high-level representations to solve desired tasks.
>  - By using embeddings, can work with sparse data.
>  - Effective in many domains: speech, vision, language modelling, user prediction, language understanding, translation, advertising, ...


