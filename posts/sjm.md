[Or Cohen](/index.html)

# Change Point Detection with Statistical Jump Models
In this post we will explore the benefits of identifying regimes in financial markets and learn how to statistically classify and predict change points in regimes. 
 
The German philosopher Georg Hegel said, *"The only thing that we learn from history is that we learn nothing from history"*. The tendency of history to repeat itself is indeed a result of our limited ability to learn from it. One valuable insight we can gain from studying the history of financial markets is the prevalence of common patterns that repeat in the evolvement of asset prices.

Market regimes are clusters of persistent conditions that affect the performance of different asset classes and the relevance of investment factors. The change of these regimes is usually a result of a change in the macroeconomic environment, e.g. a change in inflation or unemployment levels.

Fundamental macro indicators are often discovered retrospectively while the dynamic nature of markets reflects the macro environment concurrently. Early change point detection (CPD) of market regime is a valuable input to strategic investment decisions such as asset allocation and the level of exposure to various investment factors.  However the stochastic nature of prices means that identifying a persistent change is a challenging task given the low signal to noise ratio.

## Demystifying The Black Box 
A scientific approach to asset management can help us uncover insights that are hard to find in financial data and also commit in advance to a disciplined decision making framework that will protect us from our biases. Notwithstanding experience, discretion and the good old human judgement are irreplaceable. 

All the above introduction is to say that _before we decide to include a scientific method in our toolbox we must **consider our ability to both understand and evaluate the quality of its outcomes with expert knowledge**_. This is especially true in the case of CPD giving the pivotal (pun is intended) role it has in our strategic decision making. 

## Hidden Markov Models and Statistical Jump Models
We know that market regimes are patterns of persistent conditions that are not directly observable from the data. A lot of research was dedicated to the application of Hidden Markov Models (HMM) to CPD.

A Markov process is a random process in which the future is independent of the past, given the present. In the case of a discrete state the Markov processes are known as Markov chains. In an HMM, the probability distribution that generates an observation depends on the state of an unobserved Markov chain. 

The application of HMMs for CPD is prone to several errors that stem from the challenge in correctly specifying it and result in unstable and impersistent estimates of the underlying state sequence. Statistical Jump Model [1] (SJM) is a novel approach for the learning of HMMs and is particularly interesting due to its interpretability and ability to control the state transition rate via a regularization factor λ. 

The advantage of controlling the state transition rate when incorporating HMMs in our investment strategies lies in the need to balance the turnover rate in order to maximize the risk adjusted return and reduce the transactions costs associated with high turnover. In the absence of ground truth for the labeling of the hidden states, we can use our knowledge of investment strategies as a feedback for the model. 

An extension to the SJM learning approach in [1] is the Continuous Jump Model (CJM) [2] that turns the discrete hidden state variable to a probability vector over all regimes. This provides a smooth transition between regimes that is both more robust and applicable for investment decision making.

![Continuous Statistical Jump Model](/images/sjm1.png)

The figure above charts the probability of bull and bear regimes (in yellow) of the Nasdaq Index (in blue) from 1996 to 2005 as estimated by the discrete SJM and the extended CJM. While the discrete model identifies the dot-com crash as a single bear period, the continuous one is able to detect two rebound periods with a smoothed transition.

## Tuning SJMs
Alongside their scientific contribution with the novel approach to fitting HMMs, the research teams behind the two models also contributed to the open-source community by providing [a well documented python library](https://github.com/Yizhan-Oliver-Shu/jump-models?tab=readme-ov-file) of SJMs. Successful integration of the model to portfolio and risk management lies in fine-tuning the jump penalty parameter λ through cross-validation or a statistical criteria (similar to a loss function).

```python
jump_penalty=50.
# initlalize a JM instance (set the arg cont=True for CJM)
jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False)
# fit data
jm.fit(X_train_processed, log_ret, sort_by="cumret")
# make online inference
labels_test_online = jm.predict_online(X_test_processed)
```
The hyperparameter λ serves as a control parameter for the fixed-cost regularization term associated with transitions between different states. Its value reflects our prior assumptions about the frequency of state transitions. 

When λ is set to zero, the jump model becomes the K-means algorithm which does not take the temporal order into account. As we increase the value of λ, the number of state transitions decreases. 

In the next post we will see how the insights gained from this model can greatly improve the Mean-Variance asset allocation method. As always, please don't hesitate to send any questions or suggestions you have. 
Thank you for reading!

___
References:
1.  Nystrup, P. (2020). Learning hidden Markov models with persistent states by penalizing jumps, Expert Systems with Applications.
2. Aydınhan, A. O. (2024). Identifying patterns in financial markets: Extending the statistical jump model for regime identification, Annals of Operations Research.
