[Or Cohen](/index.html)
# Change Point Detection with Statistical Jump Models
In this post we will explore the benefits of identifying regimes in financial markets and learn how to statistically classify and predict change points in regimes. 
 
The German philosopher Georg Hegel said, *"The only thing that we learn from history is that we learn nothing from history"*. A history of financial markets data can teach us a lot about common patterns that prevail in markets and help us better manage our risks and portfolios. 

Market regimes are clusters of persistent market conditions that affect the relevance of investment factors and performance of various asset classes. These regimes are usually a result of a change in the macroeconomic environment, e.g. change in inflation or unemployment levels.

Fundamental macro indicators are often discovered retrospectively while the dynamic nature of markets reflects them concurrently. Early change point detection (CPD) of market regime is a valuable input to strategic investment decisions such as asset allocation and the level of exposure to various factors.  However the stochastic nature of prices means that identifying a persistent change is a challenging task given the low signal to noise ratio.

## Demystifying The Black Box 
A scientific approach to asset management can help us uncover insights that are hard to find in financial data and also commit in advance to a disciplined decision making framework that will protect us from our biases. Notwithstanding experience, discretion and the  good old human judgement are irreplaceable. 

All the above introduction is to say that before we decide to include a scientific method to our toolbox we must consider our ability to both understand and evaluate the quality of its outcomes with expert knowledge. This is especially true in the case of CPD giving the pivotal (pun is intended) role it has in our strategic decision making. 

## Hidden Markov Models and Statistical Jump Models
We know that market regimes are patterns of persistent conditions that are not directly observable from the data. A lot of research was dedicated to the application of Hidden Markov Models (HMM) to CPD.

A Markov process is a random process in which the future is independent of the past, given the present. In the case of a discrete state the Markov processes are known as Markov chains. In an HMM, the probability distribution that generates an observation depends on the state of an unobserved Markov chain. 

The application of HMMs for CPD is prone to several errors that stem from the challenge in correctly specifying it and result in unstable and impersistent estimates of the underlying state sequence. Statistical Jump Model [1] (SJM) is a novel approach for the learning of HMMs and is particularly interesting due to its interpretability and ability to control the state transition rate via a regularization factor $$\lambda$$. 

The advantage of controlling the state transition rate when incorporating HMMs in our investment strategies lies in the need to balance the turnover rate in order to maximize the risk adjusted return and reduce the transactions costs associated with high turnover. In the absence of ground truth for the labeling of the hidden states, we can use our knowledge of investment strategies as a feedback for the model. 

An extension to the SJM learning approach in [1] is the Continuous Jump Model (CJM) that turns the discrete hidden state variable to a probability vector over all regimes. This provides a smooth transition between regimes that is both more robust and applicable for investment decision making.

![Continuous Statistical Jump Model](/images/sjm1.png)

The figure above charts the probability of bull and bear regimes (in yellow) of the Nasdaq Index (in blue) from 1996 to 2005 as estimated by the discrete SJM and the extended CJM. While the discrete model identifies the dot-com crash as a single bear period, the continuous one is able to detect two rebound periods.

## Integrating SJMs
Alongside their scientific contribution with the novel approach to fitting HMMs, the research teams behind the two models also made an admirable contribution to the open-source community by providing [a well documented python library](https://github.com/Yizhan-Oliver-Shu/jump-models?tab=readme-ov-file) of SJMs. 

Successful integration of the model lies in fine-tuning the jump penalty parameter $$\lambda$$ through cross-validation or a statistical criteria that (similar to a loss function).
