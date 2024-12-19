[Or Cohen](/index.html)
# Cross Sectional Momentum and Learning to Rank with LambdaMART
In this post we will learn how algorithms that were originally developed to rank results from a search query can help us distinguish the winners and losers in our universe of investment assets. 

There's the old joke that when your taxi driver tell you to buy a stock, you know it's time to sell. It points out to the fact that the diffusion-of and investors' response to new information often lags significant price moves. This in turn leas to auto-correlation of assets' returns and the persistence of a trend, be it up or down.

The momentum effect in financial assets can be dated as far back as the Dutch merchant fleet in 1600s Amsterdam. Its existence is supported by the foundational economics theory of demand-and-supply and is a manifestation of a persistent deviation from market equilibrium. 

> "An object in motion tends to stay in motion, an object at rest tends to remain at rest." – Sir Isaac Newton

Time series momentum (TSM) is identified by looking at the asset's price history, while cross-sectional momentum (CSM) is identified by benchmarking the assets performance over similar assets. Both can be observed over varying time horizons and both successfully capture risk premia, contrary to the efficient market hypothesis.

A major advantage of the CSM strategy is the insulation against the common market moves via hedging: by buying the assets with the highest expected returns and selling the assets with lowest, the effect of the broad market movements known as beta is largely canceled out. Accurately ranking the assets is naturally the key to improved trading performance.

Like any trading strategy, momentum strategies have unique risks and overlooking them is a surefire way to losing money. The primary enemy of momentum is its counterpart effect, mean-reversion, since there is simply no guarantee that the trend will persist and not reverse sharply. Market conditions also play a crucial role since momentum performs well in trending markets but struggles in sideways or volatile markets. 

## Learning to Rank and LambdaMART

 The rising amounts of digital data and the need to effectively search through it have led to significant progress in a family of algorithms in the field of Information Retrieval that are known as Learning to Rank or machine-learned ranking (MLR). These algorithms provide in response to a query a ranking of the results based on some metric of relevance.   
![Learning to Rank](https://www.elastic.co/guide/en/elasticsearch/reference/current/images/search/learning-to-rank-overview.png)

**LambdaMART** is a state-of-the-art MLR model developed by Christopher J.C. Burges and his colleagues at Microsoft Research[2]. Under the hood, the ranking task is transformed into a pairwise classification or regression problem. That means you look at pairs of items at a time, come up with the optimal ordering for that pair of items, and then use it to come up with the final ranking for all the results. 

Like a good yogi, LambdaMART is both powerful and flexible, making it the go-to choice for a wide range of problems. For example, in the case of a search use-case we can optimize for precision in order to get only the most relevant results. In the case of CSM, we are interested in better predictions across the entire spectrum so maximizing a ranking-specific evaluation metric, such as **NDCG** (Normalized Discounted Cumulative Gain) will be a better fit.

**MART** stands for Multiple Additive Regression Trees which means the algorithm uses gradient boosting with decision tress. Boosting trees are built iteratively, where each tree corrects the errors of the previous ones with the goal of optimizing ranking metrics like NDCG or Mean Reciprocal Rank (MRR).

## Leveraging MLR for CSM
The advantage of these new MLR techniques in CSM and other financial applications stems from the fact that they circumvent the need to use a regression or classification loss functions and address the ranking quality directly. In another post we will explore how training neural networks with a PnL or Sharpe loss functions results in superior performance, much like in the ranking case.

The research by Poh et. al [3] introduces a novel framework for the use of MLR in CSM and demonstrated the dramatic improvements in performance over other techniques of ranking momentum in financial assets. 

The proposed framework used momentum indicators that were derived from series of historic daily prices from the past 3 to 12-months periods to predict the one-month ahead winners and losers. In this post we will implement and explain the various parts of that framework. 

### CTA Momentum Indicator
The proposed framework in [3] incorporated a sophisticated version of the MACD momentum indicator termed CTA Momentum that was originally introduced in [4]. 

This indicator is aimed at measuring the strength of a trend based on the premise that (a) volatility diminishes the significance of the trend and hence the MACD needs to be volatility-normalized and (b) that different assets exhibit different trend patterns across time and hence these trends need to be evaluated based on a standard score. If you got this, the rest is just fancy math. 

The algorithm for obtaining this signal is the following (as detailed in [4]):

*Step 1:* Select 3 sets of time-scales, with each set consisting of a short and a long exponentially weighted moving average (EWMA). The authors chose $$S_k = (8, 16, 32)$$ and $$L_k = (24, 48, 96).$$  Those numbers are not look-back days or half-life numbers. In fact, each number (let’s call it n) translates to a lambda decay factor ($$\lambda$$) to plug into the standard definition of an EWMA. The half-life (HL) is then given by:

$$ 
HL = \frac{\log(0.5)}{\log(\lambda)} = \frac{\log(0.5)}{\log\left(1 - \frac{1}{n}\right)} ​
$$

*Step 2:*  For each k=1,2,3 calculate:

$$
x_k = EWMA[P|S_k] - EWMA[P|L_k] 
$$

*Step 3:* We normalize with a moving standard deviation as a measure of the realized 3-months normal volatility (PW=63):

$$
y_k = \frac{x_k}{Run.StDev[P|PW]} 
$$

*Step 4:* We normalize this series with its realized annual standard deviation (SW=252):
    
$$
z_k = \frac{y_k}{Run.StDev[y_k|SW]} 
$$

*Step 5* We calculate an intermediate signal for each k=1,2,3 via a response function R:

   $$ 
 \begin{cases} u_k = R(z_k) \\ R(x) = \frac{x\exp(-\frac{x^2}{2})}{0.89} \end{cases} 
 $$

*Step 6:* The final CTA momentum signal is the weighted sum of the intermediate signals (here we have chosen equal weights $w_k = \frac{1}{3}$​):

$$
S_{CTA} = \sum_{k=1}^3 w_k u_k ​
$$

We can implement it in Python using Pandas broadcasting and vectorization methods which allow efficient computation with fast execution as follows:
```python
hl = lambda decay : np.log(0.5) / np.log(1-1/decay)
macd = lambda S, L: data.Close.ewm(halflife=hl(S)).mean() - 
data.Close.ewm(halflife=hl(L)).mean()

def CTA_momentum(s,l):
	phi = []
	result = {}
	for i in zip(s,l):
		y_k = macd(i[0],i[1]) / data.Close.rolling(63).std()
		z_k = y_k / y_k.rolling(252).std()
		result[f'macd_norm_{i}'] = z_k
		u_k = z_k.apply(lambda x: x * np.exp(-(x**2)/4) / 0.89)
		phi.append(u_k)
	result['signal'] = pd.concat(phi ,axis=1).T.groupby(level=0).mean().T
	return pd.concat(result.values(), axis=1, keys=result.keys())
		.resample('ME').last().swaplevel(axis=1).stack(level=0)
``` 
### Model setup 
The model in [3] uses the CTA momentum signal as well as the normalized MACD signals (ie. the values of $$z_k$$) for the past 1, 3,6 and 12-months periods along as well as normalized and raw return returns for the past 3, 6 and 12-months periods. All together this results in a matrix of 22 features.
___
References:
2. Burges, C (2010). From RankNet to LambdaRank to LambdaMART, Microsoft Research.
3. Poh, D (2020). Building Cross-Sectional Systematic Strategies By Learning to Rank, Oxford-Man Institute of Quantitative Finance.
4. Baz, J. (2015). Dissecting Investment Strategies in the Cross Section and Time Series, SSRN Electronic Journal.
