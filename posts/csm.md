[Or Cohen](/index.html)
# Cross Sectional Momentum and Learning to Rank with LambdaMART
In this post we will learn how algorithms that were originally developed to rank results from a search query can help us distinguish the winners and losers in our universe of investment assets. 

There's the old joke that when your taxi driver tells you to buy a stock, you know it's time to sell. It points out to the fact that the diffusion-of, and investors' response-to new information often lags significant price moves. This rippled response in turn leads to auto-correlation of assets' returns and the persistence of a trend, be it up or down.

The momentum effect in financial assets can be dated as far back as the Dutch merchant fleet in 1600s Amsterdam. Its existence is supported by the foundational economics theory of demand-and-supply and is a manifestation of a persistent deviation from market equilibrium. 

> "An object in motion tends to stay in motion, an object at rest tends to remain at rest." – Sir Isaac Newton

Time series momentum (TSM) is identified by looking at the asset's price history, while cross-sectional momentum (CSM) is identified by benchmarking the assets performance over similar assets. Both can be observed over varying time horizons and both successfully capture risk premia, contrary to the efficient market hypothesis.

A major advantage of the CSM strategy is the insulation against the common market moves via hedging: by buying the assets with the highest expected returns and selling the assets with lowest, the effect of the broad market movements known as beta is largely canceled out. Accurately ranking the assets is naturally the key to improved trading performance.

Like any trading strategy, momentum strategies have unique risks and overlooking them is a surefire way to losing money. The primary enemy of momentum is its counterpart effect, mean-reversion, since there is simply no guarantee that the trend will persist and not reverse sharply. Market conditions also play a crucial role since momentum performs well in trending markets but struggles in sideways or volatile markets. 

## Learning to Rank and LambdaMART

 The ever rising amounts of digital data and the need to effectively search through it have led to significant progress in a family of algorithms in the domain of Information Retrieval that are known as Learning to Rank or **machine-learned ranking (MLR)**. These algorithms provide in response to a query a ranking of the results based on some metric of relevance.   
![Learning to Rank](https://www.elastic.co/guide/en/elasticsearch/reference/current/images/search/learning-to-rank-overview.png)

**LambdaMART** is a state-of-the-art MLR model developed by Christopher J.C. Burges and his colleagues at Microsoft Research[1]. Under the hood, the ranking task is transformed into a pairwise classification or regression problem. That means you look at pairs of items at a time, come up with the optimal ordering for that pair of items, and then use it to come up with the final ranking for all the results. 

Like a good yogi, LambdaMART is both powerful and flexible, making it the go-to choice for a wide range of problems. For example, for ranked search results we can optimize it for precision in order to get only the most relevant results. In the case of CSM, we are interested in better predictions across the entire spectrum so maximizing a ranking-specific evaluation metric, such as **NDCG** (Normalized Discounted Cumulative Gain) will be a better fit.

**MART** stands for Multiple Additive Regression Trees which means the algorithm uses gradient boosting with decision tress. Boosting trees are built iteratively, where each tree corrects the errors of the previous ones with the goal of optimizing ranking metrics like NDCG or Mean Reciprocal Rank (MRR).

## Leveraging MLR for CSM
The advantage of these new MLR techniques in CSM and other financial applications stems from the fact that they circumvent the need to use a regression or classification loss functions and address the ranking quality directly. 

The research by Poh et. al [2] introduces a novel framework for using MLR to construct CSM portfolios and demonstrated the dramatic improvements in performance over other techniques of ranking momentum in financial assets. 

The proposed framework uses momentum indicators that are derived from historic daily prices of the past 3 to 12-months periods. These predictors are used to predict the one-month ahead winners and losers. Among them on stands out as a sophisticated version of the MACD momentum indicator termed CTA Momentum that was originally introduced in [3]. 

### CTA Momentum Indicator
This indicator is aimed at measuring the strength of a trend based on the premise that (a) volatility diminishes the significance of a trend and (b) that different assets exhibit different trend patterns across time and hence a good indicator needs to reflect a standard score. If you got this, the rest is just fancy math. 

The algorithm for obtaining this signal is the following (as detailed in [3]):

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

*Step 6:* The final CTA momentum signal is the weighted sum of the intermediate signals (here we have chosen equal weights $$w_k = \frac{1}{3}$$​):

$$
S_{CTA} = \sum_{k=1}^3 w_k u_k ​
$$

We can implement it in Python using *Pandas broadcasting and vectorization* methods which allow efficient computation with fast execution as follows:
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
The framework proposed in [2] uses the CTA momentum signal as well as the normalized MACD signals (ie. the values of $$z_k$$) for the past 1, 3, 6 and 12-months periods along as well as normalized and raw return returns for the past 3, 6 and 12-months periods. For the returns we will use the log returns for reasons that are explained  [in this post.](https://gregorygundersen.com/blog/2022/02/06/log-returns/)
```python
log_prices = data.Close.apply(np.log)
raw_ret = lambda n: log_prices.resample('ME').last().diff(n)
norm_ret = lambda n: log_prices.resample('ME').last().diff(n) / 
	log_prices.diff().ewm(span=n*21).std().resample('ME').last() * np.sqrt(12/n)
```
All together we get a matrix of 22 features for each stock and month. For the ranking objective the researchers chose to use risk adjusted return by scaling month-end returns using the ratio between target annual volatility (15%) and an exponentially weighted standard deviation with a 63-day span on daily returns. 

```python
risk_adj_ret = log_prices.resample('ME').last().diff(1) * 0.15 /
	log_prices.diff().ewm(span=63).std().resample('ME').last() * 2 
# Multiplty by sqrt(4)=2 to scale the 3-month volatiltiy to the annual target
```

We'll perform a cross validation to find the best set of hyper parameters among those proposed in the research.
```python
param_grid = {
	'eta': [1, 1e-1, 1e-2, 1e-3, 1e-3, 1e-5, 1e-6],
	'n_estimators': [5, 10, 20, 40, 80, 160, 320],
	'max_depth': [2, 4, 6, 8, 10]
	}
param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

for params in param_combinations:
cv_scores = []
# Create base ranker with fixed parameters
ranker = xgb.XGBRanker(
	objective='rank:pairwise',
	eval_metric='ndcg',
	tree_method='hist',
	eta=params['eta'],
	max_depth=params['max_depth'],
	n_estimators=params['n_estimators']
	)	
```

___
References:
2. Burges, C (2010). From RankNet to LambdaRank to LambdaMART, Microsoft Research.
3. Poh, D (2020). Building Cross-Sectional Systematic Strategies By Learning to Rank, Oxford-Man Institute of Quantitative Finance.
4. Baz, J. (2015). Dissecting Investment Strategies in the Cross Section and Time Series, SSRN Electronic Journal.
