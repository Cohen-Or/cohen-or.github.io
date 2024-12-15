---
layout: default
title: "Time Series Momentum"
date: 2024-12-15
---
# Time Series Momentum
The momentum effect in financial assets can be dated as far back as the Dutch merchant fleet in 1600s Amsterdam. The idea is very intuitive: assets that have performed well in the past tend to continue performing well in the future, while assets that have performed poorly tend to continue underperforming. 

On a high-level, the existence of momentum is supported by the foundational economics theory of demand-and-supply and is a manifestation of a persistent deviation from market equilibrium. More specifically, it can be attributed to five main causes according to [1]:
1. For futures, the persistence of roll returns, especially of their signs. 
2. The slow diffusion, analysis, and acceptance of new information.  
3. The forced sales or purchases of assets of various type of funds.  
4. Market manipulation by high-frequency traders.
5. Simultaneous triggering of trades based on stop orders. 

Time series momentum is identified by looking at the asset's price history, while cross-sectional momentum is identified by benchmarking the assets performance over other assets. Both can be identified over varying time horizons.

Like with any trading strategy, momentum strategies have their unique risks and limitations and overlooking them is a surefire way to losing money. The primary enemy of momentum is its counterpart effect, mean-reversion, since there is simply no guarantee that the trend will persist and not reverse sharply. Market conditions also play a crucial role since momentum performs well in trending markets but struggles in sideways or volatile markets. Finally, there are also the pitfalls common to all quantitative strategies such as overfitting and the low signal-to-noise ratio.

The researched of Moskowitz et al. [2] found that the the longer the time horizon the higher the likelihood of momentum to revert or decay. Shorter time horizons therefor result in higher Sharpe ratio at the expense of higher trading fees. With this is mind, let's explore a recently published HFT momentum strategy.

## Beat the Market
In this post we will explore the strategy published by Zarattini et al. [3] which mitigates the risks inherent to momentum trading in a fairly low-complexity fashion. I never shy away from sophisticated models yet I always favor a straightforward method over fancy-shmancy black-box solution.

We can break down this strategy to two parts: determining entry and exit criteria and bet-sizing.  In the rest of this post, we will implement this strategy using the ^QQQ historic OHLCV data with 5 minutes interval over the past 3 years.

### Distilling a signal from noisy data
 Starting with the entry entry and exit criteria, the strategy identifies abnormal trading activity by measuring the average absolute price movement from market Open to Close over a lookback period of 14 days. This archetype movement pattern from the market open is used to form an equilibrium zone defined by a Lower and Upper boundaries that is termed the Noise Area. Let's review it step by step.
 
 
**Step 1:** For each day \( t - i \) and time-of-day \( HH:MM \), calculate the absolute move from Open as: 

$$ 
\text{Move}_{t-i, 9:30-HH:MM} = \left| \frac{\text{Close}_{t-i, HH:MM}}{\text{Open}_{t-i, 9:30}} - 1 \right|, \quad \text{where } i = [1, 14]
$$

**Step 2:** For each time-of-day \( HH:MM \), calculate the average move over the last 14 days as: 

$$ 
\mu_{t, 9:30-HH:MM} = \frac{1}{14} \sum_{i=1}^{14} \text{Move}_{t-i, 9:30-HH:MM} 
$$

**Step 3:** For the *Upper Bound*, define the start point as the higher of today's Open or yesterday's Close (gap-up case) and for the *Lower Bound* as the lower of the two. Using this starting point, compute the Upper and Lower Boundary as:

$$ 
\text{UpperBound}_{t, HH:MM} = \max(\text{Open}_{t, 9:30}, \text{Close}_{t-1, 16:00}) \times \left( 1 + \mu_{t, 9:30-HH:MM} \right) 
$$ 

$$
\text{LowerBound}_{t, HH:MM} = \min(\text{Open}_{t, 9:30}, \text{Close}_{t-1, 16:00}) \times \left( 1 - \mu_{t, 9:30-HH:MM} \right) 
$$

**Step 4:** Compute the *Noise Area* as the area between the Upper and Lower Boundaries.  An entry signal is triggered when the price breaks the Noise Area in the corresponding direction (long for the Upper Bound and short for the Lower Bound). 

$$
\text{NoiseArea}_{t, HH:MM} = \left[ \text{LowerBound}_{t, HH:MM}, \text{UpperBound}_{t, HH:MM} \right]
$$

**Step 5:**
An exit signal is triggered when the price reverts back to the Noise Area or crosses the intraday VWAP. Otherwise, all positions are closed at the market Close.

$$ 
\text{VWAP} = \frac{\sum_{i=1}^n P_i \cdot V_i}{\sum_{i=1}^n V_i}
$$ 

$$ 
\text{Long TrailingStop}_{t, HH:MM} = \max(\text{UB}_{t, HH:MM}, \text{VWAP}_{t, HH:MM})
$$ 

$$
\text{Short TrailingStop}_{t, HH:MM} = \min(\text{LB}_{t, HH:MM}, \text{VWAP}_{t, HH:MM}) 
$$

In python, we implement this using Pandas library vectorized operations which allow efficient computation with fast execution. I chose to use log returns for  reasons that are well detailed [in this post.](https://gregorygundersen.com/blog/2022/02/06/log-returns/)

```python
days = pd.Series(data.index.date)

daily_grp = data.groupby(data.index.date, group_keys=False)

data['abs_ret'] = daily_grp['Open']
.apply(lambda x : (np.log(x) - np.log(x.iloc[0]))
.abs())

data['avg_ret'] = data
.groupby([data.index.hour, data.index.minute], 	
group_keys=False).apply(lambda x: x['abs_ret']
.rolling(14).mean())

data['open_t'] = days.map(daily_grp.Open.first()).values

data['close_tm1'] = days.map(daily_grp.Close.last()
.shift(1)).values

data['upper_bound'] = data[['close_tm1','open_t']]
.max(axis=1) * (1 + data['avg_ret'])

data['lower_bound'] = data[['close_tm1','open_t']]
.min(axis=1) * (1 - data['avg_ret'])

data['VWAP'] = daily_grp.apply(
lambda x: (x.loc[:,['High','Low','Close']]
.mean(axis=1) * x.Volume).cumsum()
 / x.Volume.cumsum())

# Entry signal
data['position'] = np.select(
[data.Close > data.upper_bound, 
data.Close < data.lower_bound], [1, -1], 		
default=np.nan)

data['position'] = data.groupby(data.index.date, 
group_keys=False).apply(lambda x: x['position']
.ffill())

# Exit signal
data['position'] = np.where(
(data.position ==1) & (data.Close <	
data[['upper_bound','VWAP']].max(axis=1)) |
(data.position == -1) & (data.Close > 	
data[['lower_bound','VWAP']].min(axis=1)),
0, data.position)

# Upon an exit signal trigger, close the position for rest of the day 
zffill = lambda s: s * (1 - (s == 0).cummax())

data['position'] = daily_grp['position'].apply(zffill).fillna(0)
```
![Example of a short trade executed by the model](/images/tsm1.png)
![Example of a long trade and following VWAP exit signal](/images/tsm2.png)

### Bet sizing
As mentioned before, volatile market conditions diminish the potential for benefiting from the momentum effect. Therefor it is reasonable to adjust the exposure of a momentum trading strategy dynamically based on the current market volatility regime. To that end, the authors propose the following sizing methodology that targets a daily market volatility of 2%:

We'll add that to our implementation:
```python
data['exposure'] = days.map(daily_grp['Close'].last()
.apply(np.log).diff().rolling(14).std()
.apply(lambda x: min(0.02/x, 4))).values
```

___
References:
7. Chan, E. (2013). Algorithmic Trading, Wiley.
8. Moskowitz, T. (2012).  Time series momentum, Journal of Financial Economics
9. Zarattini, C. (2024). Beat the Market, Swiss Finance Institute.
11. Gray, W. (2016). Quantitative Momentum.
