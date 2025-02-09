[Or Cohen](/index.html)

# Let The Best Ideas Win

In this post we will explore various approaches to generate alpha from an open source data of asset managers holdings - the SEC 13F form.

Winston Churchill once said that "One man with conviction will overwhelm a hundred who have only opinions". Academic studies that examined the performance of securities that were relatively overweighted (ex-ante)  in portfolios of professional investment managers found that they outperformed other securities in the same portfolio as well as the market. On a high-level, their conclusion was that high conviction of investment managers correlates with superior performance. 

The underlying idea is pretty intuitive; you have probably met an investment advisor or manager that had a deep knowledge and understanding of one or several companies that they recommended. In practice though, most professional investors choose to invest in a larger set of stocks for the sake of diversification. The reasons behind the decision to diversify range from regulatory requirements to reducing volatility and the need to deploy more capital.

Since 1978 all institutions that hold greater than $100 million in securities under discretionary management are required to file their holdings in US-listed stocks with the Securities and Exchange Commission (SEC) no later than 45 days after the end of each calendar quarter under a form known as ‘13F’. This data is made publicly available by the SEC to promote transparency in the asset management industry.

The financial press as well as investment professionals and amateurs review the information found in the 13F forms, often focusing on big name portfolios such as Warren Buffet's Berkshire Heathway. While a naive copycat of a famous institutional portfolio is ethically flawed and prone to failure, a meticulous and comprehensive analysis of this data can provide valuable insights for investment selection, even after the 45 days of delayed disclosure.

## Brief Literature Review

The seminal study by Cohen, Polk, and Antón [1] found that the stocks in which active mutual fund or hedge fund managers display the most conviction towards ex-ante, their "best ideas", outperform the market, as well as other stocks in those managers’ portfolios, by approximately 2.8 to 4.5 percent per year, depending on the benchmark employed. They also found that there is little overlap in the best ideas of managers. 

A study by Jian et al. [2] analyzed a comprehensive sample of active US equity funds between 1984-2008,  and found that stocks heavily overweighted by active managers outperform their underweighted counterparts by more than 7% per year, after adjustments for their loadings on the market, size, value, and momentum factors.

To tackle the question of the informative value in the delayed data, they evaluated the performance of an un-levered strategy that buys the stocks that active funds overweight and shorts the stocks that they underweight, implemented with a one-month lag, and found it generates an equal-weight four-factor alpha of 3.4% per year. The same strategy, implemented with a lag of two months, generates a four-factor alpha of 2.3%.

Other studies [3] focused solely on hedge funds using curated dataset from a proprietary data provider and found a statistically significant yet small positive five-factor alpha. Their methodology comprised overlying conviction (i.e. estimated relative overweight) with consensus (the prevalence of a the same stock across several hedge funds portfolios).

## The Value of Active Management (a.k.a Stock Picking)

Before we move on to the practical aspects let's address the basic question: is discretionary stock picking necessary? We know that in past decades passive index funds and algorithmic trading have gradually taken over the investment management industry and now command a large share of the assets under management (AUM). A 2024 study by the CFA institute [4] found that index funds AUM has seen a 1,500-fold increase between 1989-2021, but still represented only 32% of all fund AUM. However, when we zoom in on equity funds in the US Large Cap Blend category, index funds contain three times as much AUM as their active counterparts.

Proponents of passive investing will rightly point out to the now common wisdom that active mutual fund managers do not deliver consistent alpha that can offset the higher fees associated with their services. On the other hand, simply following a cap-weighted index erodes price discovery and overall market efficiency, and increase the concentration on high market-cap stocks since most passive indexes are cap-weighted. 

It's a kind of game theory problem in which the individual investor rational decision to pursue only a diversified return (beta) via an index fund, under the assumption that the market is efficient and equity prices reflect all information, eventually leads to concentration, inefficiency and possibly a lower beta return (the last claim is hard to empirically prove). 

In addition to the pervious broad argument in favor of active management, even from the investor standpoint, direct investing in equities with a separate managed account (SMA) can also prove more economically beneficial considering the opportunity for tax loss harvesting on declining stocks. This tax saving potential is wasted when the investor holds just a few index funds in her SMA. Admittedly, pursuing this entails more involvement in monitoring the portfolio as well as higher transaction costs due to more frequent rebalancing. 

## From Theory to Practice 

Now that we have the theoretical basis covered, we can move on to plan the analytical framework that will ensure we distill high quality insights from the raw data. Since the SEC don't verify or guarantee the accuracy of the data submitted, the first step in the process is the due diligence which involves parsing and validating all the information we intend to use. 

One common error by the funds that is important to be aware of is the over-report of the values of their holdings by a factor of 1,000 (i.e. overlooking the SEC specifications). Cross validation with an external data provider is the best way to ensure the holdings value is correct. [Bloomberg’s OpenFIGI](https://www.openfigi.com/) mapping API is useful for converting the CUSIP numbers that are used in the 13F reports to tickers.

```python
import json
import requests
import time
  
OPEN_FIGI_KEY = "YOUR_API_KEY"
OPEN_FIGI_API = "https://api.openfigi.com/v3/mapping"
HEADERS = {"X-OPENFIGI-APIKEY": "{}".format(OPEN_FIGI_KEY)}
  
def openfigi_cusips_mapper(cusips:list, id_type:str):
	'''
	Splits the openFIGI API calls to lots of 100 to fit the limit 
	Returns a list of tuples containing CUSIP and Ticker.
	type = "ID_CINS" or "ID_CUSIP"
	'''
	cusips_lots = [cusips[x:x+100] for x in range(0, len(cusips), 100)]
	mappings = []
	for index, lot in enumerate(cusips_lots):
		try:
			query = [{"idType": id_type, "idValue": str(cusip)} for cusip in lot]
			open_figi_resp = json.loads(requests.post(
			OPEN_FIGI_API, json=query, headers=HEADERS).text
			)
			tickers = []
			for resp in open_figi_resp:
				if resp.get("warning"):
				tickers.append('')
				else:
				tickers.append(resp["data"][0].get('ticker'))
				mappings.append(list(zip(lot, tickers)))
		except:
			print(f"Exception caught on lot: {index}")
		time.sleep(5)
	mappings = [item for sublist in mappings for item in sublist]
	missing_cusip_mappings = [item for item in mappings if item[1] == '']
	cusip_mappings = [item for item in mappings if item[1] != '']
	return cusip_mappings, missing_cusip_mappings
```

In addition to cross validation, it's also worth generating descriptive statistics and plots that can help us to detect outliers. With cleaned data at hand we can consider the analytical part. There are two main decisions to consider: (1) do we want to include all institutional investment managers or a selected subset and (2) how do we classify a certain holding as overweight. These two decisions will shape the rest of the process and determine the quality of our outcomes.

### Selecting Institutional Investment Managers 

While there is some truth in the disclaimer that past performance is not indicative of future results, in reality, professional investment managers who demonstrate good judgment, ethical conduct and a structured approach are more likely to deliver consistent positive results. Therefor, an evaluation of institutional investment managers requires a combination of quantitative and qualitative factors with equal parts. 

The quantitive elements to consider are relatively straightforward. The qualitative side though is less obvious and requires a deep understanding of the industry. For example, a leadership change or high churn of key investment professionals in an asset management firm will lead to loss of investment expertise and continuity. On the other hand, a culture that promotes independent thinking and original research will lead to a long term focus and stability.

Morningstar expert research have us covered on that end; every year, Morningstar publishes an extensive report that analyzes the largest 150 fund families in the US, called the [Fund Family Digest](https://www.morningstar.com/lp/fund-family-150). This report offers a comprehensive analysis and benchmarking of each institutional investment management firm and also includes for the majority of firms a detailed review by a dedicated analyst. Two additional pieces of information that are useful for our analysis are the share of active management and share of equity investments of the total $AUM a.

I chose to filter institutional investment managers that are predominantly engaged in active management (80% and above) with strong equity focus (at least 50% to equity or allocation funds) and Average or above Morningstar rating. In a few cases I also filtered out firms that met the criteria mentioned above but the analyst's review raised some red flags such as recent SEC charges or high churn. In a CSV file that is available on my Github you can find the [2025 list of instituional investment managers I used](https://github.com/Cohen-Or/cohen-or.github.io/blob/main/13F_analysis/SelectedManagers.csv) based on the report published at the end of 2024. 

### Classifying Overweight

The studies surveyed above chose to compare each stock's reported weight with the same stock's weight in the fund's benchmark or with a weight derived from a formula based on the assumption that fund managers capital allocation is aimed to maximize Sharp ratio. Here I propose a slightly different approach that takes into account the typical firm structure that drives the decision making process behind the scenes.

If we examine the typical institutional investment management firm we will find research departments that are divided to teams of analysts that specialize in a certain sector, and in some cases analyst that focus on an industry or geography within a sector. In a different department, portfolio managers that are dedicated to a fund with a specified objective such as a Value style or Mid-Cap stocks will use the outputs of the research department to construct their portfolios. 

Considering that typical firm structure, we can safely assume that if, for example, a Mid-Cap energy company is highlighted by the energy research team, the weight that will be allocated to this stock out of the overall Energy sector allocation by that firm will be higher, indicating a stronger conviction _at the firm level_. In this way we take out the sector tilt that may arise due to style (Growth vs. Value) or the firm's sector allocation decisions and extract a robust conviction indicator.

The constituents of the 11 MSCI Global World Sector Indexes and their weights can be found on the [index provider website](https://www.msci.com/constituents). Naturally, the number of stocks a single institutional investment manager holds in each sector is smaller than the number of stocks in the respective sector index. To adjust for this concentration bias we need to scale the manager's reported weights by dividing them by the following adjustment factor:

$$
\text{Adjustment Factor} = \frac{N_{\text{index}}}{N_{\text{portfolio}}}
$$

Where:

-   $$N_{\text{index}}$$​ is the number of stocks in the sector index.
-  $$N_{\text{portfolio}}$$ is the number of stocks in the asset manager's portfolio for that sector.

### Extending the Analysis

There is a myriad of statistical measures and ways to explore and learn from the dataset we curated by following the steps above. Some examples of questions we can answer are:
 - What stocks are most overweight by managers?
 - How many managers increased/decreased their position in the overweight stocks?
 - What is the average 'overweight tenure' (i.e. number of consecutive quarters the position is overweight) of each stock.
 - Who are the favorites of specialized managers? Specialized managers can be identified by a high concentration in a specific sector. For example, following its rapid rise Nvidia's portfolio of public investments surpassed $100M hence required the company to disclose its holdings. 
 
## Conclusion

Coming up with good investment ideas requires looking ahead to the future and forming an holistic opinion on the prospects of a company. The set of skills and creative thinking that are required to succeed in that task are likely to remain the domain of professional investment managers. 

Conviction by investment managers has been proven to be a strong indicator of a higher probability of future outperformance. Finding the best ideas of professional managers and integrating this information in our portfolio construction framework can greatly add to our overall performance.  

 In my [twitter (X) account](https://x.com/OrCohen29219725) you will find the top insights I discovered following the recipe depicted in this post. As always, please don't hesitate to send me your suggestions, comments and any other feedback you have. Thank you for reading!

___
References:
1. Antón, M (2010). Best Ideas, _Harvard Business School_.
2. Jian, H (2013). Information Content when Mutual Funds Deviate from Benchmarks, _American Finance Association 2012 meeting papers_.
3. Angelini, L (2019). Systematic 13F Hedge Fund Alpha, _WRDS-SSRN_.
4. Reinganum, M (2024). Beyond Active and Passive Investing: The Customization of Finance, _CFA Institute Research Foundation_
