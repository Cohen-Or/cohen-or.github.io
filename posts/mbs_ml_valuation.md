[Or Cohen](/index.html)

# Pricing The Collateral: Machine Learning Cash Flows For MBS And CLO Valuation

<a target="_blank" href="https://colab.research.google.com/github/Cohen-Or/cohen-or.github.io/blob/0bcf0dd6f478d45afbff3e9a30013bb4a49d02ce/notebooks/mbs_ml_valuation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

In this post we will test whether a loan-level machine-learning model of prepayment and default, trained on public Fannie Mae data, projects pool cash flows and values an agency mortgage pool more accurately out of sample than the rate-incentive baseline a desk would otherwise use.

In 1943 the statistician Abraham Wald was asked where to armour bombers returning from missions. The engineers wanted to reinforce the wings, where the bullet holes clustered; Wald replied that the holes that mattered were on the planes that never came back. A seasoned mortgage pool is a set of survivors in the same sense: the borrowers keenest to refinance have already left, and a model that ignores this misprices the cash flows.

### Problem Context

Let's assume we advise a desk that is shown secondary-market paper backed by a pool of loans, an agency mortgage pass-through or a CLO tranche. The price is the discounted cash flow of the collateral, the sum of the loans' cash flows [1] (p.18), and the unknown inputs are how fast the loans prepay and how many default. The CLO market prices tranches on one scenario of constant annualised default, prepayment and recovery rates (CADR, CAPR and CRR) with cash flows from Intex [2] (p.3), and the deal-level data lives in Intex's paywalled feeds [3] (p.6). With no public loan-level history of CLO collateral, we demonstrate the machinery on agency mortgages, using the [Fannie Mae Single-Family Loan Performance Data](https://huggingface.co/datasets/yeigen/fannie-mae-loan-performance) (public, via an unofficial Hugging Face mirror); credit goes to Fannie Mae, whose terms forbid redistribution, so the notebook downloads at run time and nothing is re-hosted. One difference matters: under the agency guarantee a defaulted loan is bought out at par, so to the investor a default is a prepayment, and the credit leg only moves the price of non-agency and CLO paper.

Our design follows Sirignano, Sadhwani and Giesecke [1]: fit a nonlinear loan-level model of monthly transitions, compare it with a linear logistic on a time split, then sum loan-level probabilities to pool prepayment counts. Three inputs are proxied: zip-level house prices by the FHFA state index, local unemployment by FRED's state series, and the *rate incentive* (note rate minus the current Freddie Mac thirty-year rate) rebuilt from FRED's MORTGAGE30US. For simplicity, we will assume level-payment amortisation at each loan's note rate, a flat discount rate and covariates frozen at the projection date.

### From Loan-Months To Prepayment And Default Labels

We take thirty-year fixed-rate loans from the 2018Q1 and 2019Q1 acquisition cohorts, sample 120,000 and keep loan-months with loan age up to 60, which leaves 3,758,385 loan-months. A terminated loan's last row carries a zero-balance code, so the prepayment label is 1 when the next row is a voluntary payoff, and the default label is 1 when the next row is three or more months delinquent or a credit event. *Burnout* is the loan's cumulative positive incentive, lagged a month.

```python
g = panel.groupby("Loan_Identifier", sort=False)
panel["next_zb"] = g["zb"].shift(-1)
panel["next_dlq"] = g["dlq"].shift(-1)
panel["y_prepay"] = (panel["next_zb"] == "01").astype(int)
panel["y_default"] = ((panel["next_dlq"] >= 3) | panel["next_zb"].isin(CREDIT_CODES)).astype(int)
panel["incentive"] = panel["Original_Interest_Rate"] - panel["m30"]
panel["pos_inc"] = panel["incentive"].clip(lower=0)
panel["burnout"] = g["pos_inc"].cumsum() - panel["pos_inc"]   # lagged one month
```

The split is by time, as in the paper: train from 2018-01 to 2022-06 (3,314,111 loan-months), test from 2022-07 to 2025-03 (444,274). The figure shows why the split is unkind to every model: the cohorts refinanced through the 2020-21 rate trough, the *conditional prepayment rate (CPR)* peaking at 49.7% in 2021-02, then rates rose sharply and the pool locked in at a mean CPR of 7.5% over the test window. The monthly prepayment rate falls from 2.66% in training to 0.72% in the test window; the models learn on a refinancing wave and are graded on its absence.

![Cohort prepayment and delinquency rates with the thirty-year rate](/images/mbs_ml_valuation1.png)
Prepayment rate (annualised as CPR) and share of loans three or more months delinquent, with the Freddie Mac thirty-year rate; the dashed line starts the test window. Source: Fannie Mae, FRED.

### The Rate-Incentive Baseline

The desk baseline is the *S-curve*, prepayment as a function of the rate incentive alone, the kind of model that Chernov, Dunn and Longstaff note is reconciled with market prices only through an ad hoc option-adjusted spread [4] (p.1). We fit it as a logistic regression of next-month prepayment on incentive and loan age. In training the empirical curve has the familiar shape: flat out of the money, steep once the incentive turns positive, then bending over as burnout sets in. In the test window the median incentive is -1.61 percentage points, almost every loan is out of the money, and the S-curve is asked about a region where it has little to say.

![Empirical S-curve, train vs test, with the fitted logistic](/images/mbs_ml_valuation2.png)
Monthly prepayment rate by incentive bucket, training versus test window, with the fitted S-curve at two loan ages.

### A Loan-Level Gradient-Boosted Model Versus The Logistic Baseline

The nonlinear model is a histogram gradient-boosted classifier on the full covariate set (incentive, burnout, loan age, FICO, LTV, DTI, balance, state unemployment, house-price change, vintage, current delinquency, purpose, borrowers and note rate); the paper used a deep network, but the point is the interactions, not the architecture. It is compared with a logistic regression on the same covariates, the paper's linear model, and with the S-curve.

On one-month prepayment the out-of-sample *area under the ROC curve (AUC)* is 0.558 for the S-curve, 0.517 for the full logistic and 0.534 for gradient boosting: all close to a coin toss. Gradient boosting still edges the full logistic, but the two-variable S-curve beats both, and nothing here resembles the paper's lift from .65 to .74 (Table 19) [1]. Lock-in prepayments are what Chernov, Dunn and Longstaff call turnover, prepayments unrelated to the rate [4] (p.2), and the loan-level covariates barely see them. One-month default is the opposite problem: both models score 0.999 because the label is dominated by loans already two months late rolling into serious delinquency, and current delinquency is a feature. The credit-relevant comparison is the 12-month label, a first serious delinquency within a year from a current loan (0.5% of test loan-months), where gradient boosting wins clearly, 0.663 against 0.564, in the direction Kündig and Sigrist find on Freddie Mac data, though their gap is far smaller [5] (p.9).

![Out-of-sample ROC curves for prepayment and default](/images/mbs_ml_valuation3.png)
Out-of-sample ROC curves: 1-month prepayment, 1-month roll to serious delinquency, and 12-month default from current.

The partial dependence plots show what each model learned: the logistic probability climbs without limit in the incentive and falls with burnout, while the boosted model has a floor at negative incentives, flattens at high ones, and rises with burnout. Lesniewski shows that burnout needs no behavioural story: with heterogeneous hazards the fast prepayers leave first and the pool hazard declines by selection alone [6] (pp. 2, 5). A loan-level model conditions on the loan, not the pool, so it need not treat burnout as a brake.

![Partial dependence of prepayment probability on four covariates](/images/mbs_ml_valuation4.png)
Partial dependence of the predicted 1-month prepayment probability on incentive, burnout, loan age and FICO, both models, test-window loans.

### From Loans To Pools

The paper's pool test sums loan-level probabilities over pools of ten thousand loans and compares the expected one-year prepayment count with the realised one (Table 22) [1]. Only 30,430 of our loans are alive and performing at 2022-07, so the notebook falls back to 6 disjoint pools of 5,000 loans, 30,000 in all, a simplification forced by the sample. Each loan's monthly hazards are chained over 12 months with covariates frozen at 2022-07 (loan age advances, burnout accrues), the paper's simplest variant and a second simplification: part of any gap is scenario error, not model error. The expected count is the survival-weighted sum of monthly probabilities,

$$
\text{Expected Prepayments} = \sum_{\text{loans}} \sum_{h=1}^{12} S_{h-1}\, p_h, \qquad S_{h-1} = \prod_{k=1}^{h-1} (1 - p_k)
$$

```python
for name, (pp, pdf) in haz.items():
    S_prev = np.cumprod(1 - pp[:, :HORIZON], axis=1)
    S_prev = np.hstack([np.ones((len(pp), 1)), S_prev[:, :-1]])
    alive["exp_prepay_" + name] = (S_prev * pp[:, :HORIZON]).sum(axis=1)
    pool_counts[name] = alive.groupby("pool")["exp_prepay_" + name].sum().values
```

The realised count averages 429.2 prepayments per pool; the S-curve predicts 957.9, the boosted model 769.6 and the full logistic 353.7, for mean absolute gaps of 528.7, 340.4 and 75.4 loans. In the paper the linear model was the one biased upward, since every pool's prediction rode on the same covariates, and the network's average gap was a fraction of it (Table 22) [1]. Here the S-curve and the boosted model both overshoot by a wide margin, and the plain logistic, which undershoots by 75.4 loans on average, is nearest the diagonal: with the incentive frozen at its July 2022 value, the boosted model's floor at negative incentives becomes a floor on the pool's speed.

![Predicted versus realised 12-month prepayment counts per pool](/images/mbs_ml_valuation5.png)
Predicted against realised 12-month prepayment counts for the 6 test pools of 5,000 loans, with the diagonal; the full logistic's gap is in the title.

### From Pool Cash Flows To A Price

To turn counts into a price we treat the pools as a *pass-through*: the investor receives scheduled principal plus interest at the note rate less 75 bp of servicing and guarantee fee, and the remaining balance at par when the loan terminates. Because of the guarantee, the termination hazard is the sum of each model's prepayment and default probabilities; in the realised path a loan terminates at its payoff, its credit event or its first serious delinquency. Cash flows run over the 32 months of the test window, the balance outstanding at 2025-03 is taken at par under every path, and price is the present value per hundred of July 2022 balance at a flat 5.41%, the Freddie Mac thirty-year rate of that month.

```python
dfac = 1 / (1 + disc_rate / 1200.0) ** np.arange(1, H_MAX + 1)
def price(cf, terminal, idx):
    pv = (cf[idx] * dfac).sum() + terminal[idx].sum() * dfac[-1]
    return 100 * pv / bal_t0[idx].sum()
```

The pool's weighted average coupon is 4.76%, so after the 75 bp fee the net coupon sits below the 5.41% discount rate: this is a discount pool, and faster termination raises the price. Over the first 12 months the realised terminations are 10.24% of the starting balance; the S-curve projects 19.11%, the boosted model 15.83% and the full logistic 9.19%. The prices follow: 96.97 realised, 97.32 for the S-curve, 97.2 for gradient boosting and 96.99 for the logistic, gaps of 0.35, 0.23 and 0.02 points. The gaps are small because the pool is priced close to par: returning par sooner is worth something, but only a fraction of a point on a pool a few points below par, of the same order as the bid-ask-sized fitting errors Chernov, Dunn and Longstaff report on the TBA coupon stack [4] (p.15).

![Projected versus realised pool cash flows and implied price](/images/mbs_ml_valuation6.png)
Left: projected and realised monthly cash flow per hundred of July 2022 balance, all pools combined. Right: pass-through price per hundred at 5.41% and the gap to realised, in points.

## Conclusion

The answer to our question is mixed. **On this public sample, in a lock-in window, the loan-level machine-learning model did not beat the desk baseline on prepayment, beat the linear model clearly on 12-month default, and for an agency discount pool no model's error moved the price by more than 0.35 points.** The paper's headline result did not survive a regime it never saw: trained on the refinancing wave, the boosted model carried a floor on prepayment speed into a market with no incentive to refinance, and the linear model, whose prepayment probability keeps falling as the incentive falls, came closest. What public data cannot show is the leg where the credit model would earn its keep: a non-agency deal or a CLO, where a default is a loss rather than a par prepayment, and where the loan-level data sits behind Intex.

I'd love to hear your thoughts, suggestions, and ideas. Thanks for reading!

### References

1. Sirignano, J., Sadhwani, A. & Giesecke, K. (2018). _Deep Learning for Mortgage Risk_. arXiv 1607.02470v2.
2. Li, Y. & Zheng, Z. (2010). _A Top-down Model for Cash CLO_. arXiv 1004.2865.
3. Cordell, L., Roberts, M. R. & Schwert, M. (2021). _CLO Performance_. NBER Working Paper 29410.
4. Chernov, M., Dunn, B. R. & Longstaff, F. A. (2016). _Macroeconomic-Driven Prepayment Risk and the Valuation of Mortgage-Backed Securities_. NBER Working Paper 22096.
5. Kündig, P. & Sigrist, F. (2024). _A Spatio-Temporal Machine Learning Model for Mortgage Credit Risk: Default Probabilities and Loan Portfolios_. arXiv 2410.02846v3.
6. Lesniewski, A. (2026). _Mortgage Burnout and Selection Effects in Heterogeneous Cox Hazard Models_. arXiv 2603.12422v3.
