
[Or Cohen](/index.html)

# Profit Sensitive ML Credit Risk Models - Part 1

In this post, we will explore how tailoring the learning objectives of machine learning algorithms can improve credit risk predictions.

A statistics riddle I like goes as follows: If a doctor can detect a rare disease with 99% accuracy, is that good enough? The short answer is: it depends. For example, if the disease affects only 1% of the population, the doctor could achieve 99% accuracy simply by diagnosing everyone as healthy—without actually detecting a single case of the disease.

Formally, classification problems in which there is a disproportionate ratio of observations in each class are referred to as **imbalanced classification**. In machine learning, imbalanced classification is a common issue that can hinder model performance and lead to suboptimal results. Well-known examples of imbalanced classification in the financial field include insurance models, fraud detection, and credit risk models.

Credit risk models are used by banks and non-banking financial institutions to assess the creditworthiness of borrowers. These models play a critical role in banking operations, making their optimization essential. The challenge arises because most borrowers repay their loans, making the detection of risky borrowers a classic case of an imbalanced classification task. Before diving into how to address this challenge, it is essential to understand the underlying reasons.

### The Role of Credit in Modern Economies

Credit plays a crucial role in modern economies by facilitating transactions, enabling economic growth, and stimulating investment. It relies on trust and responsibility from both borrowers and lenders, forming the foundation of prosperous economies and societies. While its basic principles have remained the same for years, the digital era is reshaping the dynamics of credit.

Technological advancements and the credit crunch following the global financial crisis have given rise to peer-to-peer (P2P) lending. P2P platforms such as Lending Club and Prosper enable individuals to obtain loans directly from other individuals, bypassing traditional banks. P2P lending benefits borrowers by providing credit at lower rates than traditional options, while investors often see returns that match or surpass those of other fixed-income investments. Additionally, P2P lending extends credit access to individuals typically excluded from conventional financial services.

However, disintermediation of the credit system also amplifies inherent risks. Credit risks, known since biblical times, include excessive borrower indebtedness, high lender costs (usury), and financial instability. A well-functioning credit system ensures borrowers and lenders have sufficient information to make informed decisions and engage in responsible lending practices.

It is the lender's responsibility to scrutinize borrower information, analyze credit histories, and manage defaults or late payments effectively. This is a challenging task even for large banks, but machine learning has demonstrated its effectiveness in measuring and managing credit risk despite the imbalanced classification issue.

### Overcoming the Challenge of Imbalanced Classification

A disproportionate ratio of observations in classification tasks often arises from two main factors: (1) biased data sampling or collection errors, or (2) a naturally imbalanced distribution of classes in the real world. In the latter case, the class with fewer samples is referred to as the **minority class**.

Most machine learning algorithms for supervised classification assume a balanced class distribution. Consequently, training models for imbalanced classification requires specialized techniques, which generally fall into two categories:

-   **Sampling methods:** These involve either increasing the representation of the minority class (over-sampling), decreasing the representation of the majority class (under-sampling), or using a combination of both.
    
-   **Cost-sensitive methods:** These techniques adjust the cost of misclassification, making the model more sensitive to the minority class by penalizing false positives and false negatives differently.
    

### Not All Defaults Are Created Equal

At first glance, developing a classification model for estimating a loan's probability of default may seem like a straightforward binary classification task. However, professional risk management requires a more nuanced approach. Some borrowers default earlier than others, causing greater losses. Additionally, in unsecured loans (such as those issued by P2P platforms), collection agencies may recover part of the unpaid balance. As a result, most banks and financial institutions adopt **risk-based pricing**.

To illustrate risk-based pricing with a simple example, consider a bank that offers only 1-year bullet loans, where the principal and interest are repaid in full at the end of one year. If the probability of default (**PD**) is 50%, the bank must charge a 100% interest rate to break even—since half of the borrowers will repay, while the other half will default, the bank needs to recover twice the loaned amount from those who do pay.

For amortized loans, additional factors must be considered, such as the **exposure at default (EAD)** (the outstanding principal at the time of default) and the **loss given default (LGD)** (the expected loss in case of default). These three factors—PD, EAD, and LGD—form the basis for calculating the default premium in risk-based pricing. According to the Basel II framework, the default premium is computed as:

 $$
Default Premium = Loss Given Default × Probability of Default
 $$
 
While default risk is a crucial factor in pricing, lenders must also account for **prepayment risk**. When borrowers prepay their loans, origination costs can consume a significant portion of interest income. Additionally, if interest rates decline, prepayment reduces income because old loans are replaced with new loans at lower rates. Thus, prepayment risk is an essential consideration in risk-based pricing.

## Designing Profit-Maximizing ML Models

Many excellent case studies demonstrate methods for improving binary classification performance in credit risk modeling. Most focus on optimizing statistical measures such as **Precision, Recall, F1 Score,** or the **area under the curve of the receiver operating characteristic (AUC ROC)**. While these metrics are useful, they are insufficient for a lending business focused on profit maximization. Since binary classification models ignore the profit or loss associated with each loan, they inherently favor lower-risk, lower-profit loans.

In practice, both classification errors carry costs:

-   **False positives (rejecting good loans):** Result in lost revenue opportunities.
    
-   **False negatives (approving bad loans):** Lead to varying financial losses depending on default severity.
    

In profit-sensitive models, profit and cost considerations can be incorporated **directly or indirectly**. Following the terminology introduced by Xia et al. [1]:

-   In a **direct approach**, profit and cost values are introduced into the classifier to guide its learning process.
    
-   In an **indirect approach**, these values influence other modeling steps, such as pre-processing (e.g., resampling techniques), model tuning (e.g., optimizing hyperparameters for expected profit), or decision-making (e.g., adjusting classification thresholds).
    

Petrides et al. [2] evaluated various cost-sensitive learning methods that account for variable misclassification costs by weighting records accordingly. Their study on a Romanian non-banking financial institution (NBFI) showed that cost-sensitive models improved profitability across three business channels, with single-digit gains for two channels and double-digit gains for the third.

Ariza-Garzón et al. [3] proposed incorporating profit and loss information throughout the underwriting process by:

1.  Using expected profit as an objective function in hyperparameter optimization.
    
2.  Integrating expected profit directly into XGBoost's learning process.
    
3.  Adjusting decision thresholds based on profit-sensitive criteria.
    
Other studies have explored profit scoring methods, separating profitability assessment from risk evaluation.

## Midpoint Conclusion

The consumer credit market is a fundamental component of modern economies and has evolved significantly due to P2P lending platforms. Behind the scenes, sophisticated machine learning models have helped investors manage the heightened risks of unsecured lending, enabling better rates for riskier borrowers. By integrating cost-sensitive techniques, probabilistic classification models can be enhanced to consider the financial impact of classification errors.

Next, we’ll move from theory to practice—building a cost-sensitive model using real-world Lending Club data to optimize loan profitability.

### References:
___
1. Xia, Y. (2017). Cost-sensitive boosted tree for loan evaluation in peer-to-peer lending, _Electronic Commerce Research and Applications_.
2. Petrides, G. (2022). Cost-sensitive ensemble learning: a unifying framework, _Data Mining and Knowledge Discovery_.
3. Ariza-Garzón, M. (2024). Profit-sensitive machine learning classification with explanations in credit risk: The case of small businesses in peer-to-peer lending, _European Cooperation in Science and Technology_.

