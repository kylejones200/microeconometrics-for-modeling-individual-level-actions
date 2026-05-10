---
author: "Kyle Jones"
date_published: "January 20, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/microeconometrics-for-modeling-individual-level-actions-fa5905da7137"
---

# Microeconometrics for modeling individual level actions Modern methods for studying economic decisions at the individual level

### Microeconometrics for modeling individual level actions
#### Modern methods for studying economic decisions at the individual level
Microeconometrics is a subfield of econometrics focused on the analysis of individual-level data. These data might come from households, firms, or individuals, and are often used to answer detailed economic questions about behavior, preferences, and decision-making.

Macroeconomics deals with aggregated data (e.g., GDP, inflation); microeconomics focuses on individual-level units, such as a person's income, a household's spending habits, or a firm's production decisions. This more focused lens lets us look at heterogeneity (how individuals differ in behavior or preferences), causal relationships (e.g., the effect of education on earnings), and granular policy insights.

### Software Tools for Microeconometrics
You need software to do microeconometric analysis. I generally use **Python** with`statsmodels`, `scikit-learn`, and `linearmodels` . But you can also use **R** with `plm`, `AER`, and `caret` . Or **Stata** which is often used for econometric modeling in academia and policy research. The examples below are all in Python.

### Linear Regression
Linear regression is the cornerstone of econometric analysis. It models the relationship between a dependent variable Y and one or more independent variables X.

Linear regression provides insights into how changes in X affect Y, holding other factors constant. For example, one might estimate the return to an additional year of education on wages.

We'll use a simple dataset to examine the relationship between years of education and wages.

```python
import pandas as pd
import statsmodels.api as sm

# Sample dataset
data = {
    "education": [10, 12, 14, 16, 18],
    "wages": [20, 25, 30, 35, 40]
}
df = pd.DataFrame(data)

# Define dependent and independent variables
X = sm.add_constant(df["education"])  # Add a constant for the intercept
Y = df["wages"]

# Fit the regression model
model = sm.OLS(Y, X).fit()

# Print summary
print(model.summary())
```


Note, I really like `statsmodels` for regression but you have to use make sure to add the constant for the intercept because it doesn't do that automatically.

### Panel Data Methods
Panel data consists of observations on the same entities (e.g., individuals, firms) over time. This structure allows for analysis of dynamics and unobserved heterogeneity. Two common models are:

- **Fixed Effects (FE):** Controls for time-invariant characteristics by differencing or demeaning.
- **Random Effects (RE):** Assumes that unobserved individual effects are uncorrelated with explanatory variables.

```python
from linearmodels.panel import PanelOLS
# Simulated panel data
data = {
    "entity": ["A", "A", "B", "B", "C", "C"],
    "time": [1, 2, 1, 2, 1, 2],
    "income": [50, 55, 60, 62, 45, 48],
    "education": [10, 11, 12, 13, 9, 10]
}
df = pd.DataFrame(data)
df.set_index(["entity", "time"], inplace=True)  # Set multi-level index
# Fixed effects model
mod = PanelOLS.from_formula("income ~ education + EntityEffects", data=df)
res = mod.fit()
# Print results
print(res)
```


### Instrumental Variables (IV)
When explanatory variables are endogenous (correlated with the error term), IV methods provide consistent estimators. An instrument Z must have **relevance** (Z is correlated with the endogenous variable X) and **exogeneity** (Z is uncorrelated with the error term ϵ).

IV methods are used to estimate causal effects, such as the impact of health insurance on medical expenses.

```python
"""
Instrumental Variables (IV) to address endogeneity
"""

from linearmodels.iv import IV2SLS
# Simulated dataset
data = {
    "education": [10, 12, 14, 16, 18],
    "ability": [1, 2, 3, 4, 5],  # Endogenous variable
    "instrument": [2, 3, 4, 5, 6],  # Instrument
    "wages": [20, 25, 30, 35, 40]
}
df = pd.DataFrame(data)
# Define the model
mod = IV2SLS.from_formula(
    "wages ~ 1 + [education ~ instrument]", data=df
)
res = mod.fit()
# Print results
print(res)
```


### Discrete Choice Models
Discrete choice models analyze decisions with categorical outcomes (e.g., buy/not buy, choose brand A/B/C). Two popular models are **Logit** and **Probit.**

```python
from statsmodels.discrete.discrete_model import Logit
# Simulated dataset
data = {
    "price": [1, 2, 3, 4, 5],
    "quality": [5, 4, 3, 2, 1],
    "purchase": [1, 1, 0, 0, 0]  # Binary outcome
}
df = pd.DataFrame(data)
# Define dependent and independent variables
X = sm.add_constant(df[["price", "quality"]])
Y = df["purchase"]
# Fit the logistic regression model
model = Logit(Y, X).fit()
# Print summary
print(model.summary())
```


### Models for Limited Dependent Variables
These models handle cases where the dependent variable has restrictions, such as being censored, truncated, or bounded. **Tobit Models** are used for censored data (e.g., income reported as ≥ \$100,000) and the **Heckman Selection Model** is used if you need to correct for selection bias when data is non-randomly selected.

```python
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
# Define the Tobit model
class Tobit(GenericLikelihoodModel):
    def loglike(self, params):
        beta = params[:-1]
        sigma = params[-1]
        X = self.exog
        y = self.endog
        c = (y == 0)
        z = (y - np.dot(X, beta)) / sigma
        ll = np.log(1 - c) * (-0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * z**2) + c * np.log(1 - sm.distributions.norm.cdf(z))
        return ll.sum()
# Simulated dataset
data = {
    "income": [0, 0, 10, 15, 20],
    "education": [8, 9, 10, 11, 12]
}
df = pd.DataFrame(data)
# Fit Tobit model
X = sm.add_constant(df["education"])
Y = df["income"]
model = Tobit(Y, X).fit()
# Print summary
print(model.summary())
```


### Practical Applications
Labor economists use these methods to understand what drives wages, employment patterns, and working hours. Health economists apply them to investigate how people use healthcare services and how policy changes affect health outcomes. These techniques also help industrial organization researchers examine how firms behave, compete, and set prices. Policymakers use microeconometrics to evaluate the effectiveness of government initiatives like tax credits and subsidy programs.

### So what?
Microeconometrics gives us tools to look at causal relationships and generate insights that drive decision-making.
