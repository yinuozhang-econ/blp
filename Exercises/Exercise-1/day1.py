import pyblp
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import sys
import os
MODL = os.path.dirname(os.path.abspath(__file__)).split('Exercise-1')[0]
sys.path.append(MODL)
from utils import get_data

pyblp.options.digits = 3
pyblp.options.verbose = False
pd.options.display.precision = 3
pd.options.display.max_columns = 50

import IPython.display
IPython.display.display(IPython.display.HTML('<style>pre { white-space: pre !important; }</style>'))
df = get_data('products')
print(df.columns)

## define market size - potential # of servings sold in a market
df['market_size'] = df['city_population']*90
df['market_share'] = df['servings_sold']/df['market_size']
df['outside_share'] = 1 - df.groupby(['market'])['market_share'].transform('sum')

## Estimate pure logit 
df['logit_delta'] = np.log(df['market_share']/df['outside_share'])
model = smf.ols(formula = 'logit_delta ~ mushy + price_per_serving', data = df)
# use robust standard error
result = model.fit(cov_type = 'HC0')
print(result.summary())
# williness to pay for (more) mushy: -0.0748/(-7.4801)

## play w/ PyBLP 
df_blp = df.rename(columns = {'market': 'market_ids', 'product': 'product_ids', 'market_share': 'shares', 'price_per_serving': 'prices'})
df_blp['demand_instruments0'] = df_blp['prices']
ols_problem = pyblp.Problem(pyblp.Formulation('1 + mushy + prices'), df_blp)
print(ols_problem)
ols_results = ols_problem.solve(method='1s')
print(ols_results)

## absorb market and product fixed effects
ols_problem = pyblp.Problem(pyblp.Formulation('prices', absorb='C(market_ids) + C(product_ids)'), df_blp)
print(ols_problem)
ols_results = ols_problem.solve(method='1s')
print(ols_results) # result suggests that price is positively correlated w/ unobserved quality, because using fixed effects reduces the price coefficients. Suppose organic is unobserved, increases utility and it is positively correlated with price. It is included in product_id f.e but not in mushy. W/o including it, price coefficient actually gets attenuated which is exactly what we see here.  

## add a price instrument
# check first stage in statsmodel
first_stage = smf.ols(formula = 'prices ~ price_instrument + C(market_ids) + C(product_ids)', data = df_blp)
# use robust standard error
first_stage_result = first_stage.fit(cov_type = 'HC0')
print(first_stage_result.summary()) # R-squared pretty high
# instrument using pyBLP
df_blp['demand_instruments0'] = df_blp['price_instrument']
iv_problem = pyblp.Problem(pyblp.Formulation('prices', absorb='C(market_ids) + C(product_ids)'), df_blp)
print(iv_problem)
iv_results = iv_problem.solve(method='1s')
print(iv_results) # price coefficient went from -28.6 to -30.6 once IV for prices, suggesting price is positively correlated with unobserved difference in market-product level. Suppose in NY there is a cereal movement and everyone start only wanting organic cereal, which costs more money but bring more utility. Without IV, price is positively correlated with this demand shift which is positively correlated with utility, so price coefficient is downward biased. 

## cut price in half of product F1B04 in market C01Q2
counterfactual_data = df_blp[df_blp['market_ids'] == 'C01Q2']
counterfactual_data['new_prices'] = counterfactual_data['prices']
counterfactual_data.loc[counterfactual_data.index[counterfactual_data['product_ids'] == 'F1B04'],'new_prices'] /= 2
# counterfactual_data[counterfactual_data['product_ids'] == 'F1B04']['new_prices'] = counterfactual_data[counterfactual_data['product_ids'] == 'F1B04']['new_prices']/2 TRY this
counterfactual_data['new_shares'] = iv_results.compute_shares(market_id='C01Q2',prices=counterfactual_data['new_prices'])
counterfactual_data['F1'] = counterfactual_data['product_ids'].str.contains('F1B').astype(int)

print(counterfactual_data[counterfactual_data['F1'] == 1][['product_ids', 'shares', 'new_shares']])
print(counterfactual_data[counterfactual_data['F1'] == 0][['product_ids', 'shares', 'new_shares']])
print(counterfactual_data.groupby('F1')['shares','new_shares'].sum())
# There does seem to have some cannibalization. However, firm 1 gets higher share in total now. 

# Compute demand elasticity
new = iv_results.compute_elasticities(market_id='C01Q2')
print(new)

# Compute 
counterfactual_data['change'] = 100*(counterfactual_data['new_shares']/counterfactual_data['shares']-1)
print(counterfactual_data['change'].describe)