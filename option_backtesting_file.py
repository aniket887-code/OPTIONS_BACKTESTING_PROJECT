import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
from scipy.stats import lognorm


def get_premium(options_strategy, options_data):
    condition_1 = options_data['Option Type'] == options_strategy['Option Type']
    condition_2 = options_data['Strike Price'] == options_strategy['Strike Price']
    return options_data.loc[condition_1 & condition_2, 'Last']


# ---------------------------------- Butterfly ----------------------------------

def setup_butterfly(futures_price, options_data, direction='long'):
    butterfly = pd.DataFrame()

    butterfly['Option Type'] = ['CE', 'PE']
    atm_strike_price = 50 * (round(futures_price / 50))
    butterfly['Strike Price'] = atm_strike_price
    butterfly['position'] = 1

    butterfly['premium'] = butterfly.apply(lambda r: get_premium(r, options_data), axis=1)

    deviation = round(butterfly.premium.sum() / 50) * 50
    butterfly.loc['2'] = ['CE', atm_strike_price + deviation, -1, np.nan]
    butterfly.loc['3'] = ['PE', atm_strike_price - deviation, -1, np.nan]

    if direction == 'long':
        butterfly['position'] *= -1

    butterfly['premium'] = butterfly.apply(lambda r: get_premium(r, options_data), axis=1)

    # net_premium = (butterfly.positions * butterfly.premium).sum()

    return butterfly


# ---------------------------------- Bull Call Spread ----------------------------------

def setup_call_spread(futures_price, options_data, direction='bull'):
    call_spread = pd.DataFrame(columns=['Option Type', 'Strike Price', 'position', 'premium'])

    atm_strike_price = 50 * (round(futures_price / 50))

    call_spread.loc['0'] = ['CE', atm_strike_price, 1, np.nan]

    call_spread['premium'] = call_spread.apply(lambda r: get_premium(r, options_data), axis=1)
    deviation = round(call_spread.premium.sum() / 50) * 50

    call_spread.loc['1'] = ['CE', atm_strike_price + deviation, -1, np.nan]

    if direction == 'bear':
        call_spread['position'] *= -1

    call_spread['premium'] = call_spread.apply(lambda r: get_premium(r, options_data), axis=1)

    return call_spread


# ---------------------------------- Bear Put Spread ----------------------------------

def setup_put_spread(futures_price, options_data, direction='bear'):
    put_spread = pd.DataFrame(columns=['Option Type', 'Strike Price', 'position', 'premium'])

    atm_strike_price = 50 * (round(futures_price / 50))

    put_spread.loc['0'] = ['PE', atm_strike_price, 1, np.nan]

    put_spread['premium'] = put_spread.apply(lambda r: get_premium(r, options_data), axis=1)
    deviation = round(put_spread.premium.sum() / 50) * 50

    put_spread.loc['1'] = ['PE', atm_strike_price - deviation, -1, np.nan]

    if direction == 'bull':
        put_spread['position'] *= -1

    put_spread['premium'] = put_spread.apply(lambda r: get_premium(r, options_data), axis=1)

    return put_spread


# ---------------------------------- Iron Condor ----------------------------------

def setup_iron_condor(futures_price, options_data, direction='long'):
    iron_condor = pd.DataFrame(columns=['Option Type', 'Strike Price', 'position', 'premium'])

    atm_strike_price = 50 * (round(futures_price / 50))

    first_deviation = 50
    iron_condor.loc['0'] = ['CE', atm_strike_price + first_deviation, 1, np.nan]
    iron_condor.loc['1'] = ['PE', atm_strike_price - first_deviation, 1, np.nan]

    iron_condor['premium'] = iron_condor.apply(lambda r: get_premium(r, options_data), axis=1)

    second_deviation = first_deviation + round(iron_condor.premium.sum() / 50) * 50
    iron_condor.loc['2'] = ['CE', atm_strike_price + second_deviation, -1, np.nan]
    iron_condor.loc['3'] = ['PE', atm_strike_price - second_deviation, -1, np.nan]

    if direction == 'short':
        iron_condor['position'] *= -1

    iron_condor['premium'] = iron_condor.apply(lambda r: get_premium(r, options_data), axis=1)

    return iron_condor


# ---------------------------------- Options Payoff at Expiry  ----------------------------------

# Long call payoff
def long_call_payoff(spot_price, strike_price, premium_spent):
    return np.where(spot_price < strike_price, 0, spot_price - strike_price) - premium_spent


# Long put payoff
def long_put_payoff(spot_price, strike_price, premium_spent):
    return np.where(spot_price < strike_price, strike_price - spot_price, 0) - premium_spent


# Short call payoff
def short_call_payoff(spot_price, strike_price, premium_collected):
    breakeven = strike_price + premium_collected
    return float(
        np.where(spot_price > breakeven, breakeven - spot_price, min(premium_collected, breakeven - spot_price)))


# Short put payoff
def short_put_payoff(spot_price, strike_price, premium_collected):
    breakeven = strike_price - premium_collected
    return float(
        np.where(spot_price > breakeven, min(premium_collected, spot_price - breakeven), spot_price - breakeven))


def get_payoff(spot_price_expiry, options_strategy):
    options_strategy['payoff'] = np.nan
    for i in options_strategy.index:
        if options_strategy.loc[i, 'Option Type'] == 'CE':
            if options_strategy.loc[i, 'position'] == 1:
                # long call payoff at expiry
                options_strategy.loc[i, 'payoff'] = long_call_payoff(spot_price_expiry,
                                                                     options_strategy.loc[i, 'Strike Price'],
                                                                     options_strategy.loc[i, 'premium'])

            elif options_strategy.loc[i, 'position'] == -1:
                # Short call payoff at expiry
                options_strategy.loc[i, 'payoff'] = short_call_payoff(spot_price_expiry,
                                                                      options_strategy.loc[i, 'Strike Price'],
                                                                      options_strategy.loc[i, 'premium'])

        elif options_strategy.loc[i, 'Option Type'] == 'PE':
            if options_strategy.loc[i, 'position'] == 1:
                # long put payoff at expiry
                options_strategy.loc[i, 'payoff'] = long_put_payoff(spot_price_expiry,
                                                                    options_strategy.loc[i, 'Strike Price'],
                                                                    options_strategy.loc[i, 'premium'])

            elif options_strategy.loc[i, 'position'] == -1:
                # Short put payoff at expiry
                options_strategy.loc[i, 'payoff'] = short_put_payoff(spot_price_expiry,
                                                                     options_strategy.loc[i, 'Strike Price'],
                                                                     options_strategy.loc[i, 'premium'])

    return options_strategy['payoff'].sum()

# ---------------------------------- POP Lognormal  ----------------------------------

def get_pop_lognormal(futures_data, current_price, days_to_expiry, price_range):

    trading_days_to_expiry = round(days_to_expiry * (5 / 7))
    daily_historical_volatility = np.log(futures_data.futures_close / futures_data.futures_close.shift(1)).std()
    vol_normalised = daily_historical_volatility * np.sqrt(trading_days_to_expiry)
    atm_price = 50 * (round(current_price / 50))

    # Last traded price
    mean = atm_price

    # Deviation in price
    stdev = vol_normalised * mean

    phi = (stdev ** 2 + mean ** 2) ** 0.5
    mu = np.log(mean ** 2 / phi)
    sigma = (np.log(phi ** 2 / mean ** 2)) ** 0.5

    payoff = pd.DataFrame(index=price_range)
    payoff['probability_lognormal'] = lognorm.cdf(x=price_range, scale=mean, s=sigma)
    payoff['probability_lognormal'] = payoff['probability_lognormal'].diff()
    return payoff

# ---------------------------------- POP Empirical  ----------------------------------

def get_pop_empirical(futures_data, current_price, days_to_expiry, price_range):
    # Calculate empirical POP
    trading_days_to_expiry = round(days_to_expiry * (5 / 7))
    futures_data['percent_change_in_price'] = futures_data.futures_close.pct_change(trading_days_to_expiry)
    forecasted_prices = (1 + futures_data['percent_change_in_price']) * current_price
    histogram = np.histogram(forecasted_prices.dropna(), bins=30)
    hist_dist = rv_histogram(histogram)

    payoff = pd.DataFrame(index=price_range)
    payoff['probability_empirical'] = hist_dist.cdf(x=price_range)
    payoff['probability_empirical'] = payoff['probability_empirical'].diff()
    return payoff


# ---------------------------------- Expected Profit Empirical  ----------------------------------

def get_expected_profit_empirical(futures_data, options_strategy, days_to_expiry, price_range):
    # Payoff
    payoff = pd.DataFrame()
    payoff['price_range'] = price_range
    payoff['pnl'] = payoff.apply(lambda r: get_payoff(r.price_range, options_strategy), axis=1)
    payoff.set_index('price_range', inplace=True)

    futures_price = futures_data.futures_close[-1]

    # POP Empirical
    payoff['probability_empirical'] = get_pop_empirical(futures_data, futures_price, days_to_expiry, payoff.index)

    # Expected Profit
    payoff['expected_pnl'] = payoff.probability_empirical * payoff.pnl
    exp_profit = payoff['expected_pnl'].sum()
    return exp_profit


# ---------------------------------- IV Percentile  ----------------------------------

def calculate_IV(daily_data):
    import mibian

    if daily_data.days_to_expiry == 0 or daily_data.Last == 0:
        return 0

    elif daily_data['Option Type'] == 'CE':
        return mibian.BS([daily_data['futures_close'], daily_data['Strike Price'], 0,
                          daily_data['days_to_expiry']], callPrice=daily_data['Last']).impliedVolatility

    elif daily_data['Option Type'] == 'PE':
        return mibian.BS([daily_data['futures_close'], daily_data['Strike Price'], 0,
                          daily_data['days_to_expiry']], putPrice=daily_data['Last']).impliedVolatility


def get_IV_percentile(futures_data, options_data, window):
    from scipy import stats
    import pandas as pd

    futures_data['atm_strike_price'] = 50 * round(futures_data['futures_close'] / 50)

    data = pd.merge(options_data, futures_data, left_on=['Date', 'Strike Price'], right_on=['Date', 'atm_strike_price'],
                    suffixes=('_options', '_futures'))

    data.Expiry_options = pd.to_datetime(data.Expiry_options)

    data.Expiry_options = pd.to_datetime(data.Expiry_options)
    data['days_to_expiry'] = (data['Expiry_options'] - data.index).dt.days

    data['IV'] = data.apply(calculate_IV, axis=1)

    futures_data['IV'] = data.groupby('Date').IV.mean()
    futures_data = futures_data.replace(to_replace=0, method='ffill')

    futures_data['IV_percentile'] = futures_data['IV'].rolling(window).apply(lambda x: stats.percentileofscore(x, x[-1]))

    return futures_data['IV_percentile']
