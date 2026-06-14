import pandas as pd
import numpy as np
import Functions
import warnings
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from pytictoc import TicToc
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.diagnostic as dg


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

t = TicToc()


################################################################## ----- DATA ----- ##################################################################
path = r'C:\Users\roberto\Desktop\PRICES.xlsx'

price_df = pd.read_excel(path, parse_dates=True, sheet_name='Assets').set_index('Date').dropna()

returns = pd.DataFrame(np.log(price_df/price_df.shift(1)).dropna())
train_returns = returns
volatilityReturns = Functions.getRollingVolatility(train_returns).dropna()

rates = pd.read_excel(path, parse_dates=True, sheet_name='Rates').set_index('Date').dropna()
train_rates = rates
volatility_rates = Functions.getRollingVolatility(train_rates).dropna()

inflation = pd.read_excel(path, parse_dates=True, sheet_name='Inflation').set_index('Date').dropna()
volatility_inflation = Functions.getRollingVolatility(inflation).dropna()

benchmarks_price = pd.read_excel(path, parse_dates=True, sheet_name='Benchmarks').set_index('Date').dropna()
benchmarks = pd.DataFrame(np.log(benchmarks_price/benchmarks_price.shift(1)).dropna())
volatility_benchmarks = Functions.getRollingVolatility(benchmarks).dropna()


################################################################## ----- PRELIMINARY ----- ##################################################################
preliminaryAnalysis = False
if preliminaryAnalysis:
    train_returns.info()
    train_rates.info()

    Functions.subplotRetVol(train_returns)
    Functions.subplotRetVol(train_rates)

    Functions.summaryStat(train_returns)
    Functions.summaryStat(train_rates)

    Functions.plotHistogram(train_returns)
    Functions.plotHistogram(train_rates)

    Functions.qqPlot(train_returns)
    Functions.qqPlot(train_rates)


################################################################## ----- STATIONARITY ----- ##################################################################
stationarityAnalysis = False
if stationarityAnalysis:
    # print(Functions.stationarityTest(train_returns, summary=False))
    # print(Functions.stationarityTest(train_rates, summary=False))
    print(Functions.stationarityTest(inflation, summary=False))
    # print(Functions.stationarityTest(benchmarks, summary=False))


################################################################## ----- DETREND ----- ##################################################################
ratesTrend = Functions.seasonal_decompose(train_rates, output='trend')
riskFreeDetrended = Functions.detrend(train_rates['MONEY MARKET'], ratesTrend['MONEY MARKET'])
interestRateDetrended = Functions.detrend(train_rates['DEBT INT RATE'], ratesTrend['DEBT INT RATE'])
ratesDetrended = pd.concat([riskFreeDetrended, interestRateDetrended], axis=1)
train_rates = ratesDetrended
train_rates = train_rates.dropna()

inflationTrend = Functions.seasonal_decompose(inflation, output='trend')
inflationDetrended = Functions.detrend(inflation['INFLATION'], inflationTrend['INFLATION']).dropna()
inflation['INFLATION'] = inflationDetrended

benchmarksTrend = Functions.seasonal_decompose(benchmarks, output='trend')
benchmarksDetrended = Functions.detrend(benchmarks, benchmarksTrend).dropna()
benchmarks = benchmarksDetrended


################################################################## ----- DIFFERENCING ----- ##################################################################
train_returns['US STOCKS'] = train_returns['US STOCKS'] - train_returns['US STOCKS'].shift(1)
train_returns = train_returns.dropna()

benchmarks = benchmarks - benchmarks.shift(1)
benchmarks = benchmarks.dropna()


################################################################## ----- POST DIFFERENCING STATIONARITY ----- ##################################################################
stationarityAnalysis = False
if stationarityAnalysis:
    # print(Functions.stationarityTest(train_returns, summary=False))
    # print(Functions.stationarityTest(train_rates, summary=False))
    print(Functions.stationarityTest(inflation, summary=False))
    # print(Functions.stationarityTest(benchmarks, summary=False))

    # adf_p_val = Functions.ADFtest(train_returns)
    # kpss_p_val = Functions.KPSStest(train_returns)


    # assets_list = list(adf_p_val.keys())
    # adf = list(adf_p_val[i] for i in adf_p_val)
    # kpss = list(kpss_p_val[i] for i in kpss_p_val)
    # print(adf)
    # print(kpss)


################################################################## ----- POST DIFFERENCING SUMMARY STAT ----- ##################################################################
summaryStatistics = False
if summaryStatistics:
    train_returns.info()
    rates.info()

    Functions.subplotRetVol(train_returns)
    Functions.subplotRetVol(rates)

    Functions.summaryStat(train_returns)
    Functions.summaryStat(rates)

    Functions.plotHistogram(train_returns)
    Functions.plotHistogram(rates)

    Functions.qqPlot(train_returns)
    Functions.qqPlot(rates)

    print(Functions.stationarityTest(train_returns, summary=False))
    print(Functions.stationarityTest(rates, summary=False))


################################################################## ----- AUTOCORRELATION ----- ##################################################################
autocorrelationAnalysis = False
if autocorrelationAnalysis:
    # Functions.plotAcfPacf(train_returns, title='returns')
    # Functions.plotAcfPacf(train_returns**2, title='squared returns')
    # Functions.plotAcfPacf(train_rates, title='returns')
    # Functions.plotAcfPacf(train_rates ** 2, title='squared returns')
    # acfReturns = Functions.ACF(train_returns, output='acf')
    # pacfReturns = Functions.PACF(train_returns, output='pacf')
    # acfRates = Functions.ACF(train_rates, output='acf')
    # pacfRates = Functions.PACF(train_rates, output='pacf')
    Functions.plotAcfPacf(inflation, title='inflation')


################################################################## ----- CONDITIONAL MEAN MODEL ----- ##################################################################
conditionalMeanModelAssets = False
if conditionalMeanModelAssets:
    t.tic()

    residualsArima = pd.DataFrame(index=train_returns.index)
    arimaCoeff = pd.DataFrame(index=['const', 'ar.L1', 'ar.L2', 'ar.L3', 'ma.L1'])

    ar = {}
    d = {}
    ma = {}

    for asset in train_returns:
        ar = Functions.getArimaOrder(train_returns, 'ar')
        d = Functions.getArimaOrder(train_returns, 'd')
        ma = Functions.getArimaOrder(train_returns, 'ma')

    print('Order stored successfully!')

    t_values = pd.DataFrame()
    p_values = pd.DataFrame()
    for asset in train_returns:
        model = ARIMA(train_returns[asset], order=(ar[asset], d[asset], ma[asset]))
        fitted = model.fit()
        residualsArima[asset] = fitted.resid

        print(fitted.summary())
        p_values[asset] = fitted.pvalues
        t_values[asset] = fitted.tvalues
        arimaCoeff[asset] = fitted.params

    arimaCoeff = arimaCoeff.fillna(0)

    # ----- RESIDUALS ----- #
    diagnosticAnalysis = True
    if diagnosticAnalysis:
        # Functions.subplotRetVol(residualsArima, title=['residuals', 'squared residuals', 'volatility'])
        # Functions.plotAcfPacf(residualsArima, 'ARIMA Residuals')
        # Functions.plotAcfPacf(residualsArima ** 2, 'ARIMA Squared Residuals')

        for asset in train_returns:
            model = ARIMA(train_returns[asset], order=(ar[asset], d[asset], ma[asset]))
            fitted = model.fit()
            residualsArima[asset] = fitted.resid
            plot = fitted.plot_diagnostics(lags=30)
            plt.show()

    # ----- EXPORT ----- #
    exportResults = False
    if exportResults:
        residualsArima.to_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_ASSETS.xlsx")
        arimaCoeff.to_excel(r"C:\Users\giovanna\Desktop\Export\arima_coeff_ASSETS.xlsx")
        p_values.to_excel(r"C:\Users\giovanna\Desktop\Export\p_values_ARMA_ASSETS.xlsx")
        t_values.to_excel(r"C:\Users\giovanna\Desktop\Export\t_values_ARMA_ASSETS.xlsx")

    t.toc()

conditionalMeanModelRates = False
if conditionalMeanModelRates:
    t.tic()

    residualsArima = pd.DataFrame(index=train_rates.index)
    arimaCoeff = pd.DataFrame(index=['const', 'ar.L1', 'ar.L2', 'ar.L3', 'ma.L1'])

    ar = {}
    d = {}
    ma = {}

    for asset in train_rates:
        ar = Functions.getArimaOrder(train_rates, 'ar')
        d = Functions.getArimaOrder(train_rates, 'd')
        ma = Functions.getArimaOrder(train_rates, 'ma')

    print('Order stored successfully!')

    for asset in train_rates:
        model = ARIMA(train_rates[asset], order=(ar[asset], d[asset], ma[asset]))
        fitted = model.fit()
        residualsArima[asset] = fitted.resid

        print(fitted.summary())
        arimaCoeff[asset] = fitted.params

    arimaCoeff = arimaCoeff.fillna(0)


    # ----- RESIDUALS ----- #
    diagnosticAnalysis = False
    if diagnosticAnalysis:
        Functions.subplotRetVol(residualsArima, title=['residuals', 'squared residuals', 'volatility'])
        Functions.plotAcfPacf(residualsArima, 'ARIMA Residuals')
        Functions.plotAcfPacf(residualsArima ** 2, 'ARIMA Squared Residuals')

        for asset in train_rates:
            model = ARIMA(train_rates[asset], order=(ar[asset], d[asset], ma[asset]))
            fitted = model.fit()
            residualsArima[asset] = fitted.resid
            plot = fitted.plot_diagnostics(lags=40)
            plt.show()

    # ----- EXPORT ----- #
    exportResults = True
    if exportResults:
        residualsArima.to_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_RATES.xlsx")
        arimaCoeff.to_excel(r"C:\Users\giovanna\Desktop\Export\arima_coeff_RATES.xlsx")

    t.toc()

conditionalMeanModelBenchmarks = False
if conditionalMeanModelBenchmarks:
    t.tic()

    residualsArima = pd.DataFrame(index=benchmarks.index)
    arimaCoeff = pd.DataFrame(index=['const', 'ar.L1', 'ar.L2', 'ar.L3', 'ma.L1'])

    ar = {}
    d = {}
    ma = {}

    for asset in benchmarks:
        ar = Functions.getArimaOrder(benchmarks[asset], 'ar')
        d = Functions.getArimaOrder(benchmarks[asset], 'd')
        ma = Functions.getArimaOrder(benchmarks[asset], 'ma')

    print('Order stored successfully!')

    for asset in benchmarks:
        model = ARIMA(benchmarks[asset], order=(ar[asset], d[asset], ma[asset]))
        fitted = model.fit()
        residualsArima[asset] = fitted.resid

        print(fitted.summary())
        arimaCoeff[asset] = fitted.params

    arimaCoeff = arimaCoeff.fillna(0)

    # ----- RESIDUALS ----- #
    diagnosticAnalysis = False
    if diagnosticAnalysis:
        Functions.subplotRetVol(residualsArima, title=['residuals', 'squared residuals', 'volatility'])
        Functions.plotAcfPacf(residualsArima, 'ARIMA Residuals')
        Functions.plotAcfPacf(residualsArima ** 2, 'ARIMA Squared Residuals')

        for asset in benchmarks:
            model = ARIMA(benchmarks[asset], order=(ar[asset], d[asset], ma[asset]))
            fitted = model.fit()
            residualsArima[asset] = fitted.resid
            plot = fitted.plot_diagnostics(lags=40)
            plt.show()

    # ----- EXPORT ----- #
    exportResults = True
    if exportResults:
        residualsArima.to_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_benchmarks.xlsx")
        arimaCoeff.to_excel(r"C:\Users\giovanna\Desktop\Export\arima_coeff_benchmarks.xlsx")

    t.toc()

conditionalMeanInflation = False
if conditionalMeanInflation:
    t.tic()

    residualsArima = pd.DataFrame(index=inflation.index)
    arimaCoeff = pd.DataFrame(index=['const', 'ar.L1', 'ar.L2', 'ar.L3', 'ma.L1'])

    ar = {}
    d = {}
    ma = {}

    ar = Functions.getArimaOrder(inflation, 'ar')
    d = Functions.getArimaOrder(inflation, 'd')
    ma = Functions.getArimaOrder(inflation, 'ma')

    print('Order stored successfully!')

    model = ARIMA(inflation, order=(ar, d, ma))
    fitted = model.fit()
    residualsArima = fitted.resid

    print(fitted.summary())
    arimaCoeff = fitted.params

    arimaCoeff = arimaCoeff.fillna(0)

    # ----- RESIDUALS ----- #
    diagnosticAnalysis = False
    if diagnosticAnalysis:
        Functions.subplotRetVol(residualsArima, title=['residuals', 'squared residuals', 'volatility'])
        Functions.plotAcfPacf(residualsArima, 'ARIMA Residuals')
        Functions.plotAcfPacf(residualsArima ** 2, 'ARIMA Squared Residuals')

        for asset in benchmarks:
            model = ARIMA(benchmarks[asset], order=(ar[asset], d[asset], ma[asset]))
            fitted = model.fit()
            residualsArima[asset] = fitted.resid
            plot = fitted.plot_diagnostics(lags=40)
            plt.show()

    # ----- EXPORT ----- #
    exportResults = True
    if exportResults:
        residualsArima.to_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_inflation.xlsx")
        arimaCoeff.to_excel(r"C:\Users\giovanna\Desktop\Export\arima_coeff_inflation.xlsx")

    t.toc()


################################################################## ----- CONDITIONAL VOLATILITY MODEL ----- ##################################################################
conditionalVolatilityModelAssets = False
if conditionalVolatilityModelAssets:
    t.tic()

    p = {}
    o = {}
    q = {}

    residualsArima = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_ASSETS.xlsx", parse_dates=True)
    residualsArima = residualsArima.rename(columns={'Unnamed: 0': 'Date'})
    residualsArima = residualsArima.set_index('Date')

    conditionalVolatility = pd.DataFrame(index=residualsArima.index)
    Residuals = pd.DataFrame()
    StandardizedResid = pd.DataFrame()
    garchCoeff = {}
    ljiungBox_df = pd.DataFrame()
    ljiungBox_squared_df = pd.DataFrame()

    for asset in residualsArima:
        p[asset] = Functions.getGarchOrder(max_alpha=2, max_gamma=1, max_beta=2, df_residuals=residualsArima, mean_model='zero')[asset][0]
        o[asset] = Functions.getGarchOrder(max_alpha=2, max_gamma=1, max_beta=2, df_residuals=residualsArima, mean_model='zero')[asset][1]
        q[asset] = Functions.getGarchOrder(max_alpha=2, max_gamma=1, max_beta=2, df_residuals=residualsArima, mean_model='zero')[asset][2]

        resid_model = arch_model(residualsArima[asset], mean='zero', p=p[asset][0], o=o[asset][0], q=q[asset][0], vol="GARCH")
        resid_model_results = resid_model.fit(last_obs=residualsArima[:].index[1:],
                                              update_freq=0, disp='off', show_warning=False)

        lm_test = resid_model_results.arch_lm_test(standardized=True)
        # print(lm_test)
        print(resid_model_results.summary())

        conditionalVolatility[asset] = resid_model_results.conditional_volatility
        Residuals[asset] = resid_model_results.resid
        StandardizedResid[asset] = resid_model_results.std_resid
        garchCoeff[asset] = resid_model_results.params.fillna(0)
        ljiungBox_squared_df[asset] = sm.stats.acorr_ljungbox(StandardizedResid[asset] ** 2, lags=[40], return_df=True).iloc[:, 1]
        ljiungBox_df[asset] = sm.stats.acorr_ljungbox(StandardizedResid[asset], lags=[40], return_df=True).iloc[:, 1]

        StandardizedResid['Rank'] = range(len(StandardizedResid))


    garchCoeff = pd.DataFrame(garchCoeff).fillna(0)
    garchCoeff = garchCoeff.reindex(index = ['omega','alpha[1]','gamma[1]','beta[1]','beta[2]'])

    t.toc()

    # ----- EXPORT ----- #
    exportResults = True
    if exportResults:
        StandardizedResid.to_excel(r"C:\Users\giovanna\Desktop\Export\std_residuals_ASSETS.xlsx")
        conditionalVolatility.to_excel(r"C:\Users\giovanna\Desktop\Export\conditional_volatility_ASSETS.xlsx")
        garchCoeff.to_excel(r"C:\Users\giovanna\Desktop\Export\garch_coeff_ASSETS.xlsx")

        ljiungBox_squared_df.to_excel(r"C:\Users\giovanna\Desktop\Export\ljungBox_squared_ASSETS.xlsx")
        ljiungBox_df.to_excel(r"C:\Users\giovanna\Desktop\Export\ljungBox_ASSETS.xlsx")

conditionalVolatilityModelRates = False
if conditionalVolatilityModelRates:
    t.tic()

    p = {}
    o = {}
    q = {}

    residualsArima = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_RATES.xlsx", parse_dates=True)
    residualsArima = residualsArima.rename(columns={'Unnamed: 0': 'Date'})
    residualsArima = residualsArima.set_index('Date')

    conditionalVolatility = pd.DataFrame(index=residualsArima.index)
    Residuals = pd.DataFrame()
    StandardizedResid = pd.DataFrame()
    garchCoeff = {}
    ljiungBox_df = pd.DataFrame()

    for asset in residualsArima:
        p[asset] = Functions.getGarchOrder(max_alpha=2, max_gamma=1, max_beta=2, df_residuals=residualsArima, mean_model='zero')[asset][0]
        o[asset] = Functions.getGarchOrder(max_alpha=2, max_gamma=1, max_beta=2, df_residuals=residualsArima, mean_model='zero')[asset][1]
        q[asset] = Functions.getGarchOrder(max_alpha=2, max_gamma=1, max_beta=2, df_residuals=residualsArima, mean_model='zero')[asset][2]

        resid_model = arch_model(residualsArima[asset], mean='zero', p=p[asset][0], o=o[asset][0], q=q[asset][0], vol="GARCH")
        resid_model_results = resid_model.fit(last_obs=residualsArima[:].index[1:],
                                              update_freq=0, disp='off', show_warning=False)

        # print(resid_model_results.summary())
        lm_test = resid_model_results.arch_lm_test(standardized=True)
        print(lm_test)

        conditionalVolatility[asset] = resid_model_results.conditional_volatility
        Residuals[asset] = resid_model_results.resid
        StandardizedResid[asset] = resid_model_results.std_resid
        garchCoeff[asset] = resid_model_results.params.fillna(0)
        ljiungBox_df[asset] = sm.stats.acorr_ljungbox(StandardizedResid[asset] ** 2, lags=[40], return_df=True).iloc[:, 1]
        StandardizedResid['Rank'] = range(len(StandardizedResid))

    garchCoeff = pd.DataFrame(garchCoeff).fillna(0)
    garchCoeff = garchCoeff.reindex(index = ['omega','alpha[1]','gamma[1]','beta[1]','beta[2]'])

    t.toc()

    # ----- EXPORT ----- #
    exportResults = False
    if exportResults:
        StandardizedResid.to_excel(r"C:\Users\giovanna\Desktop\Export\std_residuals_RATES.xlsx")
        conditionalVolatility.to_excel(r"C:\Users\giovanna\Desktop\Export\conditional_volatility_RATES.xlsx")
        garchCoeff.to_excel(r"C:\Users\giovanna\Desktop\Export\garch_coeff_RATES.xlsx")
        ljiungBox_df.to_excel(r"C:\Users\giovanna\Desktop\Export\ljungBox_RATES.xlsx")

conditionalVolatilityModelBenchmarks = False
if conditionalVolatilityModelBenchmarks:
    t.tic()

    p = {}
    o = {}
    q = {}

    residualsArima = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_benchmarks.xlsx", parse_dates=True)
    residualsArima = residualsArima.rename(columns={'Unnamed: 0': 'Date'})
    residualsArima = residualsArima.set_index('Date')

    conditionalVolatility = pd.DataFrame(index=residualsArima.index)
    Residuals = pd.DataFrame()
    StandardizedResid = pd.DataFrame()
    garchCoeff = {}
    ljiungBox_df = pd.DataFrame()

    for asset in residualsArima:
        p[asset] = Functions.getGarchOrder(max_alpha=2, max_gamma=1, max_beta=2, df_residuals=residualsArima, mean_model='zero')[asset][0]
        o[asset] = Functions.getGarchOrder(max_alpha=2, max_gamma=1, max_beta=2, df_residuals=residualsArima, mean_model='zero')[asset][1]
        q[asset] = Functions.getGarchOrder(max_alpha=2, max_gamma=1, max_beta=2, df_residuals=residualsArima, mean_model='zero')[asset][2]

        resid_model = arch_model(residualsArima[asset], mean='zero', p=p[asset][0], o=o[asset][0], q=q[asset][0], vol="GARCH")
        resid_model_results = resid_model.fit(last_obs=residualsArima[:].index[1:],
                                              update_freq=0, disp='off', show_warning=False)

        print(resid_model_results.summary())
        # lm_test = resid_model_results.arch_lm_test(standardized=True)
        # print(lm_test)

        conditionalVolatility[asset] = resid_model_results.conditional_volatility
        Residuals[asset] = resid_model_results.resid
        StandardizedResid[asset] = resid_model_results.std_resid
        garchCoeff[asset] = resid_model_results.params.fillna(0)
        ljiungBox_df[asset] = sm.stats.acorr_ljungbox(StandardizedResid[asset] ** 2, lags=[40], return_df=True).iloc[:, 1]
        StandardizedResid['Rank'] = range(len(StandardizedResid))

    garchCoeff = pd.DataFrame(garchCoeff).fillna(0)
    garchCoeff = garchCoeff.reindex(index = ['omega','alpha[1]','gamma[1]','beta[1]','beta[2]'])

    t.toc()

    # ----- EXPORT ----- #
    exportResults = True
    if exportResults:
        StandardizedResid.to_excel(r"C:\Users\giovanna\Desktop\Export\std_residuals_Benchmarks.xlsx")
        conditionalVolatility.to_excel(r"C:\Users\giovanna\Desktop\Export\conditional_volatility_Benchmarks.xlsx")
        garchCoeff.to_excel(r"C:\Users\giovanna\Desktop\Export\garch_coeff_Benchmarks.xlsx")
        ljiungBox_df.to_excel(r"C:\Users\giovanna\Desktop\Export\ljungBox_Benchmarks.xlsx")

conditionalVolatilityModelInflation = False
if conditionalVolatilityModelInflation:
    t.tic()

    p = {}
    o = {}
    q = {}

    residualsArima = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_inflation.xlsx",
                                   parse_dates=True).set_index('Date')
    print(type(residualsArima))
    residualsArima = residualsArima.rename(columns={'Unnamed: 0': 'Date'})
    # residualsArima = residualsArima.set_index('Date')

    conditionalVolatility = pd.DataFrame(index=residualsArima.index)
    Residuals = pd.DataFrame()
    StandardizedResid = pd.DataFrame()
    garchCoeff = {}
    ljiungBox_df = pd.DataFrame()

    p = Functions.getGarchOrder(max_alpha=2, max_gamma=1, max_beta=2, df_residuals=residualsArima, mean_model='zero')[0]
    o = p[1]
    q = p[2]

    resid_model = arch_model(residualsArima, mean='zero', p=p[0][0], o=o[0], q=q[0],vol="GARCH")
    resid_model_results = resid_model.fit(last_obs=residualsArima[:].index[1:], update_freq=0, disp='off', show_warning=False)

    print(resid_model_results.summary())
    # lm_test = resid_model_results.arch_lm_test(standardized=True)
    # print(lm_test)

    conditionalVolatility = resid_model_results.conditional_volatility
    Residuals = resid_model_results.resid
    StandardizedResid = resid_model_results.std_resid
    garchCoeff = resid_model_results.params.fillna(0)
    ljiungBox_df = sm.stats.acorr_ljungbox(StandardizedResid ** 2, lags=[40], return_df=True).iloc[:,1]
    StandardizedResid['Rank'] = range(len(StandardizedResid))

    garchCoeff = pd.DataFrame(garchCoeff).fillna(0)
    garchCoeff = garchCoeff.reindex(index=['omega', 'alpha[1]', 'gamma[1]', 'beta[1]', 'beta[2]'])

    t.toc()

    # ----- EXPORT ----- #
    exportResults = True
    if exportResults:
        StandardizedResid.to_excel(r"C:\Users\giovanna\Desktop\Export\std_residuals_Inflation.xlsx")
        conditionalVolatility.to_excel(r"C:\Users\giovanna\Desktop\Export\conditional_volatility_Inflation.xlsx")
        garchCoeff.to_excel(r"C:\Users\giovanna\Desktop\Export\garch_coeff_Inflation.xlsx")
        ljiungBox_df.to_excel(r"C:\Users\giovanna\Desktop\Export\ljungBox_Inflation.xlsx")


################################################################## ----- BACKTESTING ----- ##################################################################
testAccuracyAssets = False
if testAccuracyAssets:

    residualsAssets = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_ASSETS.xlsx").set_index('Date')
    standardizedResidualsAssets = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\std_residuals_ASSETS.xlsx").drop(columns='Date').set_index('Rank')
    standardizedSquaredResidualsAssets = standardizedResidualsAssets ** 2

    LM_dict = {}
    LM_p_values = pd.DataFrame(columns=residualsAssets.columns)
    for asset in residualsAssets:
        LM_dict[asset] = dg.het_arch(residualsAssets[asset], nlags=40)
        LM_dict[asset] = LM_dict[asset][1]
        # print(LM_dict[asset][1])
        # LM_p_values[asset] = LM_dict[asset][1]
    print(LM_dict)

    # normalityTestAsset = Functions.runShapiroWilkTest(residualsAssets, alpha=0.10)

    # print('\n--- Residuals ---\n')
    # ljungBox_dfAssets = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\ljungBox_ASSETS.xlsx")
    # ljungBox_dfAssets = ljungBox_dfAssets.drop(columns='Unnamed: 0')
    # ljungBoxTestAssets = Functions.runLjungBoxTest(ljungBox_dfAssets, alpha=0.05)
    #
    # print('\n--- Squared residuals ---\n')
    # ljungBox_squared_dfAssets = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\ljungBox_squared_ASSETS.xlsx")
    # ljungBox_squared_dfAssets = ljungBox_squared_dfAssets.drop(columns='Unnamed: 0')
    # ljungBox_squared_TestAssets = Functions.runLjungBoxTest(ljungBox_squared_dfAssets, alpha=0.05)

testAccuracyRates = False
if testAccuracyRates:

    # residualsRates = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_RATES.xlsx").set_index('Date')
    # standardizedResidualsRates = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\std_residuals_RATES.xlsx").drop(columns='Date').set_index('Rank')
    # standardizedSquaredResidualsRates = standardizedResidualsRates ** 2
    #
    # normalityTestRates = Functions.runShapiroWilkTest(residualsRates, alpha=0.10)
    # ljungBox_dfRates = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\ljungBox_RATES.xlsx")
    # ljungBox_dfRates = ljungBox_dfRates.drop(columns='Unnamed: 0')
    # ljungBoxTestRates = Functions.runLjungBoxTest(ljungBox_dfRates, alpha=0.05)

    residualsBenchmarks = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\residuals_arima_Benchmarks.xlsx").set_index('Date')
    normalityTestBenchmarks = Functions.runShapiroWilkTest(residualsBenchmarks, alpha=0.10)
    ljungBox_dfBenchmarks = pd.read_excel(r"C:\Users\giovanna\Desktop\Export\ljungBox_Benchmarks.xlsx")
    ljungBox_dfBenchmarks = ljungBox_dfBenchmarks.drop(columns='Unnamed: 0')
    ljungBoxTestBenchmarks = Functions.runLjungBoxTest(ljungBox_dfBenchmarks, alpha=0.05)


