import pandas as pd
import numpy as np
import Functions
import Functions as F
from pytictoc import TicToc

t = TicToc()

list_of_paths_assets = []
list_of_paths_rates = []
list_of_paths_Benchmarks = []

list_of_paths_inflation = []

def ParallelBootstrap(years, list_of_paths, asset_class, freq, num_scenario):

    t.tic()

    print('\n--- Running simulation ---\n')

    if freq == 'month':
        periods = years * 12
    elif freq == 'week':
        periods = years * 52
    elif freq == 'day':
        periods = years * 250
    step = list(range(periods))

    std_residuals = pd.read_excel(list_of_paths[0]).drop('Date', axis=1)
    std_residuals = std_residuals.set_index('Rank')

    resid_arima = pd.read_excel(list_of_paths[1]).drop('Date', axis=1).set_index(std_residuals.index)
    conditional_vol = pd.read_excel(list_of_paths[2]).drop('Date', axis=1).set_index(std_residuals.index)
    conditional_var = conditional_vol ** 2
    arima_coeff = pd.read_excel(list_of_paths[3]).set_index('Unnamed: 0').fillna(0)
    garch_coeff = pd.read_excel(list_of_paths[4]).set_index('Unnamed: 0').fillna(0)
    price_df = pd.read_excel(list_of_paths[5], sheet_name=asset_class).drop('Date', axis=1).dropna()
    train_returns = pd.DataFrame(np.log(price_df / price_df.shift(1)))

    train_returns.index = train_returns.index - 1
    train_returns = train_returns.dropna()

    z_dict = F.create_df_dict(price_df.columns, num_scenario, step)
    price_forecast_dict = F.create_df_dict(price_df.columns, num_scenario, step)
    vol_forecast_dict = F.create_df_dict(price_df.columns, num_scenario, step)
    var_forecast_dict = F.create_df_dict(price_df.columns, num_scenario, step)
    ret_forecast_dict = F.create_df_dict(price_df.columns, num_scenario, step)

    # ! The Indicator is built starting by the last observation in the sample -> it is one period behind, then Indicator with rank 0 is the Indicator for time Rank - 1
    indicator = pd.DataFrame(index=std_residuals.index)

    for asset in price_df.columns:
        for index, row in std_residuals.iterrows():
            if index == 0:
                if resid_arima[asset].iloc[-1] > 0:
                    indicator[asset] = 0
                else:
                    indicator[asset] = 1
            else:
                if row[asset] > 0:
                    indicator[asset][index] = 0
                else:
                    indicator[asset][index] = 1

    for s in range(num_scenario):
        print('Scenario: ' + str(s + 1) + ' out of ' + str(num_scenario))
        randomVector = np.random.randint(low=1, high=(len(std_residuals.index)), size=periods)  # vector of pseudo-random numbers

        for asset in price_df.columns:
            for i in range(periods):
                draw = randomVector[i]  # element i of the pseudo-random vector

                if i == 0:
                    var_forecast_dict[asset]['Scenario_' + str(s)] = garch_coeff[asset][0] + ((garch_coeff[asset][1]
                            + garch_coeff[asset][2] * indicator[asset][indicator.index == draw].iloc[0]) * (resid_arima[asset].iloc[0] ** 2)) \
                            + garch_coeff[asset][3] * conditional_var[asset].iloc[0] + garch_coeff[asset][4] * conditional_var[asset].iloc[1]

                    vol_forecast_dict[asset]['Scenario_' + str(s)] = var_forecast_dict[asset]['Scenario_' + str(s)] ** 0.5

                    z_dict[asset]['Scenario_' + str(s)] = std_residuals[asset][std_residuals.index == draw].iloc[0] * vol_forecast_dict[asset]['Scenario_' + str(s)].iloc[0]

                    if asset_class == 'Inflation':
                        train_returns = price_df
                        price_df[asset].iloc[0] = 1

                    price_forecast_dict[asset]['Scenario_' + str(s)] = price_df[asset].iloc[0] + price_df[asset].iloc[0] \
                            * (arima_coeff[asset][1] * train_returns[asset].iloc[0] + arima_coeff[asset][2] * train_returns[asset].iloc[1]
                            + arima_coeff[asset][3] * train_returns[asset].iloc[2] + arima_coeff[asset][4]
                            * resid_arima[asset].iloc[0] + z_dict[asset]['Scenario_' + str(s)].iloc[0])

                    ret_forecast_dict[asset]['Scenario_' + str(s)] = (arima_coeff[asset][1] * train_returns[asset].iloc[0]
                         + arima_coeff[asset][2] * train_returns[asset].iloc[1] + arima_coeff[asset][3] * train_returns[asset].iloc[2]
                         + arima_coeff[asset][4] * resid_arima[asset].iloc[0] + z_dict[asset]['Scenario_' + str(s)].iloc[0])

                elif i == 1:
                    var_forecast_dict[asset]['Scenario_' + str(s)] = garch_coeff[asset][0] + ((garch_coeff[asset][1]
                            + garch_coeff[asset][2] * indicator[asset][indicator.index == draw].iloc[0]) * (z_dict[asset]['Scenario_' + str(s)].iloc[0] ** 2)) \
                            + garch_coeff[asset][3] * var_forecast_dict[asset]['Scenario_' + str(s)].iloc[0] \
                            + garch_coeff[asset][4] * conditional_var[asset].iloc[0]

                    vol_forecast_dict[asset]['Scenario_' + str(s)].iloc[1] = var_forecast_dict[asset]['Scenario_' + str(s)].iloc[1] ** 0.5

                    z_dict[asset]['Scenario_' + str(s)].iloc[1] = std_residuals[std_residuals.index == draw][asset].iloc[0] * vol_forecast_dict[asset]['Scenario_' + str(s)].iloc[1]

                    price_forecast_dict[asset]['Scenario_' + str(s)].iloc[1] = price_forecast_dict[asset]['Scenario_' + str(s)].iloc[0] \
                           + price_forecast_dict[asset]['Scenario_' + str(s)].iloc[0] * (arima_coeff[asset][1]
                           * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[0]+ arima_coeff[asset][2]
                           * train_returns[asset].iloc[0] + arima_coeff[asset][3]
                           * train_returns[asset].iloc[1] + arima_coeff[asset][4]
                           * z_dict[asset]['Scenario_' + str(s)].iloc[0] + z_dict[asset]['Scenario_' + str(s)].iloc[1])

                    ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[1] = (arima_coeff[asset][1] * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[0]
                         + arima_coeff[asset][2] * train_returns[asset].iloc[0]
                         + arima_coeff[asset][3] * train_returns[asset].iloc[1]
                         + arima_coeff[asset][4] * z_dict[asset]['Scenario_' + str(s)].iloc[0]
                         + z_dict[asset]['Scenario_' + str(s)].iloc[1])


                elif i == 2:
                    var_forecast_dict[asset]['Scenario_' + str(s)].iloc[2] = garch_coeff[asset][0] + ((garch_coeff[asset][1] + garch_coeff[asset][2] * indicator[asset][indicator.index == draw].iloc[0])
                                                                            * (z_dict[asset]['Scenario_' + str(s)].iloc[1] ** 2)) + garch_coeff[asset][3] * var_forecast_dict[asset]['Scenario_' + str(s)].iloc[1] \
                                                                            + garch_coeff[asset][4] * var_forecast_dict[asset]['Scenario_' + str(s)].iloc[0]

                    vol_forecast_dict[asset]['Scenario_' + str(s)].iloc[2] = var_forecast_dict[asset]['Scenario_' + str(s)].iloc[2] ** 0.5

                    z_dict[asset]['Scenario_' + str(s)].iloc[2] = std_residuals[std_residuals.index == draw][asset].iloc[0] * vol_forecast_dict[asset]['Scenario_' + str(s)].iloc[2]

                    price_forecast_dict[asset]['Scenario_' + str(s)].iloc[2] = price_forecast_dict[asset]['Scenario_' + str(s)].iloc[1] + price_forecast_dict[asset]['Scenario_' + str(s)].iloc[1] * (arima_coeff[asset][1]
                           * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[1] + arima_coeff[asset][2] * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[0] + arima_coeff[asset][3]
                           * train_returns[asset].iloc[0] + arima_coeff[asset][4] * z_dict[asset]['Scenario_' + str(s)].iloc[1] + z_dict[asset]['Scenario_' + str(s)].iloc[2])

                    ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[2] = (arima_coeff[asset][1] * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[1]
                         + arima_coeff[asset][2] * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[0] + arima_coeff[asset][3] * train_returns[asset].iloc[0]
                         + arima_coeff[asset][4] * z_dict[asset]['Scenario_' + str(s)].iloc[1] + z_dict[asset]['Scenario_' + str(s)].iloc[2])

                else:
                    var_forecast_dict[asset]['Scenario_' + str(s)].iloc[i] = \
                        garch_coeff[asset][0] + \
                        ((garch_coeff[asset][1]
                          + garch_coeff[asset][2]
                          * indicator[asset][indicator.index == draw].iloc[0])
                         * (z_dict[asset]['Scenario_' + str(s)].iloc[i - 1] ** 2)) \
                        + garch_coeff[asset][3] \
                        * var_forecast_dict[asset]['Scenario_' + str(s)].iloc[i - 1] \
                        + garch_coeff[asset][4] \
                        * var_forecast_dict[asset]['Scenario_' + str(s)].iloc[i - 2]

                    vol_forecast_dict[asset]['Scenario_' + str(s)].iloc[i] = \
                    var_forecast_dict[asset]['Scenario_' + str(s)].iloc[i] ** 0.5

                    z_dict[asset]['Scenario_' + str(s)].iloc[i] = \
                        std_residuals[std_residuals.index == draw][asset].iloc[0] \
                        * vol_forecast_dict[asset]['Scenario_' + str(s)][i]

                    price_forecast_dict[asset]['Scenario_' + str(s)].iloc[i] = \
                        price_forecast_dict[asset]['Scenario_' + str(s)].iloc[i - 1] \
                        + price_forecast_dict[asset]['Scenario_' + str(s)].iloc[i - 1] \
                        * (arima_coeff[asset][1]
                           * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[i - 1]
                           + arima_coeff[asset][2]
                           * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[i - 2]
                           + arima_coeff[asset][3]
                           * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[i - 3]
                           + arima_coeff[asset][4]
                           * z_dict[asset]['Scenario_' + str(s)].iloc[i - 1]
                           + z_dict[asset]['Scenario_' + str(s)].iloc[i])

                    ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[i] = \
                        (arima_coeff[asset][1]
                         * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[i - 1]
                         + arima_coeff[asset][2]
                         * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[i - 2]
                         + arima_coeff[asset][3]
                         * ret_forecast_dict[asset]['Scenario_' + str(s)].iloc[i - 3]
                         + arima_coeff[asset][4]
                         * z_dict[asset]['Scenario_' + str(s)].iloc[i - 1]
                         + z_dict[asset]['Scenario_' + str(s)].iloc[i])

    exportResults = True
    if exportResults:
        for asset in price_df.columns:
            ret_forecast_dict[asset].to_csv(r"C:\Users\giovanna\Desktop\Simulation Results\Simulated Returns - FINAL 200_" + str(asset) + ".csv")
            # price_forecast_dict[asset].to_excel(r"C:\Users\giovanna\Desktop\Simulation Results\PRICE_FORECAST_" + str(asset) + ".xlsx")
            # vol_forecast_dict[asset].to_excel(r"C:\Users\giovanna\Desktop\Simulation Results\VOLATILITY_FORECAST_" + str(asset) + ".xlsx")
            # z_dict[asset].to_excel(r"C:\Users\giovanna\Desktop\Simulation Results\Z_FORECAST_" + str(asset) + ".xlsx")

    t.toc()

    prova = False
    if prova:
        df = pd.DataFrame()
        for asset in price_df.columns:
            for s in range(num_scenario):
                df[asset] = ret_forecast_dict[asset]['Scenario_' + str(s)]
                df.to_excel(r"C:\Users\giovanna\Desktop\Simulation Results\Assets Sample Scenario 1.xlsx")

    return ret_forecast_dict

years = 20
scenarios = 200

runSimulation = True
if runSimulation:
    # print('\n\n\n ------------------------------- \n\n\n ------------------------------- \n\n\n')
    simulationResultAssets = ParallelBootstrap(years, list_of_paths_assets, asset_class='Assets',  freq='week', num_scenario=scenarios)
    # print('\n\n\n ------------------------------- \n\n\n ------------------------------- \n\n\n')
    simulationResultRates = ParallelBootstrap(years, list_of_paths_rates, asset_class='Rates',  freq='week', num_scenario=scenarios)
    # print('\n\n\n ------------------------------- \n\n\n ------------------------------- \n\n\n')
    simulationResultBenchmarks = ParallelBootstrap(years, list_of_paths_Benchmarks, asset_class='Benchmarks',  freq='week', num_scenario=scenarios)

    simulationResultInflation = ParallelBootstrap(years, list_of_paths_inflation, asset_class='Inflation',  freq='month', num_scenario=scenarios)


