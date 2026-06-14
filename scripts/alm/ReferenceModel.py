import pyomo.environ as pyo
import pandas as pd


model = pyo.AbstractModel()
model.Assets = pyo.Set()
model.Times = pyo.Set()
model.NumScenarios = pyo.Set()


############################################################ ----- PARAMETERS ----- ############################################################
model.Lambda = pyo.Param()
model.InitialPortfolioValue = pyo.Param()
model.InitialCash = pyo.Param()
model.TargetEstate_stage = pyo.Param()
model.TargetWealth_stage = pyo.Param()
model.Upper = pyo.Param()
model.MaxBorr = pyo.Param()
model.Fee = pyo.Param()

model.InitialHoldings = pyo.Param(model.Assets)     # List of asset values: NOT WEIGHTS
model.TargetEstate = pyo.Param(model.Times)
model.TargetWealth = pyo.Param(model.Times)
model.Income = pyo.Param(model.Times)
model.Consumption = pyo.Param(model.Times)
model.Returns = pyo.Param(model.Assets, model.Times)
model.Inflation = pyo.Param(model.Times)
model.ShortRate = pyo.Param(model.Times)
model.RiskFree = pyo.Param(model.Times)
model.CreditSpread = pyo.Param()
model.B1 = pyo.Param(model.NumScenarios)
# model.B2 = pyo.Param(model.NumScenarios)


############################################################ ----- VARIABLES ----- ############################################################
model.Wealth = pyo.Var(model.Times, bounds=(1, 500000000))
model.Holdings_BOS = pyo.Var(model.Assets, model.Times, domain=pyo.PositiveReals)  # x_i_h_n
model.Holdings_EOS = pyo.Var(model.Assets, model.Times, domain=pyo.PositiveReals)    # x_i_n
model.Buy = pyo.Var(model.Assets, model.Times, domain=pyo.PositiveReals)
model.Sell = pyo.Var(model.Assets, model.Times, domain=pyo.PositiveReals)
model.Cash = pyo.Var(model.Times, bounds=(1000000, 5000000))
model.Debt = pyo.Var(model.Times, bounds=(0, 5000000))
model.Debt_plus = pyo.Var(model.Times, bounds=(0, 5000000), initialize=1)
model.Debt_minus = pyo.Var(model.Times, bounds=(0, 5000000), initialize=1)
# model.P2 = pyo.Var(model.NumScenarios, model.NumScenarios, domain=pyo.PercentFraction)  # Square matrix - Dimension: # of nodes in that stage


############################################################ ----- CONSTRAINTS ----- ############################################################
def debt_equation(model, t):

    if t == 0:
        return model.Debt[t] == 0
    else:
        return model.Debt[t] == model.Debt[t-1] + model.Debt_plus[t] - model.Debt_minus[t]
model.DebtEquation = pyo.Constraint(model.Times, rule=debt_equation)


def only_long_constraint(model, i, t):
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return model.Sell[i, t] <= model.Holdings_EOS[i,t-1] * (1 + model.Returns[i,t])
model.OnlyLong = pyo.Constraint(model.Assets, model.Times, rule=only_long_constraint)


def starting_holdings_decision(model, i, t): #x_i_0

    if t == 0:
        return model.Holdings_EOS[i, t] == model.InitialHoldings[i] + model.Buy[i, t] - model.Sell[i, t]
    else:
        return pyo.Constraint.Skip
model.HoldingsFirstDecision = pyo.Constraint(model.Assets, model.Times, rule=starting_holdings_decision)


def holdings_pre_decision(model, i, t):

    if t == 0:
        return model.Holdings_BOS[i, t] == model.InitialHoldings[i]
    else:
        return model.Holdings_BOS[i, t] == model.Holdings_EOS[i, t - 1] * (1 + model.Returns[i, t]) - model.Sell[i, t]
model.HoldingsPreDecision = pyo.Constraint(model.Assets, model.Times, rule=holdings_pre_decision)


def portfolio_upper_bound(model, i, t):

    if t == 0:
        return pyo.Constraint.Skip
    else:
        return model.Holdings_EOS[i, t] <= model.Upper * sum(model.Holdings_EOS[i, t] for i in model.Assets)
model.UpperBound = pyo.Constraint(model.Assets, model.Times, rule=portfolio_upper_bound)


def holdings_post_decision(model, i, t):

    if t == 0:
        return pyo.Constraint.Skip
    else:
        return model.Holdings_EOS[i, t] == model.Buy[i, t] + model.Holdings_BOS[i, t]
model.HoldingsPostDecision = pyo.Constraint(model.Assets, model.Times, rule=holdings_post_decision)


def starting_cash_evolution(model, t):

    if t == 0:
        return model.Cash[t] == model.InitialCash + sum([model.Sell[i, t] * (1 - model.Fee) for i in model.Assets]) \
                                - sum([model.Buy[i, t] * (1 + model.Fee) for i in model.Assets])
    else:
        return pyo.Constraint.Skip
model.StartingCashEvolution = pyo.Constraint(model.Times, rule=starting_cash_evolution)


def max_borrowing(model, t):

    return model.Debt[t] <= model.MaxBorr * model.TargetEstate[t]
model.MaxBorrowing = pyo.Constraint(model.Times, rule=max_borrowing)


def cash_evolution(model, t):

    if t == 0:
        return model.Debt_minus[t] == 0

    elif t < model.TargetEstate_stage:
        return model.Cash[t] == model.Income[t] - model.Consumption[t] + \
                                model.Cash[t - 1] * (1 + model.RiskFree[t]) + \
                                sum([model.Sell[i, t] * (1 - model.Fee) for i in model.Assets]) - \
                                sum([model.Buy[i, t] * (1 + model.Fee) for i in model.Assets]) + \
                                model.Debt_plus[t] - \
                                model.Debt[t - 1] * (model.ShortRate[t] + model.CreditSpread) - \
                                model.Debt_minus[t] - \
                                model.TargetEstate[t]

    else:
        return model.Cash[t] == model.Income[t] - model.Consumption[t] + \
                                model.Cash[t - 1] * (1 + model.RiskFree[t]) + \
                                sum([model.Sell[i, t] * (1 - model.Fee) for i in model.Assets]) - \
                                sum([model.Buy[i, t] * (1 + model.Fee) for i in model.Assets]) + \
                                model.Debt_plus[t] - \
                                model.Debt[t - 1] * (model.ShortRate[t] + model.CreditSpread) - \
                                model.Debt_minus[t]
model.CashEvolution = pyo.Constraint(model.Times, rule=cash_evolution)


def wealth_evolution(model, t):

    if t == 0:
        return model.Wealth[t] == model.InitialCash + model.InitialPortfolioValue

    elif t >= model.TargetEstate_stage:
        return model.Wealth[t] == sum([model.Holdings_EOS[i, t - 1] * (1 + model.Returns[i, t]) for i in model.Assets]) \
                                    + model.Cash[t]  \
                                    - model.Debt[t]  \
                                    + model.TargetEstate[t]

    else:
        return model.Wealth[t] == sum([model.Holdings_EOS[i, t - 1] * (1 + model.Returns[i, t]) for i in model.Assets]) \
                                    + model.Cash[t] \
                                    - model.Debt[t]
model.WealthConstraint = pyo.Constraint(model.Times, rule=wealth_evolution)


############################################################ ----- OBJECTIVE FUNCTION ----- ############################################################
def objective_function(model):

    return (1 - model.Lambda) * model.Wealth[model.TargetWealth_stage] + \
        model.Lambda * ((-model.TargetEstate[model.TargetEstate_stage]
                        + model.Wealth[model.TargetEstate_stage]) +
                        (- model.TargetWealth[model.TargetWealth_stage]
                        + model.Wealth[model.TargetWealth_stage]))
model.ObjectiveFunction = pyo.Objective(rule=objective_function, sense=-1)


############################################################ ----- SSD CONSTRAINTS ----- ############################################################
SSD = False
if SSD:
    def SSD_POSITIVITY_RULE(model, t, t_hat):

        return model.P2[t, t_hat] >= 0

    model.SSD_positivity_rule = pyo.Constraint(model.NumScenarios, model.NumScenarios, rule=SSD_POSITIVITY_RULE)


    def SSD_ROW_RULE(model, t):

        return sum(model.P2[t, t_hat] for t_hat in model.Times) == 1

    model.SSD_row_rule = pyo.Constraint(model.NumScenarios, rule=SSD_ROW_RULE)


    def SSD_COLUMN_RULE(model, t_hat):

        return sum(model.P2[t, t_hat] for t in model.Times) == 1

    model.SSD_column_rule = pyo.Constraint(model.NumScenarios, rule=SSD_COLUMN_RULE)


    def SSD(model, t):

        if t == model.TargetWealth_stage:
            return model.Wealth[model.TargetWealth_stage] >= sum([model.P2[t, t_hat] * model.B1[t_hat] for t_hat in model.NumScenarios])
        else:
            return pyo.Constraint.Skip

    model.ssd = pyo.Constraint(model.NumScenarios, rule=SSD)


############################################################ ----- FINAL WEALTH ----- ############################################################
finalWealth = []
finalCash = []

wealth = pd.DataFrame(index=range(10))
cash = pd.DataFrame(index=range(10))
debt = pd.DataFrame(index=range(10))
commodity = pd.DataFrame(index=range(10))
us_stocks = pd.DataFrame(index=range(10))
em_stocks = pd.DataFrame(index=range(10))
eu_stocks = pd.DataFrame(index=range(10))
asia_stocks = pd.DataFrame(index=range(10))
us_bonds = pd.DataFrame(index=range(10))
eu_bonds = pd.DataFrame(index=range(10))
em_bonds = pd.DataFrame(index=range(10))
portfolio_value = pd.DataFrame(index=range(10))

output_path = r"C:/Users/giovanna/Desktop/MPS Results"

lambda_param = 1    #zero = max wealth
target = 10000000
scenarios = 1112
years = 10
version = 1

for k in range(scenarios):
    print('Run optimization number: ' + str(k+1) + ' out of ' + str(scenarios))

    instance = model.create_instance(r"C:\Users\giovanna\PycharmProjects\pythonProject1\venv\Scripts\Scenario_" + str(k) + '_lambda_' + str(lambda_param) + r".dat")
    opt = pyo.SolverFactory('cplex', tee=False)
    results = opt.solve(instance, tee=False)

    for t in range(years):
        if t == 0:
            wealth['Scenario_' + str(k)] = pyo.value(instance.Wealth[t])
            cash['Scenario_' + str(k)] = pyo.value(instance.Cash[t])
            debt['Scenario_' + str(k)] = pyo.value(instance.Debt[t])
            commodity['Scenario_' + str(k)] = pyo.value(instance.Holdings_EOS['COMMODITY', t])
            us_stocks['Scenario_' + str(k)] = pyo.value(instance.Holdings_EOS['US_STOCKS', t])
            em_stocks['Scenario_' + str(k)] = pyo.value(instance.Holdings_EOS['EM_STOCKS', t])
            eu_stocks['Scenario_' + str(k)] = pyo.value(instance.Holdings_EOS['EU_STOCKS', t])
            asia_stocks['Scenario_' + str(k)] = pyo.value(instance.Holdings_EOS['ASIA_STOCKS', t])
            us_bonds['Scenario_' + str(k)] = pyo.value(instance.Holdings_EOS['US_BONDS', t])
            eu_bonds['Scenario_' + str(k)] = pyo.value(instance.Holdings_EOS['EU_BONDS', t])
            em_bonds['Scenario_' + str(k)] = pyo.value(instance.Holdings_EOS['EM_BONDS', t])
        else:
            wealth['Scenario_' + str(k)].loc[t] = pyo.value(instance.Wealth[t])
            cash['Scenario_' + str(k)].loc[t] = pyo.value(instance.Cash[t])
            debt['Scenario_' + str(k)].loc[t] = pyo.value(instance.Debt[t])
            commodity['Scenario_' + str(k)].loc[t] = pyo.value(instance.Holdings_EOS['COMMODITY', t])
            us_stocks['Scenario_' + str(k)].loc[t] = pyo.value(instance.Holdings_EOS['US_STOCKS', t])
            em_stocks['Scenario_' + str(k)].loc[t] = pyo.value(instance.Holdings_EOS['EM_STOCKS', t])
            eu_stocks['Scenario_' + str(k)].loc[t] = pyo.value(instance.Holdings_EOS['EU_STOCKS', t])
            asia_stocks['Scenario_' + str(k)].loc[t] = pyo.value(instance.Holdings_EOS['ASIA_STOCKS', t])
            us_bonds['Scenario_' + str(k)].loc[t] = pyo.value(instance.Holdings_EOS['US_BONDS', t])
            eu_bonds['Scenario_' + str(k)].loc[t] = pyo.value(instance.Holdings_EOS['EU_BONDS', t])
            em_bonds['Scenario_' + str(k)].loc[t] = pyo.value(instance.Holdings_EOS['EM_BONDS', t])

export = True
if export:
    wealth.to_csv(output_path + '/Wealth/Lambda_' + str(lambda_param) + '/wealth.csv')
    debt.to_csv(output_path + '/Debt/Lambda_' + str(lambda_param) + '/debt.csv')
    cash.to_csv(output_path + '/Cash/Lambda_' + str(lambda_param) + '/cash.csv')
    commodity.to_csv(output_path + '/Asset Allocation/Lambda_' + str(lambda_param) + '/commodity.csv')
    us_stocks.to_csv(output_path + '/Asset Allocation/Lambda_' + str(lambda_param) + '/us-stock.csv')
    em_stocks.to_csv(output_path + '/Asset Allocation/Lambda_' + str(lambda_param) + '/em-stock.csv')
    eu_stocks.to_csv(output_path + '/Asset Allocation/Lambda_' + str(lambda_param) + '/eu-stock.csv')
    asia_stocks.to_csv(output_path + '/Asset Allocation/Lambda_' + str(lambda_param) + '/asia-stock.csv')
    us_bonds.to_csv(output_path + '/Asset Allocation/Lambda_' + str(lambda_param) + '/us-bond.csv')
    eu_bonds.to_csv(output_path + '/Asset Allocation/Lambda_' + str(lambda_param) + '/eu-bond.csv')
    em_bonds.to_csv(output_path + '/Asset Allocation/Lambda_' + str(lambda_param) + '/em-bond.csv')

