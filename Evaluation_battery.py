import math
import numpy as np


#  Solar Energy Utilization Efficiency (SEUE)
def SolarEnergyUtilizationEfficiency(actual, predict):
    seue = predict / actual
    return seue * 100


#  Grid Stability Index (GSI)
def GridStability(actual, predict):
    gsi = 1 - abs(actual - predict) / actual
    return gsi * 100


# Scalability Index (SI)
def Scalability(actual, predict):
    si = predict / actual
    return si * 100


#  Emissions Deviation (lower is better)
def Emissions(actual, predict):
    emissions = abs(actual - predict)
    return emissions


#  State of Charge (SoC)
def StateOfCharge(actual, predict):
    soc = predict / actual
    return soc * 100


#  Depth of Discharge (DoD)
def DepthOfDischarge(actual, predict):
    dod = 1 - (predict / actual)
    return dod * 100


#  Charge–Discharge Efficiency
def ChargeDischargeEfficiency(actual, predict):
    cde = predict / actual
    return cde * 100


# 8. Voltage Regulation Index (VRI)
def VoltageRegulationIndex(actual, predict):
    vri = abs(actual - predict) / actual
    return vri * 100


# 9. Operating Cost Deviation
def OperatingCost(actual, predict):
    oc = abs(actual - predict)
    return oc


#  Cost of Battery Degradation
def BatteryDegradationCost(actual, predict):
    bdc = abs(actual - predict) / actual
    return bdc * 100


#  Net Present Value Difference
def NetPresentValue(actual, predict):
    npv = predict - actual
    return npv

# 12. Economic Analysis (Cost–Benefit based)
def EconomicAnalysis(actual, predict):
    ea = predict / actual
    return ea * 100

# 13. Load Demand Analysis
def LoadDemandAnalysis(actual, predict):
    lda = 1 - abs(actual - predict) / actual
    return lda * 100


def evaluation_battery(actual, predict):
    Values = np.asarray([
        SolarEnergyUtilizationEfficiency(actual, predict),
        GridStability(actual, predict),
        Scalability(actual, predict),
        Emissions(actual, predict),
        StateOfCharge(actual, predict),
        DepthOfDischarge(actual, predict),
        ChargeDischargeEfficiency(actual, predict),
        VoltageRegulationIndex(actual, predict),
        OperatingCost(actual, predict),
        BatteryDegradationCost(actual, predict),
        NetPresentValue(actual, predict),
        EconomicAnalysis(actual, predict),
        LoadDemandAnalysis(actual, predict)
    ])
    return Values



