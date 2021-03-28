# -------------------------------------------------------------------------------
# Name:        Group Project 2 (GP2)
# Course:      ESA EO training course 2021
#
# Author:      Salvatore Vicinanza
#
# Created:     25-03-2021
# Copyright:   no copyright
# Licence:
#
# Using routine for radiometric instrument sizing
# -------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import math


####################################################################
# SAR
####################################################################
# Constants
height = 833e3  # [m] highest altitude point of S-3 orbit
diameters = [1.2, 1.4, 1.6, 1.8, 2.0]  # [m] antenna diameter
La = [1, 2]  # [dB] loss factor
bands = ['Ku', 'Ka']  # signal band
frequencies = [13.5e9, 35.75e9]  # [Hz] Ku-band and Ka-band frequencies
c = 299792458  # [m/s] speed of light


# ----------------------------------------------------------
# Additional functions needed
# ----------------------------------------------------------
def boresight_directionality(alpha, wl, d):
    """
    Parameters
    ----------
    alpha : [float]
        directivity factor [-]

    wl : [float]
        wavelength characteristic of the band used [m]

    d : [float]
        diameter of the antenna [m]

    Returns
    -------
    D_bore : [float]
        boresight antenna directionality [dB]
    """

    D_bore = 10 * np.log10( alpha * (np.pi*d/wl) **2 )
    return D_bore


def boresight_gain(D_bore,La):
    """
    Parameters
    ----------
    D_bore : [float]
        boresight directionality [dB]

    La : [float]
        loss factor [dB]

    Returns
    -------
    G_bore : [float]
        boresight antenna gain [dB]
    """

    G_bore = D_bore - La
    return G_bore


def dG_newold(G_new, G_old):
    """
    Parameters
    ----------
    G_new : [float]
        antenna gain - new instrument [dB]

    G_old : [float]
        antenna gain - old instrument [dB]


    Returns
    -------
    dG_newold : [float]
        delta_G^2 [dB]
    """

    G_new = 10**(G_new/10)
    G_old = 10**(G_old/10)
    dG2 = 10 * np.log10(G_new**2/G_old**2)
    return dG2


def HPBW(beta, wl, d):
    """
    Parameters
    ----------
    beta : [float]
        beam factor

    wl : [float]
        wavelength characteristic of the band used

    d : [float]
        antenna diameter

    Returns
    -------
    HPBW : [float]
        half-power beam-width [deg]
    """

    HPBW = beta * wl / d
    return HPBW


def HPFP(HPBW, height):
    """
    Parameters
    ----------
    HPBW : [float]
        half-power beam-width [deg]

    height : [float]
        orbital height [m]

    Returns
    -------
    HPFP : [float]
        half-power footprint [km]
    """

    HPFP = 2 * height * np.tan(np.deg2rad(HPBW/2))
    return HPFP/1e3


def main_formulas():
    # Constants based on n=1 and edge illumination=-16dB
    beta = 68.2
    alpha = 0.850
    G_old = 42.36  # [dB] gain antenna boresight for Sentinel-3

    # Initialization Dataframes
    G = pd.DataFrame(columns=['band', 'diameter', 'value'])
    dG = pd.DataFrame(columns=['band', 'diameter', 'value'])
    H = pd.DataFrame(columns=['band', 'diameter', 'value'])
    FP = pd.DataFrame(columns=['band', 'diameter', 'value'])
    
    for item in range(len(diameters)):
        d = diameters[item]
        for i in range(len(bands)):
            wl = c / frequencies[i]
            D = boresight_directionality(alpha, wl, d)
            new_df_G = pd.DataFrame([[bands[i], d, boresight_gain(D,La[i])]], columns=['band', 'diameter', 'value'])
            G = G.append(new_df_G, ignore_index=True)
            
            new_df_dG = pd.DataFrame(columns=['band', 'diameter', 'value'], data=[[bands[i], d, dG_newold(G['value'].to_numpy()[-1], G_old)]])
            dG = dG.append(new_df_dG, ignore_index=True)

            new_df_H = pd.DataFrame(columns=['band', 'diameter', 'value'], data=[[bands[i], d, HPBW(beta,wl,d)]])
            H = H.append(new_df_H, ignore_index=True)
            
            new_df_FP = pd.DataFrame(columns=['band', 'diameter', 'value'], data=[[bands[i], d, HPFP(H['value'].to_numpy()[-1], height)]])
            FP = FP.append(new_df_FP, ignore_index=True)
            
    return G, dG, H, FP


def main_scaling():
    # -------------------------------------
    # Sentinel-3 values
    # -------------------------------------
    G_bore_S3 = 42.36 # [dB]
    H_S3 = 1.28 # [deg] at Ku-band
    d_S3 = 1.2 # [m] diameter antenna
    dG2_S3 = 0
    FP_S3 = 18.6e+3 # [km] half-power footprint
    La_S3 = 1.0
    wl_S3 = c/frequencies[0]

    # -------------------------------------
    # Sentinel-3 NG scaled values
    # -------------------------------------
    # Initialization Dataframes
    G = pd.DataFrame(columns=['band', 'diameter', 'value'])
    dG = pd.DataFrame(columns=['band', 'diameter', 'value'])
    H = pd.DataFrame(columns=['band', 'diameter', 'value'])
    FP = pd.DataFrame(columns=['band', 'diameter', 'value'])

    for item in range(len(diameters)):
        d = diameters[item]
        for i in range(len(bands)):
            wl = c / frequencies[i]
            D_S3 = G_bore_S3 + La_S3
            D = 10**(D_S3/10) * ((d * wl_S3)** 2)/( (d_S3 * wl) ** 2)
            new_df_G = pd.DataFrame([[bands[i], d, boresight_gain(10*np.log10(D), La[i])]], columns=['band', 'diameter', 'value'])
            G = G.append(new_df_G, ignore_index=True)

            new_df_dG = pd.DataFrame(columns=['band', 'diameter', 'value'], data=[[bands[i], d, dG_newold(G['value'].to_numpy()[-1], G_bore_S3)]])
            dG = dG.append(new_df_dG, ignore_index=True)

            new_df_H = pd.DataFrame(columns=['band', 'diameter', 'value'], data=[[bands[i], d, H_S3 * (d_S3*wl)/(d*wl_S3)]])
            H = H.append(new_df_H, ignore_index=True)

            Fp = FP_S3 * np.tan(np.deg2rad(H['value'].to_numpy()[-1]/2)) / np.tan(np.deg2rad(H_S3/2))
            new_df_FP = pd.DataFrame(columns=['band', 'diameter', 'value'], data=[[bands[i], d, Fp/1e+3]])
            FP = FP.append(new_df_FP, ignore_index=True)

    return G, dG, H, FP


# ------------------------------------------------------------------
# SAR - PART 2
# ------------------------------------------------------------------
# def horizontal_resolution(B):
#     c = 299792458
#     d_R = c/(2*B)
#     hor_res = 2*np.sqrt(2*h*d_R)
#     return hor_res

def SNR():

    S_prev = 0
    res_cell = list(9)

    for i in range(len(res_cell)):
        S = ( (Pt * (wl * G_max)**2 * tau)/((4*np.pi)**3 * r**4 * k*(T_a+(F_1-1)*T_0)) ) * s_i**2
        S = S_prev + S
        S_prev = S

    return S_prev


####################################################################
# Radiometer
####################################################################
diameters_Rad = [0.8, 1.0, 1.2, 1.4]  # [m] antenna diameter
frequency = 53.6e9  # [Hz] frequency


def main_scaling_Rad():
    # -------------------------------------
    # Sentinel-3 values
    # -------------------------------------
    H_S3 = 1.66  # [deg]
    d_S3 = 0.6  # [m] diameter antenna
    FP_S3 = HPFP(H_S3, height) * 1e+3  # [m] half-power footprint
    wl_S3 = c / 23.8e+9

    # -------------------------------------
    # Sentinel-3 NG scaled values
    # -------------------------------------
    # Initialization Dataframes
    H = pd.DataFrame(columns=['diameter', 'value'])
    FP = pd.DataFrame(columns=['diameter', 'value'])
    RFP = pd.DataFrame(columns=['diameter', 'value'])

    for item in range(len(diameters_Rad)):
        d = diameters_Rad[item]
        wl = c / frequency

        new_df_H = pd.DataFrame(columns=['diameter', 'value'], data=[[d, H_S3 * (d_S3*wl)/(d*wl_S3)]])
        H = H.append(new_df_H, ignore_index=True)

        Fp = FP_S3 * np.tan(np.deg2rad(H['value'].to_numpy()[-1]/2)) / np.tan(np.deg2rad(H_S3/2))
        new_df_FP = pd.DataFrame(columns=['diameter', 'value'], data=[[d, Fp/1e+3]])
        FP = FP.append(new_df_FP, ignore_index=True)

        new_df_RFP = pd.DataFrame(columns=['diameter', 'value'], data=[[d, 3*FP['value'].to_numpy()[-1]]])
        RFP = RFP.append(new_df_RFP, ignore_index=True)

    return H, FP, RFP


def main_formulas_Rad():
    # Constants based on n=1 and edge illumination=-16dB
    n = 2  # [-] to get the lowest sidelobe (useful to get the next values from table)
    edge = -20  # [dB] to get the lowest sidelobe (useful to get the next values from table)
    beta = 75.6

    # Initialization Dataframes
    H = pd.DataFrame(columns=['diameter', 'value'])
    FP = pd.DataFrame(columns=['diameter', 'value'])
    RFP = pd.DataFrame(columns=['diameter', 'value'])

    for item in range(len(diameters_Rad)):
        d = diameters_Rad[item]
        wl = c / frequency

        new_df_H = pd.DataFrame(columns=['diameter', 'value'], data=[[d, HPBW(beta, wl, d)]])
        H = H.append(new_df_H, ignore_index=True)

        new_df_FP = pd.DataFrame(columns=['diameter', 'value'], data=[[d, HPFP(H['value'].to_numpy()[-1], height)]])
        FP = FP.append(new_df_FP, ignore_index=True)

        new_df_RFP = pd.DataFrame(columns=['diameter', 'value'], data=[[d, 3*FP['value'].to_numpy()[-1]]])
        RFP = RFP.append(new_df_RFP, ignore_index=True)

    return H, FP, RFP


if __name__ == '__main__':

    # ------------------------------------------------------
    # SAR Altimeter (PART 1)
    # ------------------------------------------------------
    G1, dG1, H1, FP1 = main_scaling()
    G2, dG2, H2, FP2 = main_formulas()

    G = pd.concat([G1, G2], keys={'scaling', 'formulas'})
    dG = pd.concat([dG1, dG2], keys={'scaling', 'formulas'})
    H = pd.concat([H1, H2], keys={'scaling', 'formulas'})
    FP = pd.concat([FP1, FP2], keys={'scaling', 'formulas'})

    # ------------------------------------------------------
    # SAR Altimeter (PART 2)
    # ------------------------------------------------------


    # ------------------------------------------------------
    # High Resolution Radiometer for coastal ocean observation
    # ------------------------------------------------------
    H3, FP3, RFP3 = main_scaling_Rad()
    H4, FP4, RFP4 = main_formulas_Rad()

    H_Rad = pd.concat([H3, H4], keys={'scaling', 'formulas'})
    FP_Rad = pd.concat([FP3, FP4], keys={'scaling', 'formulas'})
    RFP_Rad = pd.concat([RFP3, RFP4], keys={'scaling', 'formulas'})

    n = 0