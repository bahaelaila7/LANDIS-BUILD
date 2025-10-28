import numpy as np
from dataclass import dataclass
from types import List

RNG = np.random.default_rng(seed=111)
LEAF_BIOMASS_PERCENTAGE = 0.35
SPECIES = 20
MAPCODES = 20
AGECLASSES = 10
ECOREGIONS = 4
MAX_AGE = np.full(SPECIES, 100)
DEF = np.full(SPECIES, 0)
B_MAX_SPP = np.full((SPECIES, ECOREGIONS), 10000)
ANPP_MAX_SPP = np.full((SPECIES, ECOREGIONS), 300)
ProbMort_SPP = np.full((SPECIES, ECOREGIONS), 0.03)


D = 0.01
S = 1


@dataclass
class Cohort:
    age: int
    species: int
    biomass: float


@dataclass
class Site:
    cohorts: List[Cohort]
    ecoregion: int
    deadWoodyPool: float
    deadWoodyPoolDecayRate: float
    deadNonWoodyPool: float
    deadNonWoodyPoolDecayRate: float
    prevYearMortality: float


sites = [Site() for _ in MAPCODES]


def step():
    for site in sites:
        growthReduction = 1
        capacityReduction = 1
        ######################
        # Age Related morality
        #####################
        # TODO spinup mortality fraction
        M_AGE_ij = np.array(
            [
                c.biomass
                * max(1, c.biomass * np.exp(D * (c.age / MAX_AGE[c.species] - 1)))
                for c in site.cohorts
            ]
        )
        ######################
        # Biomass Potential Growth and Competition Index
        #####################
        B_ij = np.array([c.biomass for c in site.cohorts])
        B = B_ij.sum()
        B_MAX_i = np.array(
            [
                B_MAX_SPP[c.species, site.ecoregion] * capacityReduction
                for c in site.cohorts
            ]
        )
        B_POT_ij = np.clip(B_MAX_i - B + B_ij, a_min=1, a_max=None)
        if capacityReduction == 1:
            B_POT_ij = np.clip(B_POT_ij, a_min=site.prevYearMortality, a_max=None)
        B95 = np.power(B_ij, 0.95)
        C_ij = B95 / B95.sum()
        B_PM_ij = C_ij

        ######################
        # Development Limit
        #####################
        B_AP_ij = B_ij / B_POT_ij
        B_AP_S_ij = np.power(B_AP_ij, S)
        DevLim_ij = np.clip(B_AP_S_ij * np.exp(1 - B_AP_S_ij), a_max=1, a_min=None)

        ######################
        # ANPP
        #####################
        ANPP_MAX_C_ij = (
            np.array([ANPP_MAX_SPP[c.species, site.ecoregion] for c in site.cohorts])
            * C_ij
        )
        ANPP_ACT_ij = np.clip(
            ANPP_MAX_C_ij * DevLim_ij - M_AGE_ij,
            a_min=0,
            a_max=None,
        )

        ######################
        # Development Mortality
        #####################
        M_BIO_ij = np.where(
            B_AP_ij > 1, ANPP_MAX_C_ij, ANPP_MAX_C_ij * (2 * B_AP_ij) / (1 + B_AP_ij)
        )
        M_BIO_ij = np.minimum(ANPP_MAX_C_ij, M_BIO_ij, B_ij)
        if growthReduction > 0:
            M_BIO_ij *= 1 - growthReduction
        M_BIO_ij = np.clip(M_BIO_ij - M_AGE_ij, a_min=0, a_max=None)
        M_BIO_ij = np.minimum(M_BIO_ij, ANPP_ACT_ij)
        ######################
        # Total Mortality
        #####################
        M_TOT_ij = M_BIO_ij + M_AGE_ij
        assert np.all(M_TOT_ij <= B_ij)
        ######################
        # Random Mortality
        #####################
        mortRNG_ij = RNG.uniform(shape=len(site.cohorts))
        probM_ij = np.array([ProbMort_SPP[c.species, site.ecoregion] for c in site])
        M_TOT_ij = np.where(mortRNG_ij < probM_ij, B_ij, M_TOT_ij)

        ######################
        # Defoliation
        #####################
        defoliationFactor_ij = np.array([DEF[c.species] for c in site])
        defoliationLoss_ij = (
            LEAF_BIOMASS_PERCENTAGE * ANPP_ACT_ij * defoliationFactor_ij
        )
        ######################
        # Shade
        #####################
        # TODO

        dB_ij = ANPP_ACT_ij - M_TOT_ij - defoliationLoss_ij
        nB_ij = B_ij + dB_ij
