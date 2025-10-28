import numpy as np
from dataclasses import dataclass
from sortedcontainers import SortedKeyList

RNG = np.random.default_rng(seed=111)
LEAF_BIOMASS_PERCENTAGE = 0.35
SPECIES = 20
MAPCODES = 20
AGECLASSES = 60
ECOREGIONS = 4
MAX_AGE = np.full(SPECIES, 300)
DEF = np.full(SPECIES, 0.0)
S = np.full(SPECIES, 0.5)
D = np.full(SPECIES, 5)
B_MAX_SPP = np.full((SPECIES, ECOREGIONS), 10000)
ANPP_MAX_SPP = np.full((SPECIES, ECOREGIONS), 300)
ProbMort_SPP = np.full((SPECIES, ECOREGIONS), 0.03)


@dataclass
class Cohort:
    age: int
    species: int
    biomass: float


@dataclass
class Site:
    cohorts: SortedKeyList[Cohort]
    ecoregion: int = 0
    deadWoodyPool: float = 0.0
    deadWoodyPoolDecayRate: float = 0.0
    deadNonWoodyPool: float = 0.0
    deadNonWoodyPoolDecayRate: float = 0.0
    prevYearMortality: float = 0.0
    growthReduction: float = 0.0
    capacityReduction: float = 0.0
    AGNPP: float = 0.0
    defoliationLoss: float = 0.0


def make_cohort():
    species = RNG.integers(SPECIES)
    age = RNG.integers(AGECLASSES) * 5
    biomass = np.exp(-((age - 40) ** 2) / (40**2)) * (0.5 + RNG.uniform() / 2) * 300
    assert biomass > 0
    return Cohort(species=species, age=age, biomass=biomass)


def make_site():
    cs = SortedKeyList((make_cohort() for _ in range(20)), lambda c: c.age)
    return Site(cs)


def step(sites, current_time, spinup_mortality_fraction=0.0):
    for site in sites:
        growthReduction = site.growthReduction
        capacityReduction = site.capacityReduction
        ######################
        # Site biomasses
        #####################
        B_ij = np.array([c.biomass for c in site.cohorts])
        B = B_ij.sum()
        ######################
        # Age Related morality
        #####################
        # TODO spinup mortality fraction
        M_AGE_ij = np.array(
            [
                c.biomass * np.exp(D[c.species] * (c.age / MAX_AGE[c.species] - 1))
                for c in site.cohorts
            ]
        )
        M_AGE_ij = np.clip(M_AGE_ij, a_max=B_ij, a_min=0.0)
        if current_time <= 0 and spinup_mortality_fraction > 0:
            M_AGE_ij += B_ij * spinup_mortality_fraction
        M_AGE_ij = np.clip(M_AGE_ij, a_max=B_ij, a_min=0.0)
        ######################
        # Biomass Potential Growth
        #####################
        B_MAX_i = np.array(
            [
                B_MAX_SPP[c.species, site.ecoregion] * capacityReduction
                for c in site.cohorts
            ]
        )
        B_POT_ij = np.clip(B_MAX_i - B + B_ij, a_min=1.0, a_max=None)
        if capacityReduction >= 1:  # No capacity reduction due to harvest
            B_POT_ij = np.clip(B_POT_ij, a_min=site.prevYearMortality, a_max=None)
        ######################
        # Competition Index
        #####################
        B95 = np.clip(np.power(B_ij, 0.95), a_min=1.0, a_max=None)
        # min value is 1.0 as shown in code
        # since the original code is sequential, the original code excludes
        # They disallow competition of a cohort that has the same species
        # but its age is current cohort age + 1
        # but here we SIMD baby, so who cares (famous last words)
        C_ij = B95 / B95.sum()
        # B_PM_ij = C_ij

        #######################
        # Development Limit
        #####################
        # Simplifying original code
        # First equation:
        #    //B_PM = indexC
        #    //B_AP = B/B_POT
        #    double actualANPP = maxANPP * Math.E * Math.Pow(B_AP, growthShape) * Math.Exp(-1 * Math.Pow(B_AP, growthShape)) * B_PM;

        # //first caching repeated expression
        # double B_AP_S = Math.Pow(B_AP, growthShape)
        # double actualANPP = maxANPP * Math.E * B_AP_S * Math.Exp(-1 * B_AP_S)* B_PM ;
        # //moving E inside Exp, and rearranging abit
        # double actualANPP = maxANPP * B_PM * B_AP_S * Math.Exp(1.0 - B_AP_S);
        # //lets cache stuff after B_PM
        # double DevLim =  B_AP_S * Math.Exp(1.0 - B_AP_S);
        # then final eq
        # double actualANPP = maxANPP * B_PM * DevLim;

        # Next equation:

        # // Calculated actual ANPP can not exceed the limit set by the
        # //  maximum ANPP times the ratio of potential to maximum biomass.
        # //  This down regulates actual ANPP by the available growing space.

        #    actualANPP = Math.Min(maxANPP * B_PM, actualANPP);
        # in other words, B_AP_S * Math.Exp(1 - B_AP_S)
        # therefore we can:
        # double DevLim = min(1, B_AP_S * Math.Exp(1 - B_AP_S))
        # double maxANPP_C = maxANPP * B_PM; //factored out for future reuse
        # double actualANPP = maxANPP_C * DevLim;
        # """

        B_AP_ij = B_ij / B_POT_ij
        Ss = np.array([S[c.species] for c in site.cohorts])
        B_AP_S_ij = np.power(B_AP_ij, Ss)
        DevLim_ij = np.clip(B_AP_S_ij * np.exp(1.0 - B_AP_S_ij), a_max=1, a_min=None)

        ######################
        # ANPP
        #####################
        # ANPP_MAX_C = ANPP_MAX * C_ij
        ANPP_MAX_C_ij = (
            np.array([ANPP_MAX_SPP[c.species, site.ecoregion] for c in site.cohorts])
            * C_ij
        )
        ANPP_ACT_ij = ANPP_MAX_C_ij * DevLim_ij
        if growthReduction > 0.0:
            ANPP_ACT_ij *= 1.0 - site.growthReduction

        site.AGNPP = ANPP_ACT_ij.sum()

        ######################
        # Development Mortality
        #####################
        # Simplifying original equations:
        #    if B_AP > 1.0:
        #        double M_BIO = maxANPP * B_PM; //we cached this as maxANPP_C
        #    else:
        #        double M_BIO = maxANPP * B_PM * (2.0 * B_AP)/(1.0 +B_AP);
        #
        # lets cache:
        # double M_mult =  (2.0 * B_AP)/(1.0 +B_AP)
        # We can:
        # M_BIO_ij = maxANPP_C * np.where(B_AP_ij > 1.0, 1.0, M_mult)
        # but there is later
        #    M_BIO = min(M_BIO, B)
        #    M_BIO = min(M_BIO, maxANPP * B_PM)
        # Therefore, M_mult cannot exceed one anyhow.
        # hence:
        # M_mult = min(1, (2.0 * B_AP)/(1.0 +B_AP))

        M_mult_ij = np.clip((2 * B_AP_ij) / (1 + B_AP_ij), a_max=1.0, a_min=None)
        M_BIO_ij = np.where(B_AP_ij > 1.0, ANPP_MAX_C_ij, ANPP_MAX_C_ij * M_mult_ij)
        M_BIO_ij = np.minimum(M_BIO_ij, B_ij)
        # not needed with M_mult_ij clipped a_max=1
        # M_BIO_ij = np.minimum(M_BIO_ij, ANPP_MAX_C_ij)
        if growthReduction > 0.0:
            M_BIO_ij *= 1.0 - growthReduction

        ######################
        # Discounting M_BIO_ij
        #####################
        # For the remaining equations, ANPP is discounted by M_AGE, but can't be below 1.0
        ANPP_ACT_ij = np.clip(ANPP_ACT_ij - M_AGE_ij, a_min=1.0, a_max=None)
        M_BIO_ij = np.clip(M_BIO_ij - M_AGE_ij, a_min=0, a_max=None)
        M_BIO_ij = np.minimum(M_BIO_ij, ANPP_ACT_ij)
        ######################
        # Total Mortality
        #####################
        M_TOT_ij = M_BIO_ij + M_AGE_ij
        diff_ij = B_ij - M_TOT_ij + 1e-8  # some eps to avoid round up errors
        assert np.all(diff_ij >= 0), (
            diff_ij >= 0,
            list(zip(diff_ij >= 0, M_TOT_ij, B_ij, site.cohorts)),
        )
        ######################
        # Random Mortality
        #####################
        mortRNG_ij = RNG.uniform(size=len(site.cohorts))
        probM_ij = np.array(
            [ProbMort_SPP[c.species, site.ecoregion] for c in site.cohorts]
        )
        M_TOT_ij = np.where(mortRNG_ij < probM_ij, B_ij, M_TOT_ij)

        ######################
        # Defoliation
        #####################
        defoliationFactor_ij = np.array([DEF[c.species] for c in site.cohorts])
        defoliationLoss_ij = (
            LEAF_BIOMASS_PERCENTAGE * ANPP_ACT_ij * defoliationFactor_ij
        )
        ######################
        # Shade
        #####################
        # TODO

        dB_ij = ANPP_ACT_ij - M_TOT_ij - defoliationLoss_ij
        nB_ij = B_ij + dB_ij

        site.defoliationLoss = defoliationLoss_ij.sum()
        site.prevYearMortality = M_TOT_ij.sum()
        for b, c in zip(nB_ij, site.cohorts):
            c.biomass = b
            c.age += 1


if __name__ == "__main__":
    sites = [make_site() for _ in range(MAPCODES)]
    T0 = -200
    T1 = 100
    for current_time in range(T0, T1):
        step(sites, current_time)
