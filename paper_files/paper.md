---
title: 'gcms_data_analysis: A Python package for automated analysis of GC-MS data'
tags:
  - Python
  - gas-chromatography mass spectrometry
  - cheminformatics
  - pubchempy
  - functional groups
authors:
  - name: Matteo Pecchi
    orcid: 0000-0002-0277-4304
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Jillian L. Goldfarb
    orcid: 0000-0001-8682-9714
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Biological & Environmental Engineering, Cornell University, Ithaca NY 14853, USA
   index: 1
 - name: Smith School of Chemical and Biomolecular Engineering, Cornell University, Ithaca NY 14853, USA
   index: 2
date: 20 February 2024
bibliography: paper_files/paper.bib

---
# Summary

The lack of open-source tools to automate the analysis of large datasets from Gas Chromatography coupled with Mass Spectrometry (GC-MS) of biofuels results in time-consuming manual analyses of such data that employ sub-optimal methodologies, are difficult to replicate, and present an increased risk of error. We developed a Python code to automate GC-MS data analysis of complex, heterogeneous organic mixtures. This tool reduces GC-MS analysis time from hours/days to few seconds, avoiding human-errors and promoting standardization and best practices for GC-MS data handling.

# Statement of need

Gas Chromatography-Mass Spectrometry (GC-MS) is routinely used to identify composition and therefore quality and value of biofuels [@LuLiLuFanWei2017; @SharmaPedersenToorSchuurmanRosendahl2020; @SugumaranPrakashRamuAroraBansalKagdiyalSaxena2017] and optmize process conditions [@Grams2020; @SudibyoPecchiTester2022; @WangHanHuFuWang2020].

Compounds identification is usually done by spectral library matches [@HeracleousVassouLappasRodriguezChiabergeBianchi2022; @KostyukevichVlaskinZherebkerGrigorenkoBorisovaNikolaev2019; @ZhuGuoRosendahlToorZhangSunLuZhaoYangChen2022] and several open-source options are available for the task [@OCallaghan2012]. Quantification requires the construction of a calibration dataset [@KohansalSharmaHaiderToorCastelloRosendahlZimmermannPedersen2022; @PaniskoWietsmaLemmonAlbrechtHowe2015; @VilladsenDithmerForsbergBeckerRudolfIversenIversenGlasius2012].
Depending on the GC-MS column, compounds present, and their concentrations, derivatization may be necessary to improve identification, reproducibility and therefore final quantification of species present [@LeonardisChiabergeFioraniSperaBattistelBosettiCestiRealeDeAngelis2013; @MadsenJensenMørupHoulbergChristensenKlemmerBeckerIversenGlasius2016; @WangHanHuFuWang2020]. Derivatization requires its own calibration. Calibrations are often applied manually by researchers, especially when a semi-calibration mode is adopted to estimate the concentration of non-calibrated compounds based on similar compounds’ calibration curves.

A single bio-oil sample can contain hundreds of compounds [@HaiderCastelloRosendahl2020; @HanLi2021] and a typical experimental campaign involves dozens of samples. Calibrating with pure standards for hundreds of identified compound can be infeasible, therefore most bio-oil discussions rely on relative concentrations based on identified peak areas. Authors adopt strategies to infer calibrations for compounds without existing calibration [@AhnPandeyKim2011 @HubbleGoldfarb2021; @OlceseCarréAubrietDufour2013; @PatwardhanBrownShanks2011], using some sort of nearest neighbor method, where for each compound without a calibration curve, the calibration available for the “closest” compound is used instead. These methods are often manually implemented.

Another bottleneck in the analysis of GC-MS data is the classification of identified compounds based on their functional group(s). This is especially useful for heterogeneous mixtures such as bio-oils and other thermochemical and chemical processes where aggregated compositional results (e.g., the fraction of compounds having a given functional group present in each sample) can be used to derive mechanistic conclusions [@HeracleousVassouLappasRodriguezChiabergeBianchi2022; @SudibyoPecchiTester2022; @YangHeCorscaddenNiu2018; @ZhuGuoRosendahlToorZhangSunLuZhaoYangChen2022]. Automate these tasks could save time and minimize human error, while also increasing transparency in the methodological approach, specifically in how functional groups are attributed to each compound.

To address these needs, we designed an open-source Python tool to fully automate the handling of multiple GC-MS datasets simultaneously.  The tool relies on the Python package PubChemPy [@MattSwain2017] to access the PubChem website [@KimChen2023] and build its own database with all identified chemicals and their relevant chemical properties. It also performs functional group fragmentation using each compound’s SMILES [@Weininger1988] retrieved from PubChem, employing the automatic fragmentation algorithm developed by [@Müller2019] to split each molecule into its functional groups and then assign to each group its mass fraction. Functional groups can also be specified by the user using their SMARTS codes [@DaylightChemicalInformationSystems]. The code can apply calibrations for quantifying components present in a sample, with a semi-calibration option based on Tanimoto similarity with tunable thresholds [@Bajusz2015; @ChenReynolds2002]. For derivatized samples, the procedure is unchanged except that the non-derivatized form of each identified compound is used to retrieve chemical information so that non-derivatized and derivatized samples can be directly compared.

The tool groups replicate "files" into the same "sample" to evaluate reproducibility, and, besides producing single sample reports, it produces comprehensive reports that include all compounds across a series of samples as well as aggregated reports for all samples based on functional group mass fractions. The code also provides plotting functions for the aggregated reports to visualize the results and enable a preliminary investigation of their statistical significance.

# Automatic fragmentation and aggregation by functional group

A common strategy is to aggregate compounds based on functional group, either by count or concentration [@CastelloHaiderRosendahl2019; @HeracleousVassouLappasRodriguezChiabergeBianchi2022; @SudibyoPecchiTester2022; @ZhuGuoRosendahlToorZhangSunLuZhaoYangChen2022]. The implemented approach calculates the mass fraction of each functional group present in each compound, and use these mass fractions to split the concentration of each molecule into weighted averages of the different functional groups present. This allows to compute the aggregated value for all compounds present in the sample and has the advantage of fully accounting for all functional groups in a given sample. The fragmentation into functional groups relies on the fragmentation algorithm developed by [@Müller2019], available on [GitHub](https://github.com/simonmb/fragmentation_algorithm).

The aggregated concentration of each functional group ($C_{fg}$) in the sample is obtained as the sum over all n identified compounds in the sample of the concentration of the compound ($C_i$) multiplied by the mass fraction of the considered functional group in the compound itself ($mf_{fg, i}$). This equally applies to area, concentration, and yield:
$$ C_{fg} = \sum_{i=1}^{n} C_i \cdot mf_{fg,i} $$
This latter approach is automated in the present tool.

# Semi-calibration based on Tanimoto and molecular weight similarity

An effective approach, so far unexplored in the biofuel GC-MS literature but common in cheminformatics [@Butina1999], is to use molecular fingerprints (encodings of the molecular structure) to compute similarity indices to select the most similar calibrated compound; a common choice is to use the Tanimoto similarity index [@Bajusz2015; @ChenReynolds2002].
The present code allows for semi-calibration. For compounds without a calibration curve, their Tanimoto similarity with all calibrated compounds is evaluated. The calibrated compound with the highest similarity is selected (if more compounds share the same similarity, as happens for compounds of the same class, the compound with the closest molecular weight is selected). The Python package rdkit [@LandrumToscoKelleyetal] is used to convert canonical SMILES into molecular fingerprints and to compute the Tanimoto similarity among compounds.

## Error associated with Tanimoto thresholds and default parameters

The error associated with the use of the semi-calibration approach (for example, assuming use of the calibration curve of compound c1 to estimate the concentration of compound c2) can be evaluated, if both c1 and c2 calibration curves are known, as the average error between the calibration curves of those two compounds. This error equals the error that would come from using the calibration of c1 for estimating the concentration of c2, should c2 not be available. The average error can be computed using the following equation, where $cal_{c1}$ and $cal_{c2}$ are the calibration curves for c1 and c2 obtained by the linear interpolation of runs at known concentrations:

$$ \text{Average error [\%]} = \overline{\left( \frac{|cal_{c1} - cal_{c2}|}{cal_{c1}} \right)} \times 100 $$

We assessed this error for all combinations of calibrated compounds calibration dataset available in our GC-MS. The dataset comprises 89 compounds for which more than 4 points calibration (up to 6 points) is available in the form of mg/L vs detected area; this results in 3827 combinations. For each combination of compounds, the Tanimoto similarity and the molecular weight difference is also computed. Figure \autoref{fig:tanimoto_error} plots the average error as a function of the Tanimoto similarity of compounds and their molecular weight difference. The error decreases with the increasing Tanimoto similarity, while there seems to be no marked effect of molecular weight difference. Selecting a Tanimoto similarity threshold of 0.4 minimizes the risk of errors that are above one order of magnitude, while a similarity of 0.7 avoids this almost entirely (at least in our dataset). The percentage error threshold of 100% may seem unreasonably high, however the alternative would be to simply ignore all the compounds for which no calibration is available which also implies a large data loss. The selection of a Tanimoto similarity threshold that avoids unrealistic overestimation of concentrations (underestimations are less of a concern, since the alternative would be to ignore the compound entirely) improves the quality of GC-MS results. A similarity threshold of 0.4 was arbitrarily chosen as the default threshold in the code based on these results, with a molecular weight difference a threshold of 100 atomic mass units set as the default.

![Average error between calibration curves of combination of compounds as a function of their Tanimoto similarity. The molecular weight difference of the compounds is reported in the colormap. The horizontal like indicates 100% error, the vertical line reports the default similarity threshold adopted in the code.\label{fig:tanimoto_error}](paper_files\tanimoto_error.png){ width=60% }

# Acknowledgements

This work was supported by the National Science Foundation CBET under grant numbers 1933071 and 2031710.
The authors thank James Li Adair for the constructive criticism on the manuscript; Alessandro Cascioli for the help on Python package structuring; and Alessandro Cascioli, James Li Adair, and Madeline Karod for being early testers of the code.


# References