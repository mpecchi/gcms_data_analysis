# PyGCMS 

## A Python tool to manage multiple GCMS qualitative tables and automatically split chemicals into functional groups. 
![GA](https://github.com/mpecchi/PyGCMS/blob/main/GA.PNG)
An open-source Python tool that can automatically: 
- handle multiple GCMS semi-quantitative data tables (derivatized or not)
- duild a database of all identified compounds and their relevant properties using PubChemPy
- split each compound into its functional groups using a published fragmentation algorithm
- apply calibrations and/or semi-calibration using Tanimoto and molecular weight similarities
- produce single sample reports, comprehensive multi-sample reports and aggregated reports based on functional group mass fractions in the samples

## Documentation
The full description of the algorithm capabilities are provided at (paper_link).
A scheme of the algorithm is provided here.

![Algorithm](https://github.com/mpecchi/PyGCMS/blob/main/Algorithm.png)
