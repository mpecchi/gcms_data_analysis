import pathlib as plib
import pandas as pd
import numpy as np
import pytest
import rdkit
from gcms_data_analysis.main import Project

@pytest.fixture
def gcms():
    folder_path = plib.Path(plib.Path(__file__).parent.parent, 'tests/data_for_testing/')
    Project.set_folder_path(folder_path)
    return Project()

@pytest.fixture
def checked_files_info():
    files_info = pd.DataFrame(
        index=pd.Index(['A_1', 'A_2', 'Ader_1', 'Ader_2', 'B_1', 'B_2'], name='filename'),
        data={
            'samplename': ['A', 'A', 'Ader', 'Ader', 'B', 'B'],
            'derivatized': [False, False, True, True, False, False],
            'dilution_factor': [25, 25, 125, 125, 1, 1],
            'total_sample_conc_in_vial_mg_L': [560.0000000000001, 560.0000000000001, 112.0, 112.0, 2800.0, 2800.0],
            'sample_yield_on_feedstock_basis_fr': [0.45, 0.46, 0.47, 0.48, 0.49, 0.5],
            'calibration_file': ['calibration', 'calibration', 'deriv_calibration', 'deriv_calibration', 'calibration', 'calibration'],
            }
    )
    return files_info

@pytest.fixture
def checked_created_files_info():
    created_files_info = pd.DataFrame(
        index=pd.Index(['A_1', 'A_2', 'Ader_1', 'Ader_2', 'B_1', 'B_2'], name='filename'),
        data={
            'samplename': ['A', 'A', 'Ader', 'Ader', 'B', 'B'],
            'replicate_number': ['1', '2', '1', '2', '1', '2'],
            'derivatized': [False, False, False, False, False, False],
            'calibration_file': [False, False, False, False, False, False],
            'dilution_factor': [1, 1, 1, 1, 1, 1],
            'total_sample_conc_in_vial_mg_L': [1, 1, 1, 1, 1, 1],
            'sample_yield_on_feedstock_basis_fr': [1, 1, 1, 1, 1, 1],
        }
    )
    return created_files_info


@pytest.fixture
def checked_files():
    files = {
        'A_1': pd.DataFrame(
            index=pd.Index(['unidentified', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'n-hexadecanoic acid', '9,12-octadecadienoic acid (z,z)-', 'oleic acid'], name='A_1'),
            data={
                'iupac_name': ['n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.'],
                'retention_time': [6.025, 36.163, 40.052, 40.492, 43.847, 43.986],
                'area': [23386, 44389, 15068, 1878180, 1456119, 6379752],
                'height': [24797, 15019, 5705, 493759, 339605, 1147599],
                'area_if_undiluted': [584650, 1109725, 376700, 46954500, 36402975, 159493800],
            }),
        'A_2': pd.DataFrame(
            index=pd.Index(['unidentified', 'n-decanoic acid', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'n-hexadecanoic acid', '9,12-octadecadienoic acid (z,z)-', 'oleic acid'], name='A_2'),
            data={
                'iupac_name': ['n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.'],
                'retention_time': [6.025, 26.284, 36.158, 40.041, 40.494, 43.847, 43.988],
                'area': [25493, 10952, 50650, 21294, 1656756, 1371069, 6394708],
                'height': [25716, 4259, 14520, 6739, 461942, 324690, 1138647],
                'area_if_undiluted': [637325, 273800, 1266250, 532350, 41418900, 34276725, 159867700],
            }),
        'Ader_1': pd.DataFrame(
            index=pd.Index(['unidentified', 'myristic acid, tms derivative', 'palmitelaidic acid, tms derivative', 'palmitic acid, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative', '9-octadecenoic acid, (z)-, tms derivative'], name='Ader_1'),
            data={
                'iupac_name': ['n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.'],
                'retention_time': [6.027, 38.123, 41.729, 42.157, 45.253, 45.369],
                'area': [16741, 49508, 27798, 1415205, 519476, 1724814],
                'height': [13451, 18415, 9132, 484890, 180850, 501749],
                'area_if_undiluted': [2092625, 6188500, 3474750, 176900625, 64934500, 215601750],
            }),
        'Ader_2': pd.DataFrame(
            index=pd.Index(['unidentified', 'myristic acid, tms derivative', 'palmitelaidic acid, tms derivative', 'palmitic acid, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative', '9-octadecenoic acid, (z)-, tms derivative'], name='Ader_2'),
            data={
                'iupac_name': ['n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.'],
                'retention_time': [6.027, 38.125, 41.744, 42.161, 45.258, 45.37],
                'area': [14698, 53613, 25213, 1402990, 605137, 1956560],
                'height': [12802, 18373, 8775, 496504, 202599, 594688],
                'area_if_undiluted': [1837250, 6701625, 3151625, 175373750, 75642125, 244570000],
            }),
        'B_1': pd.DataFrame(
            index=pd.Index(['2-butanone', '2-cyclopenten-1-one, 2-methyl-', 'trans-2-pentenoic acid', '2,5-hexanedione', '1-hexene, 4,5-dimethyl-', 'phenol'], name='B_1'),
            data={
                'iupac_name': ['n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.'],
                'retention_time': [8.527, 10.507, 11.071, 11.486, 12.214, 13.687],
                'area': [147566, 69223, 40376, 441077, 19522, 200947],
                'height': [39393, 18515, 12132, 112797, 7194, 64421],
                'area_if_undiluted': [147566, 69223, 40376, 441077, 19522, 200947],
            }),
        'B_2': pd.DataFrame(
            index=pd.Index(['2-butanone', '2-cyclopenten-1-one, 2-methyl-', 'trans-2-pentenoic acid', '2,5-hexanedione', 'phenol'], name='B_2'),
            data={
                'iupac_name': ['n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.'],
                'retention_time': [8.502, 10.474, 11.027, 11.456, 13.661],
                'area': [181021, 64531, 35791, 472362, 228750],
                'height': [44551, 19823, 12737, 120142, 75153],
                'area_if_undiluted': [181021, 64531, 35791, 472362, 228750],
            })
    }
    return files

@pytest.fixture
def checked_is_files_deriv():
    is_files_deriv = {
        'A_1': False, 'A_2': False, 'Ader_1': True,
        'Ader_2': True, 'B_1': False, 'B_2': False
    }
    return is_files_deriv

@pytest.fixture
def checked_load_class_code_fractions():
    class_code_fractions = pd.DataFrame(
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        data={
            'classes': ['ester', 'ester_1', 'ester_2', 'ester_3', 'ester_4', 'ester_5', 'ester_6', 'carboxyl', 'ketone', 'ketone_1', 'ketone_2', 'ketone_3', 'ketone_4', 'ketone_5', 'ketone_6', 'ketone_7', 'ketone_8', 'ketone_9', 'ketone_10', 'ketone_11', 'ketone_12', 'ketone_13', 'ketone_14', 'ketone_15', 'ketone_16', 'ketone_17', 'ketone_18', 'ketone_19', 'ketone_20', 'ketone_21', 'ketone_22', 'ketone_23', 'ketone_24', 'ketone_25', 'ketone_26', 'ketone_27', 'aldehyde', 'ether', 'ether_1', 'ether_2', 'ether_3', 'ether_4', 'ether_5', 'ether_6', 'ether_7', 'ether_8', 'ether_9', 'ether_10', 'ether_11', 'ether_12', 'ether_13', 'ether_14', 'ether_15', 'ether_16', 'ether_17', 'ether_18', 'ether_19', 'ether_20', 'ether_21', 'ether_22', 'ether_23', 'ether_24', 'ether_25', 'ether_26', 'ether_27', 'alcohol', 'C-aliph', 'C-aliph_1', 'C-aliph_2', 'C-aliph_3', 'C-arom', 'C-arom_1', 'C-arom_2', 'N-aliph', 'N-aliph_1', 'N-aliph_3', 'N-arom', 'N-arom_2', 'O-arom', 'O-aliph'],
            'codes': ['[CH0](=O)O[CH3]', '[CH0](=O)O[CH2]', '[CH0](=O)O[CH1]', '[CH0](=O)O[C]', '[CH0](=O)O[cH2]', '[CH0](=O)O[cH1]', '[CH0](=O)O[c]', '[CH0](=O)O', '[CH3]C(=O)[CH3]', '[CH3]C(=O)[CH2]', '[CH3]C(=O)[CH]', '[CH3]C(=O)[C]', '[CH3]C(=O)[cH2]', '[CH3]C(=O)[cH]', '[CH3]C(=O)[c]', '[CH2]C(=O)[CH2]', '[CH2]C(=O)[CH]', '[CH2]C(=O)[C]', '[CH2]C(=O)[cH2]', '[CH2]C(=O)[cH]', '[CH2]C(=O)[c]', '[CH]C(=O)[CH]', '[CH]C(=O)[C]', '[CH]C(=O)[cH2]', '[CH]C(=O)[cH]', '[CH]C(=O)[c]', '[C]C(=O)[C]', '[C]C(=O)[cH2]', '[C]C(=O)[cH]', '[C]C(=O)[c]', '[cH2]C(=O)[cH2]', '[cH2]C(=O)[cH]', '[cH2]C(=O)[c]', '[cH]C(=O)[cH]', '[cH]C(=O)[c]', '[c]C(=O)[c]', '[CH]=O', '[CH3]O[CH3]', '[CH3]O[CH2]', '[CH3]O[CH]', '[CH3]O[C]', '[CH3]O[cH2]', '[CH3]O[cH]', '[CH3]O[c]', '[CH2]O[CH2]', '[CH2]O[CH]', '[CH2]O[C]', '[CH2]O[cH2]', '[CH2]O[cH]', '[CH2]O[c]', '[CH]O[CH]', '[CH]O[C]', '[CH]O[cH2]', '[CH]O[cH]', '[CH]O[c]', '[C]O[C]', '[C]O[cH2]', '[C]O[cH]', '[C]O[c]', '[cH2]O[cH2]', '[cH2]O[cH]', '[cH2]O[c]', '[cH]O[cH]', '[cH]O[c]', '[c]O[c]', '[OH1]', '[CH3]', '[CH2]', '[CH1]', '[C]', '[cH2]', '[cH1]', '[c]', '[NH2]', '[NH1]', '[NH0]', '[nH1]', '[n]', '[o]', '[O]'],
            'mfs': [59.044, 58.035999999999994, 57.028, 56.019999999999996, 58.035999999999994, 57.028, 56.019999999999996, 45.017, 58.080000000000005, 57.072, 56.06400000000001, 55.056000000000004, 57.072, 56.06400000000001, 55.056000000000004, 56.06400000000001, 55.056000000000004, 57.072, 56.06400000000001, 55.056000000000004, 54.048, 54.048, 53.040000000000006, 55.056000000000004, 54.048, 53.040000000000006, 52.032000000000004, 54.048, 53.040000000000006, 52.032000000000004, 56.06400000000001, 55.056000000000004, 54.048, 54.048, 53.040000000000006, 52.032000000000004, 29.017999999999997, 46.069, 45.061, 44.053, 43.045, 45.061, 44.053, 43.045, 44.053, 43.045, 45.061, 44.053, 43.045, 42.037, 42.037, 41.029, 43.045, 42.037, 41.029, 40.021, 42.037, 41.029, 40.021, 44.053, 43.045, 42.037, 42.037, 41.029, 40.021, 17.007, 15.035, 14.027, 13.018999999999998, 12.011, 14.027, 13.018999999999998, 12.011, 16.023, 15.015, 14.007, 15.015, 14.007, 15.999, 15.999],
        }
    )
    return class_code_fractions

@pytest.fixture
def checked_load_calibrations():
    calibrations = {
        'calibration': pd.DataFrame(
            index=pd.Index(['phenol', '2-methylcyclopent-2-en-1-one', '2,4,5-trichlorophenol', 'tetradecanoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'], name='comp_name'),
            data={
                'iupac_name': ['n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.'],
                'MW': [94.11, 96.1271, 197.4, 228.3709, 256.4241, 280.4455, 282.4614],
                'PPM 1': [5.0, 10.0, 5.0, 10.0, 10.0, np.nan, 10.0],
                'PPM 2': [10, 20, 10, 20, 20, 20, 20],
                'PPM 3': [20, 30, 20, 35, 35, 35, 35],
                'PPM 4': [50.0, 50.0, 50.0, 50.0, 50.0, np.nan, 50.0],
                'PPM 5': [np.nan, np.nan, np.nan, 100.0, 100.0, 100.0, 100.0],
                'PPM 6': [np.nan, np.nan, np.nan, 300.0, 300.0, 300.0, 300.0],
                'Area 1': [135884.0, 175083.0, 155710.0, 70675.0, 51545.0, np.nan, 31509.0],
                'Area 2': [304546, 759316, 343277, 203215, 130834, 22338, 133847],
                'Area 3': [678618, 1070146, 805095, 500430, 361070, 63841, 551470],
                'Area 4': [1866918.0, 1928385.0, 2302730.0, 469543.0, 430809.0, np.nan, 494928.0],
                'Area 5': [np.nan, np.nan, np.nan, 2957268.0, 3164919.0, 741540.0, 5345977.0],
                'Area 6': [np.nan, np.nan, np.nan, 11730886.0, 12451729.0, 3975200.0, 19779576.0],
            }
        ),
        'deriv_calibration': pd.DataFrame(
            index=pd.Index(['benzoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '9-octadecenoic acid, (e)-', 'phenol', '4-oxopentanoic acid', 'benzene-1,2-diol'], name='comp_name'),
            data={
                'iupac_name': ['n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.'],
                'MW': [122.1213, 256.4241, 280.4455, 282.4614, 94.1112, 116.1152, 110.1106],
                'PPM 1': [np.nan, 5.0, 5.0, 5.0, np.nan, 5.0, 5.0],
                'PPM 2': [np.nan, 10.0, 10.0, 10.0, np.nan, 10.0, 10.0],
                'PPM 3': [np.nan, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
                'PPM 4': [np.nan, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                'PPM 5': [30, 30, 30, 30, 25, 25, 25],
                'PPM 6': [50, 50, 50, 50, 30, 30, 30],
                'Area 1': [np.nan, 403058.0, 126644.0, 467088.0, np.nan, 48330.0, 184752.0],
                'Area 2': [np.nan, 570479.0, 183307.0, 741971.0, np.nan, 206224.0, 729379.0],
                'Area 3': [np.nan, 694901.0, 241591.0, 953554.0, 17168.0, 620353.0, 1607583.0],
                'Area 4': [np.nan, 936570.0, 350170.0, 1408563.0, 21329.0, 885337.0, 2232039.0],
                'Area 5': [73458, 1474014, 475205, 2476003, 21557, 1096645, 2972508],
                'Area 6': [113812, 2605959, 824267, 4300414, 71706, 1394486, 3629582],
                'CAS': ['65-85-0', '57-10-3', '60-33-3', '112-79-8', '108-95-2', '123-76-2', '120-80-9'],
            }
        )
    }
    return calibrations

@pytest.fixture
def checked_is_calibrations_deriv():
    is_calibrations_deriv = {'calibration': False, 'deriv_calibration': True}
    return is_calibrations_deriv

@pytest.fixture
def checked_list_of_all_compounds():
    list_of_all_compounds = ['tetradecanoic acid', 'oxacycloheptadecan-2-one', 'n-hexadecanoic acid',
        '9,12-octadecadienoic acid (z,z)-', 'oleic acid',
        'n-decanoic acid', '2-butanone', '2-cyclopenten-1-one, 2-methyl-',
        'trans-2-pentenoic acid', '2,5-hexanedione',
        '1-hexene, 4,5-dimethyl-', 'phenol',
        '2-methylcyclopent-2-en-1-one', '2,4,5-trichlorophenol', 'hexadecanoic acid',
        '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'
    ]
    return list_of_all_compounds

@pytest.fixture
def checked_list_of_all_deriv_compounds():
    list_of_all_deriv_compounds = ['myristic acid, tms derivative', 'palmitelaidic acid, tms derivative',
        'palmitic acid, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative',
        '9-octadecenoic acid, (z)-, tms derivative', 'benzoic acid, deriv.',
        'hexadecanoic acid, deriv.', '(9z,12z)-octadeca-9,12-dienoic acid, deriv.',
        '9-octadecenoic acid, (e)-, deriv.', 'phenol, deriv.',
        '4-oxopentanoic acid, deriv.', 'benzene-1,2-diol, deriv.'
    ]
    return list_of_all_deriv_compounds

@pytest.fixture
def checked_compounds_properties():
    compounds_properties = pd.DataFrame(
        index=pd.Index(['tetradecanoic acid', 'oxacycloheptadecan-2-one', 'n-hexadecanoic acid', '9,12-octadecadienoic acid (z,z)-', 'oleic acid', 'n-decanoic acid', '2-butanone', '2-cyclopenten-1-one, 2-methyl-', 'trans-2-pentenoic acid', '2,5-hexanedione', '1-hexene, 4,5-dimethyl-', 'phenol', '2-methylcyclopent-2-en-1-one', '2,4,5-trichlorophenol', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'], name='comp_name'),
        data={
            'iupac_name': ['tetradecanoic acid', 'oxacycloheptadecan-2-one', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid', 'decanoic acid', 'butan-2-one', '2-methylcyclopent-2-en-1-one', '(e)-pent-2-enoic acid', 'hexane-2,5-dione', '4,5-dimethylhex-1-ene', 'phenol', '2-methylcyclopent-2-en-1-one', '2,4,5-trichlorophenol', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
            'molecular_formula': ['C14H28O2', 'C16H30O2', 'C16H32O2', 'C18H32O2', 'C18H34O2', 'C10H20O2', 'C4H8O', 'C6H8O', 'C5H8O2', 'C6H10O2', 'C8H16', 'C6H6O', 'C6H8O', 'C6H3Cl3O', 'C16H32O2', 'C18H32O2', 'C18H34O2'],
            'canonical_smiles': ['CCCCCCCCCCCCCC(=O)O', 'C1CCCCCCCC(=O)OCCCCCCC1', 'CCCCCCCCCCCCCCCC(=O)O', 'CCCCCC=CCC=CCCCCCCCC(=O)O', 'CCCCCCCCC=CCCCCCCCC(=O)O', 'CCCCCCCCCC(=O)O', 'CCC(=O)C', 'CC1=CCCC1=O', 'CCC=CC(=O)O', 'CC(=O)CCC(=O)C', 'CC(C)C(C)CC=C', 'C1=CC=C(C=C1)O', 'CC1=CCCC1=O', 'C1=C(C(=CC(=C1Cl)Cl)Cl)O', 'CCCCCCCCCCCCCCCC(=O)O', 'CCCCCC=CCC=CCCCCCCCC(=O)O', 'CCCCCCCCC=CCCCCCCCC(=O)O'],
            'molecular_weight': [228.37, 254.41, 256.42, 280.4, 282.5, 172.26, 72.11, 96.13, 100.12, 114.14, 112.21, 94.11, 96.13, 197.4, 256.42, 280.4, 282.5],
            'xlogp': [5.3, 6.3, 6.4, 6.8, 6.5, 4.1, 0.3, 0.9, 1.0, -0.3, 3.5, 1.5, 0.9, 3.7, 6.4, 6.8, 6.5],
            'el_C': [14, 16, 16, 18, 18, 10, 4, 6, 5, 6, 8, 6, 6, 6, 16, 18, 18],
            'el_Cl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            'el_H': [28, 30, 32, 32, 34, 20, 8, 8, 8, 10, 16, 6, 8, 3, 32, 32, 34],
            'el_O': [2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2],
            'el_mf_C': [0.7363226343214958, 0.7553791124562713, 0.7494579205990172, 0.7710342368045648, 0.7653026548672566, 0.6972599558806455, 0.6662598807377618, 0.7496723187350464, 0.5998302037554933, 0.6313825127036973, 0.8563229658675697, 0.765763468281798, 0.7496723187350464, 0.3650759878419453, 0.7494579205990172, 0.7710342368045648, 0.7653026548672566],
            'el_mf_Cl': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5387537993920973, 0.0, 0.0, 0.0],
            'el_mf_H': [0.12358891272934273, 0.11886325223065132, 0.12579361984244597, 0.11503566333808846, 0.12131681415929203, 0.11703239289446186, 0.11182914990985994, 0.08388640382814938, 0.08054334798242109, 0.08831259856316805, 0.1437305053025577, 0.06426522154925088, 0.08388640382814938, 0.015319148936170212, 0.12579361984244597, 0.11503566333808846, 0.12131681415929203],
            'el_mf_O': [0.1401147261023777, 0.12577335796548877, 0.12478745807659308, 0.11411554921540658, 0.11326725663716815, 0.18575409265064438, 0.22186936624601306, 0.16643087485696453, 0.31959648421893727, 0.28033993341510427, 0.0, 0.17000318775900541, 0.16643087485696453, 0.08104863221884498, 0.12478745807659308, 0.11411554921540658, 0.11326725663716815],
            'fg_C-aliph': [13, 14, 15, 17, 17, 9, 1, 3, 4, 0, 8, 0, 3, 0, 15, 17, 17],
            'fg_C-arom': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 6, 0, 0, 0],
            'fg_Cl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            'fg_alcohol': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            'fg_carboxyl': [1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            'fg_ester': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'fg_hetero_atoms': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            'fg_ketone': [0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0],
            'fg_mf_C-aliph': [0.8029031834303979, 0.771895758814512, 0.8244793697839481, 0.8396398002853066, 0.8405345132743363, 0.7387147335423198, 0.20850090140063793, 0.4377509622386352, 0.5503395924890131, 0.0, 1.0000534711701274, 0.0, 0.4377509622386352, 0.0, 0.8244793697839481, 0.8396398002853066, 0.8405345132743363],
            'fg_mf_C-arom': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8193178195728402, 0.0, 0.3752887537993921, 0.0, 0.0, 0.0],
            'fg_mf_Cl': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5387537993920973, 0.0, 0.0, 0.0],
            'fg_mf_alcohol': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1807140580172139, 0.0, 0.0861550151975684, 0.0, 0.0, 0.0],
            'fg_mf_carboxyl': [0.19712308972281825, 0.0, 0.1755596287341081, 0.16054564907275323, 0.15935221238938055, 0.2613317078834321, 0.0, 0.0, 0.4496304434678386, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1755596287341081, 0.16054564907275323, 0.15935221238938055],
            'fg_mf_ester': [0.0, 0.22811996383789943, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'fg_mf_hetero_atoms': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5387537993920973, 0.0, 0.0, 0.0],
            'fg_mf_ketone': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7914574954929968, 0.5936960366170811, 0.0, 1.0000350446819695, 0.0, 0.0, 0.5936960366170811, 0.0, 0.0, 0.0, 0.0],
            'fg_mf_total': [0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168, 0.9998867256637168],
        }
    )
    return compounds_properties

@pytest.fixture
def checked_deriv_compounds_properties():
    deriv_compounds_properties = pd.DataFrame(
        index=pd.Index(['myristic acid, tms derivative', 'palmitelaidic acid, tms derivative', 'palmitic acid, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative', '9-octadecenoic acid, (z)-, tms derivative', 'benzoic acid, deriv.', 'hexadecanoic acid, deriv.', '(9z,12z)-octadeca-9,12-dienoic acid, deriv.', '9-octadecenoic acid, (e)-, deriv.', 'phenol, deriv.', '4-oxopentanoic acid, deriv.', 'benzene-1,2-diol, deriv.'], name='comp_name'),
            data={
                'iupac_name': ['tetradecanoic acid', '(e)-hexadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid', 'benzoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(e)-octadec-9-enoic acid', 'phenol', '4-oxopentanoic acid', 'benzene-1,2-diol'],
                'molecular_formula': ['C14H28O2', 'C16H30O2', 'C16H32O2', 'C18H32O2', 'C18H34O2', 'C7H6O2', 'C16H32O2', 'C18H32O2', 'C18H34O2', 'C6H6O', 'C5H8O3', 'C6H6O2'],
                'canonical_smiles': ['CCCCCCCCCCCCCC(=O)O', 'CCCCCCC=CCCCCCCCC(=O)O', 'CCCCCCCCCCCCCCCC(=O)O', 'CCCCCC=CCC=CCCCCCCCC(=O)O', 'CCCCCCCCC=CCCCCCCCC(=O)O', 'C1=CC=C(C=C1)C(=O)O', 'CCCCCCCCCCCCCCCC(=O)O', 'CCCCCC=CCC=CCCCCCCCC(=O)O', 'CCCCCCCCC=CCCCCCCCC(=O)O', 'C1=CC=C(C=C1)O', 'CC(=O)CCC(=O)O', 'C1=CC=C(C(=C1)O)O'],
                'molecular_weight': [228.37, 254.41, 256.42, 280.4, 282.5, 122.12, 256.42, 280.4, 282.5, 94.11, 116.11, 110.11],
                'xlogp': [5.3, 6.4, 6.4, 6.8, 6.5, 1.9, 6.4, 6.8, 6.5, 1.5, -0.5, 0.9],
                'underiv_comp_name': ['myristic acid', 'palmitelaidic acid', 'palmitic acid', '9,12-octadecadienoic acid (z,z)-', '9-octadecenoic acid, (z)-', 'benzoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '9-octadecenoic acid, (e)-', 'phenol', '4-oxopentanoic acid', 'benzene-1,2-diol'],
                'el_C': [14, 16, 16, 18, 18, 7, 16, 18, 18, 6, 5, 6],
                'el_H': [28, 30, 32, 32, 34, 6, 32, 32, 34, 6, 8, 6],
                'el_O': [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 2],
                'el_mf_C': [0.7363226343214958, 0.7553791124562713, 0.7494579205990172, 0.7710342368045648, 0.7653026548672566, 0.6884785456927611, 0.7494579205990172, 0.7710342368045648, 0.7653026548672566, 0.765763468281798, 0.5172250452157436, 0.6544909635818728],
                'el_mf_H': [0.12358891272934273, 0.11886325223065132, 0.12579361984244597, 0.11503566333808846, 0.12131681415929203, 0.04952505732066819, 0.12579361984244597, 0.11503566333808846, 0.12131681415929203, 0.06426522154925088, 0.06945138230987856, 0.054926891290527656],
                'el_mf_O': [0.1401147261023777, 0.12577335796548877, 0.12478745807659308, 0.11411554921540658, 0.11326725663716815, 0.26202096298722566, 0.12478745807659308, 0.11411554921540658, 0.11326725663716815, 0.17000318775900541, 0.413375247610025, 0.29060030878212695],
                'fg_C-aliph': [13, 15, 15, 17, 17, 0, 15, 17, 17, 0, 1, 0],
                'fg_C-arom': [0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6],
                'fg_alcohol': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2],
                'fg_carboxyl': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                'fg_ketone': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                'fg_mf_C-aliph': [0.8029031834303979, 0.8230690617507173, 0.8244793697839481, 0.8396398002853066, 0.8405345132743363, 0.0, 0.8244793697839481, 0.8396398002853066, 0.8405345132743363, 0.0, 0.12080785462061837, 0.0],
                'fg_mf_C-arom': [0.0, 0.0, 0.0, 0.0, 0.0, 0.6313953488372093, 0.0, 0.0, 0.0, 0.8193178195728402, 0.0, 0.6911088911088911],
                'fg_mf_alcohol': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1807140580172139, 0.0, 0.3089092725456362],
                'fg_mf_carboxyl': [0.19712308972281825, 0.17694666090169414, 0.1755596287341081, 0.16054564907275323, 0.15935221238938055, 0.3686292171634458, 0.1755596287341081, 0.16054564907275323, 0.15935221238938055, 0.0, 0.3877099302385669, 0.0],
                'fg_mf_ketone': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.491533890276462, 0.0],
                'fg_mf_total': [1.0000181636545273, 1.0000181636545273, 1.0000181636545273, 1.0000181636545273, 1.0000181636545273, 1.0000181636545273, 1.0000181636545273, 1.0000181636545273, 1.0000181636545273, 1.0000181636545273, 1.0000181636545273, 1.0000181636545273],
        }
    )
    return deriv_compounds_properties

@pytest.fixture
def checked_calibrations_added_iupac_only_iupac_and_mw():
    calibrations = {
    'calibration': pd.DataFrame(
        index=pd.Index(['phenol', '2-methylcyclopent-2-en-1-one', '2,4,5-trichlorophenol', 'tetradecanoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'], name='comp_name'),
        data={
            'iupac_name': ['phenol', '2-methylcyclopent-2-en-1-one', '2,4,5-trichlorophenol', 'tetradecanoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
            'MW': [94.11, 96.1271, 197.4, 228.3709, 256.4241, 280.4455, 282.4614],
        }),
    'deriv_calibration': pd.DataFrame(
        index=pd.Index(['benzoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '9-octadecenoic acid, (e)-', 'phenol', '4-oxopentanoic acid', 'benzene-1,2-diol'], name='comp_name'),
        data={
            'iupac_name': ['benzoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(e)-octadec-9-enoic acid', 'phenol', '4-oxopentanoic acid', 'benzene-1,2-diol'],
            'MW': [122.1213, 256.4241, 280.4455, 282.4614, 94.1112, 116.1152, 110.1106],
        })
    }
    return calibrations

@pytest.fixture
def checked_files_added_iupac_only_iupac_and_time():
    files = {
        'A_1': pd.DataFrame(
            index=pd.Index(['unidentified', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'n-hexadecanoic acid', '9,12-octadecadienoic acid (z,z)-', 'oleic acid'], name='A_1'),
            data={
                'iupac_name': ['unidentified', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.025, 36.163, 40.052, 40.492, 43.847, 43.986],
            }),
        'A_2': pd.DataFrame(
            index=pd.Index(['unidentified', 'n-decanoic acid', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'n-hexadecanoic acid', '9,12-octadecadienoic acid (z,z)-', 'oleic acid'], name='A_2'),
            data={
                'iupac_name': ['unidentified', 'decanoic acid', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.025, 26.284, 36.158, 40.041, 40.494, 43.847, 43.988],
            }),
        'Ader_1': pd.DataFrame(
            index=pd.Index(['unidentified', 'myristic acid, tms derivative', 'palmitelaidic acid, tms derivative', 'palmitic acid, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative', '9-octadecenoic acid, (z)-, tms derivative'], name='Ader_1'),
            data={
                'iupac_name': ['unidentified', 'tetradecanoic acid', '(e)-hexadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.027, 38.123, 41.729, 42.157, 45.253, 45.369],
            }),
        'Ader_2': pd.DataFrame(
            index=pd.Index(['unidentified', 'myristic acid, tms derivative', 'palmitelaidic acid, tms derivative', 'palmitic acid, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative', '9-octadecenoic acid, (z)-, tms derivative'], name='Ader_2'),
            data={
                'iupac_name': ['unidentified', 'tetradecanoic acid', '(e)-hexadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.027, 38.125, 41.744, 42.161, 45.258, 45.37],
            }),
        'B_1': pd.DataFrame(
            index=pd.Index(['2-butanone', '2-cyclopenten-1-one, 2-methyl-', 'trans-2-pentenoic acid', '2,5-hexanedione', '1-hexene, 4,5-dimethyl-', 'phenol'], name='B_1'),
            data={
                'iupac_name': ['butan-2-one', '2-methylcyclopent-2-en-1-one', '(e)-pent-2-enoic acid', 'hexane-2,5-dione', '4,5-dimethylhex-1-ene', 'phenol'],
                'retention_time': [8.527, 10.507, 11.071, 11.486, 12.214, 13.687],
            }),
        'B_2': pd.DataFrame(
            index=pd.Index(['2-butanone', '2-cyclopenten-1-one, 2-methyl-', 'trans-2-pentenoic acid', '2,5-hexanedione', 'phenol'], name='B_2'),
            data={
                'iupac_name': ['butan-2-one', '2-methylcyclopent-2-en-1-one', '(e)-pent-2-enoic acid', 'hexane-2,5-dione', 'phenol'],
                'retention_time': [8.502, 10.474, 11.027, 11.456, 13.661],
            })
    }
    return files

@pytest.fixture
def checked_files_applied_calibration():
    files = {
        'A_1': pd.DataFrame(
            index=pd.Index(['unidentified', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'n-hexadecanoic acid', '9,12-octadecadienoic acid (z,z)-', 'oleic acid'], name='A_1'),
            data={
                'iupac_name': ['unidentified', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.025, 36.163, 40.052, 40.492, 43.847, 43.986],
                'area': [23386, 44389, 15068, 1878180, 1456119, 6379752],
                'height': [24797, 15019, 5705, 493759, 339605, 1147599],
                'area_if_undiluted': [584650, 1109725, 376700, 46954500, 36402975, 159493800],
                'conc_vial_mg_L': [np.nan, 23.581503644987627, np.nan, 66.05436178187291, 131.18800047103497, 113.61850020825628],
                'conc_vial_if_undiluted_mg_L': [np.nan, 589.5375911246907, np.nan, 1651.3590445468228, 3279.7000117758744, 2840.462505206407],
                'fraction_of_sample_fr': [np.nan, 0.042109827937477896, np.nan, 0.11795421746763018, 0.23426428655541953, 0.20289017894331474],
                'fraction_of_feedstock_fr': [np.nan, 0.018949422571865052, np.nan, 0.053079397860433586, 0.10541892894993879, 0.09130058052449164],
                'compound_used_for_calibration': ['n.a.', 'self', 'n.a.', 'self', 'self', 'self'],
            }),
        'A_2': pd.DataFrame(
            index=pd.Index(['unidentified', 'n-decanoic acid', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'n-hexadecanoic acid', '9,12-octadecadienoic acid (z,z)-', 'oleic acid'], name='A_2'),
            data={
                'iupac_name': ['unidentified', 'decanoic acid', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.025, 26.284, 36.158, 40.041, 40.494, 43.847, 43.988],
                'area': [25493, 10952, 50650, 21294, 1656756, 1371069, 6394708],
                'height': [25716, 4259, 14520, 6739, 461942, 324690, 1138647],
                'area_if_undiluted': [637325, 273800, 1266250, 532350, 41418900, 34276725, 159867700],
                'conc_vial_mg_L': [np.nan, 22.78427785050836, 23.730782309318595, np.nan, 61.11672684588226, 125.38077898437679, 113.82730072166243],
                'conc_vial_if_undiluted_mg_L': [np.nan, 569.606946262709, 593.2695577329649, np.nan, 1527.9181711470565, 3134.51947460942, 2845.682518041561],
                'fraction_of_sample_fr': [np.nan, 0.04068621044733635, 0.04237639698092605, np.nan, 0.10913701222478973, 0.2238942481863871, 0.20326303700296858],
                'fraction_of_feedstock_fr': [np.nan, 0.018715656805774722, 0.019493142611225985, np.nan, 0.05020302562340328, 0.10299135416573807, 0.09350099702136555],
                'compound_used_for_calibration': ['n.a.', 'tetradecanoic acid (sim=1.0; dwt=56)', 'self', 'n.a.', 'self', 'self', 'self'],
            }),
        'Ader_1': pd.DataFrame(
            index=pd.Index(['unidentified', 'myristic acid, tms derivative', 'palmitelaidic acid, tms derivative', 'palmitic acid, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative', '9-octadecenoic acid, (z)-, tms derivative'], name='Ader_1'),
            data={
                'iupac_name': ['unidentified', 'tetradecanoic acid', '(e)-hexadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.027, 38.123, 41.729, 42.157, 45.253, 45.369],
                'area': [16741, 49508, 27798, 1415205, 519476, 1724814],
                'height': [13451, 18415, 9132, 484890, 180850, 501749],
                'area_if_undiluted': [2092625, 6188500, 3474750, 176900625, 64934500, 215601750],
                'conc_vial_mg_L': [np.nan, 0.600983241036704, 2.5980281295127825, 27.623189632994073, 31.36776718294773, 21.669084708496513],
                'conc_vial_if_undiluted_mg_L': [np.nan, 75.12290512958799, 324.7535161890978, 3452.898704124259, 3920.970897868466, 2708.635588562064],
                'fraction_of_sample_fr': [np.nan, 0.005365921794970571, 0.023196679727792702, 0.24663562172316136, 0.2800693498477476, 0.193473970611576],
                'fraction_of_feedstock_fr': [np.nan, 0.0025219832436361683, 0.01090243947206257, 0.11591874220988584, 0.13163259442844139, 0.09093276618744071],
                'compound_used_for_calibration': ['n.a.', 'hexadecanoic acid (sim=1.0; dwt=28)', '(e)-octadec-9-enoic acid (sim=1.0; dwt=28)', 'self', 'self', '(e)-octadec-9-enoic acid (sim=1.0; dwt=0)'],
            }),
        'Ader_2': pd.DataFrame(
            index=pd.Index(['unidentified', 'myristic acid, tms derivative', 'palmitelaidic acid, tms derivative', 'palmitic acid, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative', '9-octadecenoic acid, (z)-, tms derivative'], name='Ader_2'),
            data={
                'iupac_name': ['unidentified', 'tetradecanoic acid', '(e)-hexadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.027, 38.125, 41.744, 42.161, 45.258, 45.37],
                'area': [14698, 53613, 25213, 1402990, 605137, 1956560],
                'height': [12802, 18373, 8775, 496504, 202599, 594688],
                'area_if_undiluted': [1837250, 6701625, 3151625, 175373750, 75642125, 244570000],
                'conc_vial_mg_L': [np.nan, 0.6822063507301317, 2.5689779135709925, 27.38149894239597, 36.81298755438084, 24.27344499617392],
                'conc_vial_if_undiluted_mg_L': [np.nan, 85.27579384126646, 321.12223919637404, 3422.6873677994963, 4601.623444297605, 3034.1806245217404],
                'fraction_of_sample_fr': [np.nan, 0.006091128131519033, 0.022937302799741006, 0.24447766912853547, 0.3286873888784004, 0.21672718746583858],
                'fraction_of_feedstock_fr': [np.nan, 0.0029237415031291357, 0.011009905343875682, 0.11734928118169702, 0.1577699466616322, 0.10402904998360252],
                'compound_used_for_calibration': ['n.a.', 'hexadecanoic acid (sim=1.0; dwt=28)', '(e)-octadec-9-enoic acid (sim=1.0; dwt=28)', 'self', 'self', '(e)-octadec-9-enoic acid (sim=1.0; dwt=0)'],
            }),
        'B_1': pd.DataFrame(
            index=pd.Index(['2-butanone', '2-cyclopenten-1-one, 2-methyl-', 'trans-2-pentenoic acid', '2,5-hexanedione', '1-hexene, 4,5-dimethyl-', 'phenol'], name='B_1'),
            data={
                'iupac_name': ['butan-2-one', '2-methylcyclopent-2-en-1-one', '(e)-pent-2-enoic acid', 'hexane-2,5-dione', '4,5-dimethylhex-1-ene', 'phenol'],
                'retention_time': [8.527, 10.507, 11.071, 11.486, 12.214, 13.687],
                'area': [147566, 69223, 40376, 441077, 19522, 200947],
                'height': [39393, 18515, 12132, 112797, 7194, 64421],
                'area_if_undiluted': [147566, 69223, 40376, 441077, 19522, 200947],
                'conc_vial_mg_L': [np.nan, 6.243800844792131, np.nan, np.nan, np.nan, 7.167230535550548],
                'conc_vial_if_undiluted_mg_L': [np.nan, 6.243800844792131, np.nan, np.nan, np.nan, 7.167230535550548],
                'fraction_of_sample_fr': [np.nan, 0.0022299288731400468, np.nan, np.nan, np.nan, 0.0025597251912680527],
                'fraction_of_feedstock_fr': [np.nan, 0.001092665147838623, np.nan, np.nan, np.nan, 0.0012542653437213457],
                'compound_used_for_calibration': ['n.a.', 'self', 'n.a.', 'n.a.', 'n.a.', 'self'],
            }),
        'B_2': pd.DataFrame(
            index=pd.Index(['2-butanone', '2-cyclopenten-1-one, 2-methyl-', 'trans-2-pentenoic acid', '2,5-hexanedione', 'phenol'], name='B_2'),
            data={
                'iupac_name': ['butan-2-one', '2-methylcyclopent-2-en-1-one', '(e)-pent-2-enoic acid', 'hexane-2,5-dione', 'phenol'],
                'retention_time': [8.502, 10.474, 11.027, 11.456, 13.661],
                'area': [181021, 64531, 35791, 472362, 228750],
                'height': [44551, 19823, 12737, 120142, 75153],
                'area_if_undiluted': [181021, 64531, 35791, 472362, 228750],
                'conc_vial_mg_L': [np.nan, 6.134683722446865, np.nan, np.nan, 7.884941445329839],
                'conc_vial_if_undiluted_mg_L': [np.nan, 6.134683722446865, np.nan, np.nan, 7.884941445329839],
                'fraction_of_sample_fr': [np.nan, 0.0021909584723024517, np.nan, np.nan, 0.002816050516189228],
                'fraction_of_feedstock_fr': [np.nan, 0.0010954792361512259, np.nan, np.nan, 0.001408025258094614],
                'compound_used_for_calibration': ['n.a.', 'self', 'n.a.', 'n.a.', 'self'],
            })
        }
    return files

@pytest.fixture
def checked_files_info_added_stats():
    files_info = pd.DataFrame(
        index=pd.Index(['A_1', 'A_2', 'Ader_1', 'Ader_2', 'B_1', 'B_2'], name='filename'),
        data={
            'samplename': ['A', 'A', 'Ader', 'Ader', 'B', 'B'],
            'derivatized': [False, False, True, True, False, False],
            'dilution_factor': [25, 25, 125, 125, 1, 1],
            'total_sample_conc_in_vial_mg_L': [560.0000000000001, 560.0000000000001, 112.0, 112.0, 2800.0, 2800.0],
            'sample_yield_on_feedstock_basis_fr': [0.45, 0.46, 0.47, 0.48, 0.49, 0.5],
            'calibration_file': ['calibration', 'calibration', 'deriv_calibration', 'deriv_calibration', 'calibration', 'calibration'],
            'max_height': [1147599.0, 1138647.0, 501749.0, 594688.0, 112797.0, 120142.0],
            'total_height': [2026484.0, 1976513.0, 1208487.0, 1333741.0, 254452.0, 272406.0],
            'max_area': [6379752.0, 6394708.0, 1724814.0, 1956560.0, 441077.0, 472362.0],
            'total_area': [9796894.0, 9530922.0, 3753542.0, 4058211.0, 918711.0, 982455.0],
            'max_area_if_undiluted': [159493800.0, 159867700.0, 215601750.0, 244570000.0, 441077.0, 472362.0],
            'total_area_if_undiluted': [244922350.0, 238273050.0, 469192750.0, 507276375.0, 918711.0, 982455.0],
            'max_conc_vial_mg_L': [131.18800047103497, 125.38077898437679, 31.36776718294773, 36.81298755438084, 7.167230535550548, 7.884941445329839],
            'total_conc_vial_mg_L': [334.4423661061518, 346.8398667117484, 83.8590528949878, 91.71911575725186, 13.411031380342678, 14.019625167776704],
            'max_conc_vial_if_undiluted_mg_L': [3279.7000117758744, 3134.51947460942, 3920.970897868466, 4601.623444297605, 7.167230535550548, 7.884941445329839],
            'total_conc_vial_if_undiluted_mg_L': [8361.059152653796, 8670.996667793712, 10482.381611873476, 11464.889469656482, 13.411031380342678, 14.019625167776704],
            'max_fraction_of_sample_fr': [0.23426428655541953, 0.2238942481863871, 0.2800693498477476, 0.3286873888784004, 0.0025597251912680527, 0.002816050516189228],
            'total_fraction_of_sample_fr': [0.5972185109038424, 0.6193569048424078, 0.7487415437052483, 0.8189206764040344, 0.0047896540644080995, 0.005007008988491679],
            'max_fraction_of_feedstock_fr': [0.10541892894993879, 0.10299135416573807, 0.13163259442844139, 0.1577699466616322, 0.0012542653437213457, 0.001408025258094614],
            'total_fraction_of_feedstock_fr': [0.26874832990672903, 0.28490417622750763, 0.3519085255414667, 0.39308192467393654, 0.0023469304915599686, 0.0025035044942458397],
            'compound_with_max_area': ['oleic acid', 'oleic acid', '9-octadecenoic acid, (z)-, tms derivative', '9-octadecenoic acid, (z)-, tms derivative', '2,5-hexanedione', '2,5-hexanedione'],
            'compound_with_max_conc': ['9,12-octadecadienoic acid (z,z)-', '9,12-octadecadienoic acid (z,z)-', '9,12-octadecadienoic acid (z,z)-, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative', 'phenol', 'phenol'],
        }
    )
    return files_info

@pytest.fixture
def checked_samples_info():
    samples_info = pd.DataFrame(
        index=pd.Index(['A', 'Ader', 'B'], name='samplename'),
        data={
            'filename': [['A_1', 'A_2'], ['Ader_1', 'Ader_2'], ['B_1', 'B_2']],
            'derivatized': [[False, False], [True, True], [False, False]],
            'dilution_factor': [[25, 25], [125, 125], [1, 1]],
            'total_sample_conc_in_vial_mg_L': [[560.0000000000001, 560.0000000000001], [112.0, 112.0], [2800.0, 2800.0]],
            'sample_yield_on_feedstock_basis_fr': [[0.45, 0.46], [0.47, 0.48], [0.49, 0.5]],
            'calibration_file': [['calibration', 'calibration'], ['deriv_calibration', 'deriv_calibration'], ['calibration', 'calibration']],
        }
    )
    return samples_info

@pytest.fixture
def checked_samples_info_no_calibrations():
    samples_info = pd.DataFrame(
        index=pd.Index(['A', 'Ader', 'B'], name='samplename'),
        data={
            'filename': [['A_1', 'A_2'], ['Ader_1', 'Ader_2'], ['B_1', 'B_2']],
            'replicate_number': [['1', '2'], ['1', '2'], ['1', '2']],
            'derivatized': [[False, False], [False, False], [False, False]],
            'calibration_file': [[False, False], [False, False], [False, False]],
            'dilution_factor': [[1, 1], [1, 1], [1, 1]],
            'total_sample_conc_in_vial_mg_L': [[1, 1], [1, 1], [1, 1]],
            'sample_yield_on_feedstock_basis_fr': [[1, 1], [1, 1], [1, 1]],
            'compound_with_max_area': ['oleic acid', '9-octadecenoic acid, (z)-, tms derivative', '2,5-hexanedione'],
            'max_height': [1143123.0, 548218.5, 116469.5],
            'max_area': [6387230.0, 1840687.0, 456719.5],
            'max_area_if_undiluted': [6387230.0, 1840687.0, 456719.5],
            'total_height': [2003628.0, 1271114.0, 263429.0],
            'total_area': [9669384.0, 3905876.5, 950583.0],
            'total_area_if_undiluted': [9669384.0, 3905876.5, 950583.0],
        }
    )
    return samples_info

@pytest.fixture
def checked_samples_info_applied_calibration():
    samples_info = pd.DataFrame(
    index=pd.Index(['A', 'Ader', 'B'], name='samplename'),
    data={
        'filename': [['A_1', 'A_2'], ['Ader_1', 'Ader_2'], ['B_1', 'B_2']],
        'derivatized': [[False, False], [True, True], [False, False]],
        'dilution_factor': [[25, 25], [125, 125], [1, 1]],
        'total_sample_conc_in_vial_mg_L': [[560.0000000000001, 560.0000000000001], [112.0, 112.0], [2800.0, 2800.0]],
        'sample_yield_on_feedstock_basis_fr': [[0.45, 0.46], [0.47, 0.48], [0.49, 0.5]],
        'calibration_file': [['calibration', 'calibration'], ['deriv_calibration', 'deriv_calibration'], ['calibration', 'calibration']],
        'compound_with_max_area': ['oleic acid', '9-octadecenoic acid, (z)-, tms derivative', '2,5-hexanedione'],
        'compound_with_max_conc': ['9,12-octadecadienoic acid (z,z)-', '9,12-octadecadienoic acid (z,z)-, tms derivative', 'phenol'],
        'max_height': [1143123.0, 548218.5, 116469.5],
        'max_area': [6387230.0, 1840687.0, 456719.5],
        'max_area_if_undiluted': [159680750.0, 230085875.0, 456719.5],
        'max_conc_vial_mg_L': [128.28438972770587, 34.090377368664285, 7.526085990440194],
        'max_conc_vial_if_undiluted_mg_L': [3207.1097431926473, 4261.297171083035, 7.526085990440194],
        'max_fraction_of_sample_fr': [0.2290792673709033, 0.304378369363074, 0.0026878878537286406],
        'max_fraction_of_feedstock_fr': [0.10420514155783843, 0.1447012705450368, 0.00133114530090798],
        'total_height': [2003628.0, 1271114.0, 263429.0],
        'total_area': [9669384.0, 3905876.5, 950583.0],
        'total_area_if_undiluted': [241734600.0, 488234562.5, 950583.0],
        'total_conc_vial_mg_L': [352.0332553342043, 87.78908432611982, 13.715328274059692],
        'total_conc_vial_if_undiluted_mg_L': [8800.831383355107, 10973.635540764979, 13.715328274059692],
        'total_fraction_of_sample_fr': [0.6286308130967933, 0.7838311100546413, 0.0048983315264498895],
        'total_fraction_of_feedstock_fr': [0.28618408147000574, 0.3724952251077016, 0.0024252174929029046],
    }
)
    return samples_info


@pytest.fixture
def checked_samples():
    samples = {
        'A': pd.DataFrame(
            index=pd.Index(['unidentified', 'n-decanoic acid', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'n-hexadecanoic acid', '9,12-octadecadienoic acid (z,z)-', 'oleic acid'], name='A'),
            data={
                'iupac_name': ['unidentified', 'decanoic acid', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.025, 26.284, 36.1605, 40.046499999999995, 40.492999999999995, 43.847, 43.986999999999995],
                'area': [24439.5, 10952.0, 47519.5, 18181.0, 1767468.0, 1413594.0, 6387230.0],
                'height': [25256.5, 4259.0, 14769.5, 6222.0, 477850.5, 332147.5, 1143123.0],
                'area_if_undiluted': [610987.5, 273800.0, 1187987.5, 454525.0, 44186700.0, 35339850.0, 159680750.0],
                'conc_vial_mg_L': [0.0, 22.78427785050836, 23.65614297715311, 0.0, 63.58554431387759, 128.28438972770587, 113.72290046495935],
                'conc_vial_if_undiluted_mg_L': [0.0, 569.606946262709, 591.4035744288278, 0.0, 1589.6386078469395, 3207.1097431926473, 2843.0725116239837],
                'fraction_of_sample_fr': [0.0, 0.04068621044733635, 0.042243112459201974, 0.0, 0.11354561484620995, 0.2290792673709033, 0.20307660797314164],
                'fraction_of_feedstock_fr': [0.0, 0.018715656805774722, 0.01922128259154552, 0.0, 0.051641211741918436, 0.10420514155783843, 0.0924007887729286],
                'compound_used_for_calibration': ['n.a.', 'tetradecanoic acid (sim=1.0; dwt=56)', 'self', 'n.a.', 'self', 'self', 'self'],
            }
        ),
        'Ader': pd.DataFrame(
            index=pd.Index(['unidentified', 'myristic acid, tms derivative', 'palmitelaidic acid, tms derivative', 'palmitic acid, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative', '9-octadecenoic acid, (z)-, tms derivative'], name='Ader'),
            data={
                'iupac_name': ['unidentified', 'tetradecanoic acid', '(e)-hexadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.027, 38.123999999999995, 41.7365, 42.159, 45.2555, 45.3695],
                'area': [15719.5, 51560.5, 26505.5, 1409097.5, 562306.5, 1840687.0],
                'height': [13126.5, 18394.0, 8953.5, 490697.0, 191724.5, 548218.5],
                'area_if_undiluted': [1964937.5, 6445062.5, 3313187.5, 176137187.5, 70288312.5, 230085875.0],
                'conc_vial_mg_L': [0.0, 0.6415947958834178, 2.5835030215418877, 27.502344287695024, 34.090377368664285, 22.971264852335217],
                'conc_vial_if_undiluted_mg_L': [0.0, 80.19934948542723, 322.9378776927359, 3437.7930359618776, 4261.297171083035, 2871.4081065419023],
                'fraction_of_sample_fr': [0.0, 0.005728524963244802, 0.023066991263766854, 0.24555664542584843, 0.304378369363074, 0.2051005790387073],
                'fraction_of_feedstock_fr': [0.0, 0.002722862373382652, 0.010956172407969126, 0.11663401169579143, 0.1447012705450368, 0.09748090808552162],
                'compound_used_for_calibration': ['n.a.', 'hexadecanoic acid (sim=1.0; dwt=28)', '(e)-octadec-9-enoic acid (sim=1.0; dwt=28)', 'self', 'self', '(e)-octadec-9-enoic acid (sim=1.0; dwt=0)'],
            }
        )
,
        'B': pd.DataFrame(
            index=pd.Index(['1-hexene, 4,5-dimethyl-', '2-butanone', '2-cyclopenten-1-one, 2-methyl-', 'trans-2-pentenoic acid', '2,5-hexanedione', 'phenol'], name='B'),
            data={
                'iupac_name': ['4,5-dimethylhex-1-ene', 'butan-2-one', '2-methylcyclopent-2-en-1-one', '(e)-pent-2-enoic acid', 'hexane-2,5-dione', 'phenol'],
                'retention_time': [6.107, 8.5145, 10.4905, 11.049, 11.471, 13.674],
                'area': [9761.0, 164293.5, 66877.0, 38083.5, 456719.5, 214848.5],
                'height': [3597.0, 41972.0, 19169.0, 12434.5, 116469.5, 69787.0],
                'area_if_undiluted': [9761.0, 164293.5, 66877.0, 38083.5, 456719.5, 214848.5],
                'conc_vial_mg_L': [0.0, 0.0, 6.189242283619498, 0.0, 0.0, 7.526085990440194],
                'conc_vial_if_undiluted_mg_L': [0.0, 0.0, 6.189242283619498, 0.0, 0.0, 7.526085990440194],
                'fraction_of_sample_fr': [0.0, 0.0, 0.0022104436727212492, 0.0, 0.0, 0.0026878878537286406],
                'fraction_of_feedstock_fr': [0.0, 0.0, 0.0010940721919949245, 0.0, 0.0, 0.00133114530090798],
                'compound_used_for_calibration': ['n.a.', 'n.a.', 'self', 'n.a.', 'n.a.', 'self'],
            }
        )
    }
    return samples

@pytest.fixture
def checked_samples_applied_calibration():
    samples = {
        'A': pd.DataFrame(
            index=pd.Index(['unidentified', 'n-decanoic acid', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'n-hexadecanoic acid', '9,12-octadecadienoic acid (z,z)-', 'oleic acid'], name='A'),
            data={
                'iupac_name': ['unidentified', 'decanoic acid', 'tetradecanoic acid', 'oxacycloheptadecan-2-one', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.025, 26.284, 36.1605, 40.046499999999995, 40.492999999999995, 43.847, 43.986999999999995],
                'area': [24439.5, 10952.0, 47519.5, 18181.0, 1767468.0, 1413594.0, 6387230.0],
                'height': [25256.5, 4259.0, 14769.5, 6222.0, 477850.5, 332147.5, 1143123.0],
                'area_if_undiluted': [610987.5, 273800.0, 1187987.5, 454525.0, 44186700.0, 35339850.0, 159680750.0],
                'conc_vial_mg_L': [0.0, 22.78427785050836, 23.65614297715311, 0.0, 63.58554431387759, 128.28438972770587, 113.72290046495935],
                'conc_vial_if_undiluted_mg_L': [0.0, 569.606946262709, 591.4035744288278, 0.0, 1589.6386078469395, 3207.1097431926473, 2843.0725116239837],
                'fraction_of_sample_fr': [0.0, 0.04068621044733635, 0.042243112459201974, 0.0, 0.11354561484620995, 0.2290792673709033, 0.20307660797314164],
                'fraction_of_feedstock_fr': [0.0, 0.018715656805774722, 0.01922128259154552, 0.0, 0.051641211741918436, 0.10420514155783843, 0.0924007887729286],
                'compound_used_for_calibration': ['n.a.', 'tetradecanoic acid (sim=1.0; dwt=56)', 'self', 'n.a.', 'self', 'self', 'self'],
            }
        ),
        'Ader': pd.DataFrame(
            index=pd.Index(['unidentified', 'myristic acid, tms derivative', 'palmitelaidic acid, tms derivative', 'palmitic acid, tms derivative', '9,12-octadecadienoic acid (z,z)-, tms derivative', '9-octadecenoic acid, (z)-, tms derivative'], name='Ader'),
            data={
                'iupac_name': ['unidentified', 'tetradecanoic acid', '(e)-hexadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid'],
                'retention_time': [6.027, 38.123999999999995, 41.7365, 42.159, 45.2555, 45.3695],
                'area': [15719.5, 51560.5, 26505.5, 1409097.5, 562306.5, 1840687.0],
                'height': [13126.5, 18394.0, 8953.5, 490697.0, 191724.5, 548218.5],
                'area_if_undiluted': [1964937.5, 6445062.5, 3313187.5, 176137187.5, 70288312.5, 230085875.0],
                'conc_vial_mg_L': [0.0, 0.6415947958834178, 2.5835030215418877, 27.502344287695024, 34.090377368664285, 22.971264852335217],
                'conc_vial_if_undiluted_mg_L': [0.0, 80.19934948542723, 322.9378776927359, 3437.7930359618776, 4261.297171083035, 2871.4081065419023],
                'fraction_of_sample_fr': [0.0, 0.005728524963244802, 0.023066991263766854, 0.24555664542584843, 0.304378369363074, 0.2051005790387073],
                'fraction_of_feedstock_fr': [0.0, 0.002722862373382652, 0.010956172407969126, 0.11663401169579143, 0.1447012705450368, 0.09748090808552162],
                'compound_used_for_calibration': ['n.a.', 'hexadecanoic acid (sim=1.0; dwt=28)', '(e)-octadec-9-enoic acid (sim=1.0; dwt=28)', 'self', 'self', '(e)-octadec-9-enoic acid (sim=1.0; dwt=0)'],
            }
        )
,
        'B': pd.DataFrame(
            index=pd.Index(['1-hexene, 4,5-dimethyl-', '2-butanone', '2-cyclopenten-1-one, 2-methyl-', 'trans-2-pentenoic acid', '2,5-hexanedione', 'phenol'], name='B'),
            data={
                'iupac_name': ['4,5-dimethylhex-1-ene', 'butan-2-one', '2-methylcyclopent-2-en-1-one', '(e)-pent-2-enoic acid', 'hexane-2,5-dione', 'phenol'],
                'retention_time': [6.107, 8.5145, 10.4905, 11.049, 11.471, 13.674],
                'area': [9761.0, 164293.5, 66877.0, 38083.5, 456719.5, 214848.5],
                'height': [3597.0, 41972.0, 19169.0, 12434.5, 116469.5, 69787.0],
                'area_if_undiluted': [9761.0, 164293.5, 66877.0, 38083.5, 456719.5, 214848.5],
                'conc_vial_mg_L': [0.0, 0.0, 6.189242283619498, 0.0, 0.0, 7.526085990440194],
                'conc_vial_if_undiluted_mg_L': [0.0, 0.0, 6.189242283619498, 0.0, 0.0, 7.526085990440194],
                'fraction_of_sample_fr': [0.0, 0.0, 0.0022104436727212492, 0.0, 0.0, 0.0026878878537286406],
                'fraction_of_feedstock_fr': [0.0, 0.0, 0.0010940721919949245, 0.0, 0.0, 0.00133114530090798],
                'compound_used_for_calibration': ['n.a.', 'n.a.', 'self', 'n.a.', 'n.a.', 'self'],
            }
        )
    }
    return samples

@pytest.fixture
def checked_files_param_reports():
    reports = {
        'height': pd.DataFrame(
            index=pd.Index(['(z)-octadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', 'hexane-2,5-dione', 'phenol', 'butan-2-one', '2-methylcyclopent-2-en-1-one', 'tetradecanoic acid', '(e)-pent-2-enoic acid', '(e)-hexadec-9-enoic acid', '4,5-dimethylhex-1-ene', 'oxacycloheptadecan-2-one', 'decanoic acid'], name='height'),
            data={
            'A_1': [1147599.0, 493759.0, 339605.0, 0.0, 0.0, 0.0, 0.0, 15019.0, 0.0, 0.0, 0.0, 5705.0, 0.0],
            'A_2': [1138647.0, 461942.0, 324690.0, 0.0, 0.0, 0.0, 0.0, 14520.0, 0.0, 0.0, 0.0, 6739.0, 4259.0],
            'Ader_1': [501749.0, 484890.0, 180850.0, 0.0, 0.0, 0.0, 0.0, 18415.0, 0.0, 9132.0, 0.0, 0.0, 0.0],
            'Ader_2': [594688.0, 496504.0, 202599.0, 0.0, 0.0, 0.0, 0.0, 18373.0, 0.0, 8775.0, 0.0, 0.0, 0.0],
            'B_1': [0.0, 0.0, 0.0, 112797.0, 64421.0, 39393.0, 18515.0, 0.0, 12132.0, 0.0, 7194.0, 0.0, 0.0],
            'B_2': [0.0, 0.0, 0.0, 120142.0, 75153.0, 44551.0, 19823.0, 0.0, 12737.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
            'area': pd.DataFrame(
            index=pd.Index(['(z)-octadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', 'hexane-2,5-dione', 'phenol', 'butan-2-one', '2-methylcyclopent-2-en-1-one', 'tetradecanoic acid', '(e)-pent-2-enoic acid', '(e)-hexadec-9-enoic acid', 'oxacycloheptadecan-2-one', '4,5-dimethylhex-1-ene', 'decanoic acid'], name='area'),
            data={
                'A_1': [6379752.0, 1878180.0, 1456119.0, 0.0, 0.0, 0.0, 0.0, 44389.0, 0.0, 0.0, 15068.0, 0.0, 0.0],
                'A_2': [6394708.0, 1656756.0, 1371069.0, 0.0, 0.0, 0.0, 0.0, 50650.0, 0.0, 0.0, 21294.0, 0.0, 10952.0],
                'Ader_1': [1724814.0, 1415205.0, 519476.0, 0.0, 0.0, 0.0, 0.0, 49508.0, 0.0, 27798.0, 0.0, 0.0, 0.0],
                'Ader_2': [1956560.0, 1402990.0, 605137.0, 0.0, 0.0, 0.0, 0.0, 53613.0, 0.0, 25213.0, 0.0, 0.0, 0.0],
                'B_1': [0.0, 0.0, 0.0, 441077.0, 200947.0, 147566.0, 69223.0, 0.0, 40376.0, 0.0, 0.0, 19522.0, 0.0],
                'B_2': [0.0, 0.0, 0.0, 472362.0, 228750.0, 181021.0, 64531.0, 0.0, 35791.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
        'area_if_undiluted': pd.DataFrame(
            index=pd.Index(['(z)-octadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', 'tetradecanoic acid', '(e)-hexadec-9-enoic acid', 'oxacycloheptadecan-2-one', 'hexane-2,5-dione', 'decanoic acid', 'phenol', 'butan-2-one', '2-methylcyclopent-2-en-1-one', '(e)-pent-2-enoic acid', '4,5-dimethylhex-1-ene'], name='area_if_undiluted'),
            data={
                'A_1': [159493800.0, 46954500.0, 36402975.0, 1109725.0, 0.0, 376700.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'A_2': [159867700.0, 41418900.0, 34276725.0, 1266250.0, 0.0, 532350.0, 0.0, 273800.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'Ader_1': [215601750.0, 176900625.0, 64934500.0, 6188500.0, 3474750.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'Ader_2': [244570000.0, 175373750.0, 75642125.0, 6701625.0, 3151625.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'B_1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 441077.0, 0.0, 200947.0, 147566.0, 69223.0, 40376.0, 19522.0],
                'B_2': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 472362.0, 0.0, 228750.0, 181021.0, 64531.0, 35791.0, 0.0],
            }
        ),
        'conc_vial_mg_L': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid', 'hexadecanoic acid', 'tetradecanoic acid', 'decanoic acid', 'phenol', '2-methylcyclopent-2-en-1-one', '(e)-hexadec-9-enoic acid'], name='conc_vial_mg_L'),
            data={
                'A_1': [131.18800047103497, 113.61850020825628, 66.05436178187291, 23.581503644987627, 0.0, 0.0, 0.0, 0.0],
                'A_2': [125.38077898437679, 113.82730072166243, 61.11672684588226, 23.730782309318595, 22.78427785050836, 0.0, 0.0, 0.0],
                'Ader_1': [31.36776718294773, 21.669084708496513, 27.623189632994073, 0.600983241036704, 0.0, 0.0, 0.0, 2.5980281295127825],
                'Ader_2': [36.81298755438084, 24.27344499617392, 27.38149894239597, 0.6822063507301317, 0.0, 0.0, 0.0, 2.5689779135709925],
                'B_1': [0.0, 0.0, 0.0, 0.0, 0.0, 7.167230535550548, 6.243800844792131, 0.0],
                'B_2': [0.0, 0.0, 0.0, 0.0, 0.0, 7.884941445329839, 6.134683722446865, 0.0],
            }
        ),
        'conc_vial_if_undiluted_mg_L': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', 'hexadecanoic acid', '(z)-octadec-9-enoic acid', 'tetradecanoic acid', 'decanoic acid', '(e)-hexadec-9-enoic acid', 'phenol', '2-methylcyclopent-2-en-1-one'], name='conc_vial_if_undiluted_mg_L'),
            data={
                'A_1': [3279.7000117758744, 1651.3590445468228, 2840.462505206407, 589.5375911246907, 0.0, 0.0, 0.0, 0.0],
                'A_2': [3134.51947460942, 1527.9181711470565, 2845.682518041561, 593.2695577329649, 569.606946262709, 0.0, 0.0, 0.0],
                'Ader_1': [3920.970897868466, 3452.898704124259, 2708.635588562064, 75.12290512958799, 0.0, 324.7535161890978, 0.0, 0.0],
                'Ader_2': [4601.623444297605, 3422.6873677994963, 3034.1806245217404, 85.27579384126646, 0.0, 321.12223919637404, 0.0, 0.0],
                'B_1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.167230535550548, 6.243800844792131],
                'B_2': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.884941445329839, 6.134683722446865],
            }
        ),
        'fraction_of_sample_fr': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', 'hexadecanoic acid', '(z)-octadec-9-enoic acid', 'tetradecanoic acid', 'decanoic acid', '(e)-hexadec-9-enoic acid', 'phenol', '2-methylcyclopent-2-en-1-one'], name='fraction_of_sample_fr'),
            data={
                'A_1': [0.23426428655541953, 0.11795421746763018, 0.20289017894331474, 0.042109827937477896, 0.0, 0.0, 0.0, 0.0],
                'A_2': [0.2238942481863871, 0.10913701222478973, 0.20326303700296858, 0.04237639698092605, 0.04068621044733635, 0.0, 0.0, 0.0],
                'Ader_1': [0.2800693498477476, 0.24663562172316136, 0.193473970611576, 0.005365921794970571, 0.0, 0.023196679727792702, 0.0, 0.0],
                'Ader_2': [0.3286873888784004, 0.24447766912853547, 0.21672718746583858, 0.006091128131519033, 0.0, 0.022937302799741006, 0.0, 0.0],
                'B_1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0025597251912680527, 0.0022299288731400468],
                'B_2': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002816050516189228, 0.0021909584723024517],
            }
        ),
        'fraction_of_feedstock_fr': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', 'hexadecanoic acid', '(z)-octadec-9-enoic acid', 'tetradecanoic acid', 'decanoic acid', '(e)-hexadec-9-enoic acid', 'phenol', '2-methylcyclopent-2-en-1-one'], name='fraction_of_feedstock_fr'),
            data={
                'A_1': [0.10541892894993879, 0.053079397860433586, 0.09130058052449164, 0.018949422571865052, 0.0, 0.0, 0.0, 0.0],
                'A_2': [0.10299135416573807, 0.05020302562340328, 0.09350099702136555, 0.019493142611225985, 0.018715656805774722, 0.0, 0.0, 0.0],
                'Ader_1': [0.13163259442844139, 0.11591874220988584, 0.09093276618744071, 0.0025219832436361683, 0.0, 0.01090243947206257, 0.0, 0.0],
                'Ader_2': [0.1577699466616322, 0.11734928118169702, 0.10402904998360252, 0.0029237415031291357, 0.0, 0.011009905343875682, 0.0, 0.0],
                'B_1': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0012542653437213457, 0.001092665147838623],
                'B_2': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001408025258094614, 0.0010954792361512259],
            }
        ),

    }
    return reports

@pytest.fixture
def checked_files_param_aggrreps():
    reports = {
        'height': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'ketone', 'C-arom', 'alcohol', 'ester'], name='height'),
            data={
                'A_1': [1673299.018636137, 327039.28314786457, 0.0, 0.0, 0.0, 1301.4243936952162],
                'A_2': [1630562.5435198732, 318647.09040130535, 0.0, 0.0, 0.0, 1537.300436303604],
                'Ader_1': [995669.7397917995, 199362.50083044838, 0.0, 0.0, 0.0, 0.0],
                'Ader_2': [1101297.4487493301, 219631.74381979596, 0.0, 0.0, 0.0, 0.0],
                'B_1': [30189.53968239826, 5454.916540151818, 154971.12017291295, 52781.27325470194, 11641.780331526938, 0.0],
                'B_2': [24976.13637228884, 5726.94295844986, 167175.26975375, 61574.19209435766, 13581.203602167678, 0.0],
            }
        ),
        'area': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'ketone', 'C-arom', 'alcohol', 'ester'], name='area'),
            data={
                'A_1': [8180808.865926539, 1588883.8460032765, 0.0, 0.0, 0.0, 3437.311615109468],
                'A_2': [7957332.218705356, 1542835.8875348074, 0.0, 0.0, 0.0, 4857.58650996423],
                'Ader_1': [3115375.5519706816, 621383.3360462904, 0.0, 0.0, 0.0, 0.0],
                'Ader_2': [3373187.7167576416, 670272.3970037951, 0.0, 0.0, 0.0, 0.0],
                'B_1': [102813.63412565118, 18154.278785457453, 598982.0949258526, 164639.4578897035, 36313.94781638508, 0.0],
                'B_2': [85688.75337144051, 16092.723202157411, 653960.780006639, 187418.9512272872, 41338.34077143768, 0.0],
            }
        ),
        'area_if_undiluted': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'ketone', 'C-arom', 'ester', 'alcohol'], name='area_if_undiluted'),
            data={
                'A_1': [204520221.64816347, 39722096.15008192, 0.0, 0.0, 85932.7903777367, 0.0],
                'A_2': [198933305.4676339, 38570897.18837019, 0.0, 0.0, 121439.66274910574, 0.0],
                'Ader_1': [389421943.99633527, 77672917.00578631, 0.0, 0.0, 0.0, 0.0],
                'Ader_2': [421648464.59470516, 83784049.6254744, 0.0, 0.0, 0.0, 0.0],
                'B_1': [102813.63412565118, 18154.278785457453, 598982.0949258526, 164639.4578897035, 0.0, 36313.94781638508],
                'B_2': [85688.75337144051, 16092.723202157411, 653960.780006639, 187418.9512272872, 0.0, 41338.34077143768],
            }
        ),
        'conc_vial_mg_L': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='conc_vial_mg_L'),
            data={
                'A_1': [279.04506020687086, 55.41196015223182, 0.0, 0.0, 0.0],
                'A_2': [287.2245498713612, 59.62973999202588, 0.0, 0.0, 0.0],
                'Ader_1': [69.94687725385289, 13.916672123312216, 0.0, 0.0, 0.0],
                'Ader_2': [76.54999067692155, 15.17432861960896, 0.0, 0.0, 0.0],
                'B_1': [2.7332298278341587, 0.0, 5.872239694763155, 3.706919814979471, 1.2952193148242288],
                'B_2': [2.6854637025308077, 0.0, 6.460273032447163, 3.6421374119160257, 1.424919765813671],
            }
        ),
        'conc_vial_if_undiluted_mg_L': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='conc_vial_if_undiluted_mg_L'),
            data={
                'A_1': [6976.126505171772, 1385.2990038057956, 0.0, 0.0, 0.0],
                'A_2': [7180.61374678403, 1490.7434998006472, 0.0, 0.0, 0.0],
                'Ader_1': [8743.35965673161, 1739.5840154140271, 0.0, 0.0, 0.0],
                'Ader_2': [9568.748834615195, 1896.79107745112, 0.0, 0.0, 0.0],
                'B_1': [2.7332298278341587, 0.0, 5.872239694763155, 3.706919814979471, 1.2952193148242288],
                'B_2': [2.6854637025308077, 0.0, 6.460273032447163, 3.6421374119160257, 1.424919765813671],
            }
        ),
        'fraction_of_sample_fr': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='fraction_of_sample_fr'),
            data={
                'A_1': [0.4982947503694122, 0.09894992884327108, 0.0, 0.0, 0.0],
                'A_2': [0.5129009819131448, 0.10648167855718907, 0.0, 0.0, 0.0],
                'Ader_1': [0.6245256897665437, 0.12425600110100192, 0.0, 0.0, 0.0],
                'Ader_2': [0.6834820596153709, 0.1354850769607943, 0.0, 0.0, 0.0],
                'B_1': [0.0009761535099407709, 0.0, 0.0020972284624154124, 0.0013238999339212397, 0.00046257832672293885],
                'B_2': [0.0009590941794752884, 0.0, 0.0023072403687311297, 0.0013007633613985805, 0.0005088999163620254],
            }
        ),
        'fraction_of_feedstock_fr': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='fraction_of_feedstock_fr'),
            data={
                'A_1': [0.22423263766623547, 0.04452746797947199, 0.0, 0.0, 0.0],
                'A_2': [0.23593445168004665, 0.04898157213630697, 0.0, 0.0, 0.0],
                'Ader_1': [0.29352707419027546, 0.058400320517470905, 0.0, 0.0, 0.0],
                'Ader_2': [0.32807138861537805, 0.06503283694118125, 0.0, 0.0, 0.0],
                'B_1': [0.00047831521987097775, 0.0, 0.001027641946583552, 0.0006487109676214074, 0.00022666338009424],
                'B_2': [0.0004795470897376442, 0.0, 0.0011536201843655649, 0.0006503816806992902, 0.0002544499581810127],
            }
        )

    }
    return reports

@pytest.fixture
def checked_samples_param_reports():
    reports = {
        'height': pd.DataFrame(
            index=pd.Index(['(z)-octadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', 'hexane-2,5-dione', 'phenol', 'butan-2-one', '2-methylcyclopent-2-en-1-one', 'tetradecanoic acid', '(e)-pent-2-enoic acid', '(e)-hexadec-9-enoic acid', 'oxacycloheptadecan-2-one', 'decanoic acid', '4,5-dimethylhex-1-ene'], name='height'),
            data={
                'A': [1143123.0, 477850.5, 332147.5, 0.0, 0.0, 0.0, 0.0, 14769.5, 0.0, 0.0, 6222.0, 4259.0, 0.0],
                'Ader': [548218.5, 490697.0, 191724.5, 0.0, 0.0, 0.0, 0.0, 18394.0, 0.0, 8953.5, 0.0, 0.0, 0.0],
                'B': [0.0, 0.0, 0.0, 116469.5, 69787.0, 41972.0, 19169.0, 0.0, 12434.5, 0.0, 0.0, 0.0, 3597.0],
            }
        ),
        'area': pd.DataFrame(
            index=pd.Index(['(z)-octadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', 'hexane-2,5-dione', 'phenol', 'butan-2-one', '2-methylcyclopent-2-en-1-one', 'tetradecanoic acid', '(e)-pent-2-enoic acid', '(e)-hexadec-9-enoic acid', 'oxacycloheptadecan-2-one', 'decanoic acid', '4,5-dimethylhex-1-ene'], name='area'),
            data={
                'A': [6387230.0, 1767468.0, 1413594.0, 0.0, 0.0, 0.0, 0.0, 47519.5, 0.0, 0.0, 18181.0, 10952.0, 0.0],
                'Ader': [1840687.0, 1409097.5, 562306.5, 0.0, 0.0, 0.0, 0.0, 51560.5, 0.0, 26505.5, 0.0, 0.0, 0.0],
                'B': [0.0, 0.0, 0.0, 456719.5, 214848.5, 164293.5, 66877.0, 0.0, 38083.5, 0.0, 0.0, 0.0, 9761.0],
            }
        ),
        'area_if_undiluted': pd.DataFrame(
            index=pd.Index(['(z)-octadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', 'tetradecanoic acid', '(e)-hexadec-9-enoic acid', 'hexane-2,5-dione', 'oxacycloheptadecan-2-one', 'decanoic acid', 'phenol', 'butan-2-one', '2-methylcyclopent-2-en-1-one', '(e)-pent-2-enoic acid', '4,5-dimethylhex-1-ene'], name='area_if_undiluted'),
            data={
                'A': [159680750.0, 44186700.0, 35339850.0, 1187987.5, 0.0, 0.0, 454525.0, 273800.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'Ader': [230085875.0, 176137187.5, 70288312.5, 6445062.5, 3313187.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'B': [0.0, 0.0, 0.0, 0.0, 0.0, 456719.5, 0.0, 0.0, 214848.5, 164293.5, 66877.0, 38083.5, 9761.0],
            }
        ),
        'conc_vial_mg_L': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid', 'hexadecanoic acid', 'tetradecanoic acid', 'decanoic acid', 'phenol', '2-methylcyclopent-2-en-1-one', '(e)-hexadec-9-enoic acid'], name='conc_vial_mg_L'),
            data={
                'A': [128.28438972770587, 113.72290046495935, 63.58554431387759, 23.65614297715311, 22.78427785050836, 0.0, 0.0, 0.0],
                'Ader': [34.090377368664285, 22.971264852335217, 27.502344287695024, 0.6415947958834178, 0.0, 0.0, 0.0, 2.5835030215418877],
                'B': [0.0, 0.0, 0.0, 0.0, 0.0, 7.526085990440194, 6.189242283619498, 0.0],
            }
        ),
        'conc_vial_if_undiluted_mg_L': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', 'hexadecanoic acid', '(z)-octadec-9-enoic acid', 'tetradecanoic acid', 'decanoic acid', '(e)-hexadec-9-enoic acid', 'phenol', '2-methylcyclopent-2-en-1-one'], name='conc_vial_if_undiluted_mg_L'),
            data={
                'A': [3207.1097431926473, 1589.6386078469395, 2843.0725116239837, 591.4035744288278, 569.606946262709, 0.0, 0.0, 0.0],
                'Ader': [4261.297171083035, 3437.7930359618776, 2871.4081065419023, 80.19934948542723, 0.0, 322.9378776927359, 0.0, 0.0],
                'B': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.526085990440194, 6.189242283619498],
            }
        ),
        'fraction_of_sample_fr': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', 'hexadecanoic acid', '(z)-octadec-9-enoic acid', 'tetradecanoic acid', 'decanoic acid', '(e)-hexadec-9-enoic acid', 'phenol', '2-methylcyclopent-2-en-1-one'], name='fraction_of_sample_fr'),
            data={
                'A': [0.2290792673709033, 0.11354561484620995, 0.20307660797314164, 0.042243112459201974, 0.04068621044733635, 0.0, 0.0, 0.0],
                'Ader': [0.304378369363074, 0.24555664542584843, 0.2051005790387073, 0.005728524963244802, 0.0, 0.023066991263766854, 0.0, 0.0],
                'B': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0026878878537286406, 0.0022104436727212492],
            }
        ),
        'fraction_of_feedstock_fr': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', 'hexadecanoic acid', '(z)-octadec-9-enoic acid', 'tetradecanoic acid', 'decanoic acid', '(e)-hexadec-9-enoic acid', 'phenol', '2-methylcyclopent-2-en-1-one'], name='fraction_of_feedstock_fr'),
            data={
                'A': [0.10420514155783843, 0.051641211741918436, 0.0924007887729286, 0.01922128259154552, 0.018715656805774722, 0.0, 0.0, 0.0],
                'Ader': [0.1447012705450368, 0.11663401169579143, 0.09748090808552162, 0.002722862373382652, 0.0, 0.010956172407969126, 0.0, 0.0],
                'B': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00133114530090798, 0.0010940721919949245],
            }
        )
    }
    return reports

@pytest.fixture
def checked_samples_param_reports_std():
    reports = {
        'height': pd.DataFrame(
            index=pd.Index(['(z)-octadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', 'hexane-2,5-dione', 'phenol', 'butan-2-one', '2-methylcyclopent-2-en-1-one', 'tetradecanoic acid', '(e)-pent-2-enoic acid', '(e)-hexadec-9-enoic acid', 'oxacycloheptadecan-2-one', 'decanoic acid', '4,5-dimethylhex-1-ene'], name='height'),
            data={
                'A': [6330.019905181974, 22498.01645701238, 10546.497641397356, np.nan, np.nan, np.nan, np.nan, 352.8462838120872, np.nan, np.nan, 731.1484117468901, np.nan, np.nan],
                'Ader': [65717.79713669654, 8212.338156700564, 15378.865384026221, np.nan, np.nan, np.nan, np.nan, 29.698484809834994, np.nan, 252.43712088359746, np.nan, np.nan, np.nan],
                'B': [np.nan, np.nan, np.nan, 5193.699307815192, 7588.669975694028, 3647.2567773602123, 924.8956697920041, np.nan, 427.79960261786124, np.nan, np.nan, np.nan, 5086.926183856023],
            }
        ),
        'area': pd.DataFrame(
            index=pd.Index(['(z)-octadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', 'hexane-2,5-dione', 'phenol', 'butan-2-one', '2-methylcyclopent-2-en-1-one', 'tetradecanoic acid', '(e)-pent-2-enoic acid', '(e)-hexadec-9-enoic acid', 'oxacycloheptadecan-2-one', 'decanoic acid', '4,5-dimethylhex-1-ene'], name='area'),
            data={
                'A': [10575.489019426004, 156570.4119174501, 60139.43173991587, np.nan, np.nan, np.nan, np.nan, 4427.195557008974, np.nan, np.nan, 4402.446819667445, np.nan, np.nan],
                'Ader': [163869.16811285765, 8637.309332193678, 60571.47398322085, np.nan, np.nan, np.nan, np.nan, 2902.6733367707775, np.nan, 1827.8710293672254, np.nan, np.nan, np.nan],
                'B': [np.nan, np.nan, np.nan, 22121.83564942114, 19659.68983732958, 23656.25736459595, 3317.745017327281, np.nan, 3242.0845917403203, np.nan, np.nan, np.nan, 13804.138582323782],
            }
        ),
        'area_if_undiluted': pd.DataFrame(
            index=pd.Index(['(z)-octadec-9-enoic acid', 'hexadecanoic acid', '(9z,12z)-octadeca-9,12-dienoic acid', 'tetradecanoic acid', '(e)-hexadec-9-enoic acid', 'hexane-2,5-dione', 'oxacycloheptadecan-2-one', 'decanoic acid', 'phenol', 'butan-2-one', '2-methylcyclopent-2-en-1-one', '(e)-pent-2-enoic acid', '4,5-dimethylhex-1-ene'], name='area_if_undiluted'),
            data={
                'A': [264387.2254856501, 3914260.2979362523, 1503485.7934978968, 110679.88892522435, np.nan, np.nan, 110061.17049168612, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                'Ader': [20483646.014107205, 1079663.6665242098, 7571434.247902606, 362834.1670963472, 228483.87867090316, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                'B': [np.nan, np.nan, np.nan, np.nan, np.nan, 22121.83564942114, np.nan, np.nan, 19659.68983732958, 23656.25736459595, 3317.745017327281, 3242.0845917403203, 13804.138582323782],
            }
        ),
        'conc_vial_mg_L': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', '(z)-octadec-9-enoic acid', 'hexadecanoic acid', 'tetradecanoic acid', 'decanoic acid', 'phenol', '2-methylcyclopent-2-en-1-one', '(e)-hexadec-9-enoic acid'], name='conc_vial_mg_L'),
            data={
                'A': [4.106325693068217, 0.14764425894471855, 3.4914351462625968, 0.10555595583489899, np.nan, np.nan, np.nan, np.nan],
                'Ader': [3.8503522496954825, 1.8415608200696434, 0.1709011262715786, 0.05743341165328154, np.nan, np.nan, np.nan, 0.02054160468737343],
                'B': [np.nan, np.nan, np.nan, np.nan, np.nan, 0.5074982512365029, 0.07715745715389961, np.nan],
            }
        ),
        'conc_vial_if_undiluted_mg_L': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', 'hexadecanoic acid', '(z)-octadec-9-enoic acid', 'tetradecanoic acid', 'decanoic acid', '(e)-hexadec-9-enoic acid', 'phenol', '2-methylcyclopent-2-en-1-one'], name='conc_vial_if_undiluted_mg_L'),
            data={
                'A': [102.65814232670576, 87.28587865656482, 3.691106473618185, 2.638898895872451, np.nan, np.nan, np.nan, np.nan],
                'Ader': [481.29403121193536, 21.362640783947153, 230.19510250870547, 7.179176456660192, np.nan, 2.5677005859216706, np.nan, np.nan],
                'B': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.5074982512365029, 0.07715745715389961],
            }
        ),
        'fraction_of_sample_fr': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', 'hexadecanoic acid', '(z)-octadec-9-enoic acid', 'tetradecanoic acid', 'decanoic acid', '(e)-hexadec-9-enoic acid', 'phenol', '2-methylcyclopent-2-en-1-one'], name='fraction_of_sample_fr'),
            data={
                'A': [0.007332724451907525, 0.006234705618326057, 0.00026365046240129915, 0.00018849277827660252, np.nan, np.nan, np.nan, np.nan],
                'Ader': [0.03437814508656681, 0.001525902913139085, 0.01644250732205038, 0.0005127983183328709, np.nan, 0.00018340718470868995, np.nan, np.nan],
                'B': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00018124937544160814, 2.7556234697821363e-05],
            }
        ),
        'fraction_of_feedstock_fr': pd.DataFrame(
            index=pd.Index(['(9z,12z)-octadeca-9,12-dienoic acid', 'hexadecanoic acid', '(z)-octadec-9-enoic acid', 'tetradecanoic acid', 'decanoic acid', '(e)-hexadec-9-enoic acid', 'phenol', '2-methylcyclopent-2-en-1-one'], name='fraction_of_feedstock_fr'),
            data={
                'A': [0.001716554591745799, 0.002033902314020852, 0.001555929426374292, 0.0003844681268991305, np.nan, np.nan, np.nan, np.nan],
                'Ader': [0.01848189900635057, 0.0010115438077193133, 0.009260471080609506, 0.00028408598968518184, np.nan, 7.598984670517598e-05, np.nan, np.nan],
                'B': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00010872467812800095, 1.9898609286992866e-06],
            }
        )
    }
    return reports


@pytest.fixture
def checked_samples_param_aggrreps():
    reports = {
        'height': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'ketone', 'C-arom', 'alcohol', 'ester'], name='height'),
            data={
                'A': [1653503.8741030835, 323399.6926465227, 0.0, 0.0, 0.0, 1419.36241499941],
                'Ader': [1048483.5942705647, 209497.12232512212, 0.0, 0.0, 0.0, 0.0],
                'B': [27582.83802734355, 5590.929749300839, 161073.19496333148, 57177.7326745298, 12611.491966847307, 0.0],
            }
        ),
        'area': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'ketone', 'C-arom', 'alcohol', 'ester'], name='area'),
            data={
                'A': [8073115.744196826, 1567290.9192014115, 0.0, 0.0, 0.0, 4147.449062536849],
                'Ader': [3244281.634364161, 645827.8665250427, 0.0, 0.0, 0.0, 0.0],
                'B': [94251.19374854585, 17123.500993807433, 626471.4374662458, 176029.20455849537, 38826.144293911384, 0.0],
            }
        ),
        'area_if_undiluted': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'ketone', 'C-arom', 'ester', 'alcohol'], name='area_if_undiluted'),
            data={
                'A': [201827893.6049206, 39182272.9800353, 0.0, 0.0, 103686.22656342122, 0.0],
                'Ader': [405535204.2955202, 80728483.31563035, 0.0, 0.0, 0.0, 0.0],
                'B': [94251.19374854585, 17123.500993807433, 626471.4374662458, 176029.20455849537, 0.0, 38826.144293911384],
            }
        ),
        'conc_vial_mg_L': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='conc_vial_mg_L'),
            data={
                'A': [291.5503459097622, 60.497977193910856, 0.0, 0.0, 0.0],
                'Ader': [73.2484339653872, 14.545500371460587, 0.0, 0.0, 0.0],
                'B': [2.709346765182483, 0.0, 6.166256363605159, 3.674528613447748, 1.36006954031895],
            }
        ),
        'conc_vial_if_undiluted_mg_L': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='conc_vial_if_undiluted_mg_L'),
            data={
                'A': [7288.758647744056, 1512.4494298477714, 0.0, 0.0, 0.0],
                'Ader': [9156.054245673402, 1818.1875464325735, 0.0, 0.0, 0.0],
                'B': [2.709346765182483, 0.0, 6.166256363605159, 3.674528613447748, 1.36006954031895],
            }
        ),
        'fraction_of_sample_fr': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='fraction_of_sample_fr'),
            data={
                'A': [0.5206256176960039, 0.10803210213198365, 0.0, 0.0, 0.0],
                'Ader': [0.6540038746909574, 0.12987053903089812, 0.0, 0.0, 0.0],
                'B': [0.0009676238447080297, 0.0, 0.002202234415573271, 0.0013123316476599102, 0.00048573912154248214],
            }
        ),
        'fraction_of_feedstock_fr': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='fraction_of_feedstock_fr'),
            data={
                'A': [0.23699631038831473, 0.04920001733649612, 0.0, 0.0, 0.0],
                'Ader': [0.3107992314028268, 0.06171657872932608, 0.0, 0.0, 0.0],
                'B': [0.000478931154804311, 0.0, 0.0010906310654745584, 0.0006495463241603489, 0.00024055666913762634],
            }
        )
    }
    return reports

@pytest.fixture
def checked_samples_param_aggrreps_std():
    reports = {
        'height': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'ketone', 'C-arom', 'alcohol', 'ester'], name='height'),
            data={
                'A': [33572.681565843486, 6721.194551751164, 0.0, 0.0, 0.0, 166.78954924783815],
                'Ader': [75153.30566953593, 14433.563492713163, 0.0, 0.0, 0.0, 0.0],
                'B': [6487.963541864086, 192.35172504043408, 8629.636927224863, 6217.532537943509, 1371.3793462610597, 0.0],
            }
        ),
        'area': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'ketone', 'C-arom', 'alcohol', 'ester'], name='area'),
            data={
                'A': [195426.4379503677, 39700.497529875116, 0.0, 0.0, 0.0, 1004.2860093008128],
                'Ader': [199552.22488919983, 38249.3835185481, 0.0, 0.0, 0.0, 0.0],
                'B': [21973.821276880837, 1457.7399327444466, 42815.265175930195, 16107.534210999198, 3552.7823298636085, 0.0],
            }
        ),
        'area_if_undiluted': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'ketone', 'C-arom', 'ester', 'alcohol'], name='area_if_undiluted'),
            data={
                'A': [4885660.948759192, 992512.4382468779, 0.0, 0.0, 25107.15023252032, 0.0],
                'Ader': [24944028.111149978, 4781172.939818514, 0.0, 0.0, 0.0, 0.0],
                'B': [21973.821276880837, 1457.7399327444466, 42815.26517593019, 16107.534210999198, 0.0, 3552.7823298636085],
            }
        ),
        'conc_vial_mg_L': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='conc_vial_mg_L'),
            data={
                'A': [6.5353020421660615, 1.3165427371870901, 0.0, 0.0, 0.0],
                'Ader': [4.984729502757059, 0.9565736502096863, 0.0, 0.0, 0.0],
                'B': [0.03377575111300582, 0.0, 0.41580236064012105, 0.04580807650772246, 0.09171206841758799],
            }
        ),
        'conc_vial_if_undiluted_mg_L': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='conc_vial_if_undiluted_mg_L'),
            data={
                'A': [163.3825510541519, 32.91356842967732, 0.0, 0.0, 0.0],
                'Ader': [623.0911878446323, 119.57170627621078, 0.0, 0.0, 0.0],
                'B': [0.03377575111300582, 0.0, 0.41580236064012105, 0.04580807650772246, 0.09171206841758799],
            }
        ),
        'fraction_of_sample_fr': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='fraction_of_sample_fr'),
            data={
                'A': [0.01167018221815368, 0.002350969173548375, 0.0, 0.0, 0.0],
                'Ader': [0.04450651341747372, 0.00854083616258648, 0.0, 0.0, 0.0],
                'B': [1.2062768254644968e-05, 0.0, 0.0001485008430857575, 1.6360027324186634e-05, 3.275431014913856e-05],
            }
        ),
        'fraction_of_feedstock_fr': pd.DataFrame(
            index=pd.Index(['C-aliph', 'carboxyl', 'C-arom', 'ketone', 'alcohol'], name='fraction_of_feedstock_fr'),
            data={
                'A': [0.004734701118759861, 0.0009563848477203726, 0.0, 0.0, 0.0],
                'Ader': [0.024426518981430268, 0.004689897339536736, 0.0, 0.0, 0.0],
                'B': [8.710635362591769e-07, 0.0, 8.908006621759261e-05, 1.1813725467879507e-06, 1.9648077791126472e-05],
            }
        )
    }
    return reports

