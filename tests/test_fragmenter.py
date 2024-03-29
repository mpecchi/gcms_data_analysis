import pytest
from gcms_data_analysis.fragmenter import Fragmenter


def test_fragmenter_simple():
    """test simple algorithm for fragmenter,
    gcms_data analysi only uses the simple fragmentation
    """
    algorithm = "simple"
    smiles = ["CCCCO", "CCCO", "CCO", "CO"]
    fragmentation_scheme = {
        "CH2": "[CH2]",
        "OH": "[OH]",
        "CH3": "[CH3]",
        "CH2-CH2": "[CH2][CH2]",
    }

    checked_fragmentations_1 = {
        "CCCCO": {"CH2-CH2": 1, "CH3": 1, "CH2": 1, "OH": 1},
        "CCCO": {"CH2-CH2": 1, "CH3": 1, "OH": 1},
        "CCO": {"CH3": 1, "CH2": 1, "OH": 1},
        "CO": {"CH3": 1, "OH": 1},
    }

    fragmentation_scheme_order_1 = ["CH2-CH2", "CH3", "CH2", "OH"]

    for smi in smiles:
        frg = Fragmenter(
            fragmentation_scheme,
            fragmentation_scheme_order=fragmentation_scheme_order_1,
            algorithm=algorithm,
        )
        fragmentation, _, _ = frg.fragment(smi)
        assert fragmentation == checked_fragmentations_1[smi]

    fragmentation_scheme_order_2 = ["CH3", "CH2", "CH2-CH2", "OH"]
    checked_fragmentations_2 = {
        "CCCCO": {"CH3": 1, "CH2": 3, "OH": 1},
        "CCCO": {"CH3": 1, "CH2": 2, "OH": 1},
        "CCO": {"CH3": 1, "CH2": 1, "OH": 1},
        "CO": {"CH3": 1, "OH": 1},
    }
    for smi in smiles:
        frg = Fragmenter(
            fragmentation_scheme,
            fragmentation_scheme_order=fragmentation_scheme_order_2,
            algorithm=algorithm,
        )
        fragmentation, _, _ = frg.fragment(smi)
        assert fragmentation == checked_fragmentations_2[smi]
