import pytest
from gcms_data_analysis import name_to_properties
from pandas.testing import assert_frame_equal
import pandas as pd
import numpy as np


@pytest.mark.parametrize("compound_name", [" ", None, False, np.nan])
def test_name_to_properties_wrong_input_df_empty(
    compound_name, dicts_classifications_codes_fractions
):
    dict_class_to_code, dict_class_to_mass_fraction = (
        dicts_classifications_codes_fractions
    )
    df = pd.DataFrame()
    to_check = name_to_properties(
        compound_name, dict_class_to_code, dict_class_to_mass_fraction, df
    )
    assert to_check.empty


@pytest.mark.parametrize("compound_name", [" ", None, False, np.nan])
def test_name_to_properties_wrong_input_df_not_empty(
    compound_name,
    dicts_classifications_codes_fractions,
    checked_n2p_compounds_properties,
):
    dict_class_to_code, dict_class_to_mass_fraction = (
        dicts_classifications_codes_fractions
    )
    to_check = name_to_properties(
        compound_name,
        dict_class_to_code,
        dict_class_to_mass_fraction,
        checked_n2p_compounds_properties,
    )
    assert_frame_equal(
        to_check,
        checked_n2p_compounds_properties,
        check_exact=False,
        atol=1e-3,
        rtol=1e-3,
    )


def test_name_to_properties_name_not_on_pubchem_df_empty(
    dicts_classifications_codes_fractions,
):
    dict_class_to_code, dict_class_to_mass_fraction = (
        dicts_classifications_codes_fractions
    )
    df = pd.DataFrame()
    to_check = name_to_properties(
        "name_not_on_pcp", dict_class_to_code, dict_class_to_mass_fraction, df
    )
    df.loc["name_not_on_pcp", "iupac_name"] = "unidentified"
    assert_frame_equal(
        to_check,
        df,
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_name_to_properties_name_not_on_pubchem_df_not_empty(
    dicts_classifications_codes_fractions,
    checked_n2p_compounds_properties,
):
    dict_class_to_code, dict_class_to_mass_fraction = (
        dicts_classifications_codes_fractions
    )
    to_check = name_to_properties(
        "name_not_on_pcp",
        dict_class_to_code,
        dict_class_to_mass_fraction,
        checked_n2p_compounds_properties,
    )
    checked_n2p_compounds_properties.loc["name_not_on_pcp", "iupac_name"] = (
        "unidentified"
    )
    assert_frame_equal(
        to_check,
        checked_n2p_compounds_properties,
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "compound",
    [
        "2-methylcyclopent-2-en-1-one",  # Comment: small ketone
        "hexadecanoic acid",  # Comment: another compound
        "n-hexadecanoic acid",  # Comment: different names, same compounds
        "phenol",  # Comment: a ring structure
        "phenol",  # Comment: repeated compound to test idempotency
        "carbolic acid",  # same iupac but different comp_name
        "2,4,5-trichlorophenol",  # Comment: chlorine, unidentified
        "phenoxytrimethylsilane",  # Comment: silane, not listed in fg
        "bromophenol",  # Comment: Br not listed
        "9-octadecenoic acid, 1,2,3-propanetriyl ester, (e,e,e)-",  # Comment: large compound
    ],
)
def test_name_to_properties_single_compounds(
    compound, dicts_classifications_codes_fractions, checked_n2p_compounds_properties
):
    dict_class_to_code, dict_class_to_mass_fraction = (
        dicts_classifications_codes_fractions
    )

    to_check = name_to_properties(
        compound, dict_class_to_code, dict_class_to_mass_fraction
    )
    to_check = to_check.loc[[compound], :]
    to_check = to_check.loc[:, (to_check != 0).any(axis=0)]
    checked = checked_n2p_compounds_properties.loc[[compound], :]
    checked = checked.loc[:, (checked != 0).any(axis=0)]
    assert_frame_equal(
        to_check,
        checked,
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_name_to_properties_all_compounds(
    dicts_classifications_codes_fractions, checked_n2p_compounds_properties
):
    dict_class_to_code, dict_class_to_mass_fraction = (
        dicts_classifications_codes_fractions
    )

    compounds = [
        "2-methylcyclopent-2-en-1-one",  # Comment: small ketone
        "hexadecanoic acid",  # Comment: another compound
        "n-hexadecanoic acid",  # Comment: different names, same compounds
        "phenol",  # Comment: a ring structure
        "phenol",  # Comment: repeated compound to test
        "carbolic acid",  # same iupac but different comp_name
        "2,4,5-trichlorophenol",  # Comment: chlorine, unidentified
        "phenoxytrimethylsilane",  # Comment: silane, not listed in fg
        "bromophenol",  # Comment: Br not listed
        "9-octadecenoic acid, 1,2,3-propanetriyl ester, (e,e,e)-",  # Comment: large compound
        "name_not_on_pcp",  # test for legit string that gives no pcp result
        " ",  # wrong entry or datatype
        None,
        False,
        np.nan,
    ]
    to_check = pd.DataFrame()
    for compound in compounds:
        to_check = name_to_properties(
            compound, dict_class_to_code, dict_class_to_mass_fraction, to_check
        )
    checked = checked_n2p_compounds_properties
    assert_frame_equal(
        to_check,
        checked,
        check_exact=False,
        atol=1e-3,
        rtol=1e-3,
    )
