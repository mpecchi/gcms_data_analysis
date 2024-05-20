import pytest
from pandas.testing import assert_frame_equal


def test_files_info(proj):
    files_info_created = proj.create_files_info(update_saved_files_info=False)
    files_info_loaded = proj.load_files_info(update_saved_files_info=False)
    assert_frame_equal(
        files_info_created[["samplename", "replicatenumber"]],
        files_info_loaded[["samplename", "replicatenumber"]],
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )
    assert files_info_created["dilution_factor"].tolist() == [1, 1, 1, 1, 1]
    assert files_info_loaded["dilution_factor"].tolist() == [2, 2, 1, 1, 1]


def test_load_all_files(proj, checked_files, checked_is_files_deriv):

    files = proj.load_all_files()
    assert list(files.keys()) == ["S_1", "S_2", "T_1", "T_2", "T_3"]


def test_load_all_files_wrong_names(proj):
    wrong_files_info = proj.create_files_info()
    wrong_files_info.index = ["Wrong_filename"] + wrong_files_info.index.tolist()[1:]
    proj.files_info = wrong_files_info
    with pytest.raises(FileNotFoundError):
        proj.load_all_files()


def test_load_class_code_fractions(proj, checked_load_class_code_fractions):
    to_check = proj.load_class_code_frac()
    assert_frame_equal(
        to_check,
        checked_load_class_code_fractions,
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_load_calibrations(
    proj,
    checked_load_calibrations,
):
    calibrations = proj.load_calibrations()
    assert calibrations["cal_minimal"].index.tolist() == [
        "phenol",
        "naphthalene",
        "dodecane",
        "capric acid",
    ]
    assert calibrations["cal_minimal"].columns.tolist() == [
        "MW",
        "PPM 1",
        "PPM 2",
        "PPM 3",
        "PPM 4",
        "PPM 5",
        "PPM 6",
        "Area 1",
        "Area 2",
        "Area 3",
        "Area 4",
        "Area 5",
        "Area 6",
    ]


def test_list_of_all_compounds(proj):
    to_check = proj.create_list_of_all_compounds()
    print(to_check.sort())
    assert (
        to_check.sort()
        == [
            "capric acid",
            "dodecane",
            "notvalidcomp",
            "p-dichlorobenzene",
            # "oleic acid",
            # "phenol",
            # "naphthalene",
            "palmitic acid",
        ].sort()
    )
    # assert to_check.sort() == checked_list_of_all_compounds.sort()


# def test_list_of_all_deriv_compounds(proj, checked_list_of_all_deriv_compounds):
#     to_check = proj.create_list_of_all_deriv_compounds()
#     assert to_check.sort() == checked_list_of_all_deriv_compounds.sort()


@pytest.mark.slow
def test_create_compounds_properties(proj, checked_compounds_properties):
    to_check = proj.create_compounds_properties()
    assert_frame_equal(
        to_check.sort_index(),
        checked_compounds_properties.sort_index(),
        check_exact=False,
        check_dtype=False,
        atol=1e-5,
        rtol=1e-5,
    )


# @pytest.mark.slow
# def test_create_deriv_compounds_properties(proj, checked_deriv_compounds_properties):
#     to_check = proj.create_deriv_compounds_properties()
#     assert_frame_equal(
#         to_check.sort_index(),
#         checked_deriv_compounds_properties.sort_index(),
#         check_exact=False,
#         atol=1e-3,
#         rtol=1e-3,
#     )


def test_load_compounds_properties(proj, checked_compounds_properties):
    to_check = proj.load_compounds_properties()
    assert_frame_equal(
        to_check.sort_index(),
        checked_compounds_properties.sort_index(),
        check_exact=False,
        atol=1e-3,
        rtol=1e-3,
    )


# def test_load_deriv_compounds_properties(proj, checked_deriv_compounds_properties):
#     to_check = proj.load_deriv_compounds_properties()
#     assert_frame_equal(
#         to_check.sort_index(),
#         checked_deriv_compounds_properties.sort_index(),
#         check_exact=False,
#         atol=1e-3,
#         rtol=1e-3,
#     )


def test_create_samples_info(proj, checked_samples_info, checked_samples_info_std):
    to_check, to_check_std = proj.create_samples_info()
    assert_frame_equal(
        to_check, checked_samples_info, check_exact=False, atol=1e-5, rtol=1e-5
    )
    assert_frame_equal(
        to_check_std, checked_samples_info_std, check_exact=False, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "parameter",
    [
        "height",
        "area",
        "area_if_undiluted",
        "conc_vial_mg_L",
        "conc_vial_if_undiluted_mg_L",
        "fraction_of_sample_fr",
        "fraction_of_feedstock_fr",
    ],
)
def test_files_param_reports(proj, checked_files_param_reports, parameter):
    to_check = proj.create_files_param_report(param=parameter)
    checked_report = checked_files_param_reports[parameter]
    assert_frame_equal(
        to_check.sort_index(),
        checked_report.sort_index(),
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_files_param_reports_exception(proj):
    with pytest.raises(ValueError):
        proj.create_files_param_report(param="wrong_parameter")


@pytest.mark.parametrize(
    "parameter",
    [
        "height",
        "area",
        "area_if_undiluted",
        "conc_vial_mg_L",
        "conc_vial_if_undiluted_mg_L",
        "fraction_of_sample_fr",
        "fraction_of_feedstock_fr",
    ],
)
def test_files_param_aggrreps(proj, checked_files_param_aggrreps, parameter):
    to_check = proj.create_files_param_aggrrep(param=parameter)
    checked_report = checked_files_param_aggrreps[parameter]
    assert_frame_equal(
        to_check.sort_index(),
        checked_report.sort_index(),
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_files_param_aggreps_exception(proj):
    with pytest.raises(ValueError):
        proj.create_files_param_aggrrep(param="wrong_parameter")


@pytest.mark.parametrize(
    "parameter",
    [
        "height",
        "area",
        "area_if_undiluted",
        "conc_vial_mg_L",
        "conc_vial_if_undiluted_mg_L",
        "fraction_of_sample_fr",
        "fraction_of_feedstock_fr",
    ],
)
def test_samples_param_reports(
    proj, checked_samples_param_reports, checked_samples_param_reports_std, parameter
):
    to_check, to_check_std = proj.create_samples_param_report(param=parameter)
    checked_report = checked_samples_param_reports[parameter]
    checked_report_std = checked_samples_param_reports_std[parameter]
    assert_frame_equal(
        to_check.sort_index(),
        checked_report.sort_index(),
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )
    assert_frame_equal(
        to_check_std.sort_index(),
        checked_report_std.sort_index(),
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_samples_param_reports_exception(proj):
    with pytest.raises(ValueError):
        proj.create_samples_param_report(param="wrong_parameter")


@pytest.mark.parametrize(
    "parameter",
    [
        "height",
        "area",
        "area_if_undiluted",
        "conc_vial_mg_L",
        "conc_vial_if_undiluted_mg_L",
        "fraction_of_sample_fr",
        "fraction_of_feedstock_fr",
    ],
)
def test_samples_param_aggrreps(
    proj, checked_samples_param_aggrreps, checked_samples_param_aggrreps_std, parameter
):
    to_check, to_check_std = proj.create_samples_param_aggrrep(param=parameter)
    checked_report = checked_samples_param_aggrreps[parameter]
    checked_report_std = checked_samples_param_aggrreps_std[parameter]
    assert_frame_equal(
        to_check.sort_index(),
        checked_report.sort_index(),
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )
    assert_frame_equal(
        to_check_std.sort_index(),
        checked_report_std.sort_index(),
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_samples_param_aggrreps_exception(proj):
    with pytest.raises(ValueError):
        proj.create_samples_param_aggrrep(param="wrong_parameter")
