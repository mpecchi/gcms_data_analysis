import pytest
from pandas.testing import assert_frame_equal


def test_load_files_info(gcms, checked_files_info):
    to_check = gcms.load_files_info()

    assert_frame_equal(
        to_check, checked_files_info, check_exact=False, atol=1e-5, rtol=1e-5
    )


def test_create_files_info(gcms, checked_created_files_info):
    to_check = gcms.create_files_info()
    assert_frame_equal(
        to_check, checked_created_files_info, check_exact=False, atol=1e-5, rtol=1e-5
    )


def test_load_all_files(gcms, checked_files, checked_is_files_deriv):

    files_to_check, is_deriv_files_to_check = gcms.load_all_files()
    for filename_to_check, checked_filename in zip(files_to_check, checked_files):
        assert filename_to_check == checked_filename

    for file_to_check, checked_file in zip(
        files_to_check.values(), checked_files.values()
    ):
        assert_frame_equal(
            file_to_check, checked_file, check_exact=False, atol=1e-5, rtol=1e-5
        )

    assert is_deriv_files_to_check == checked_is_files_deriv


def test_load_all_files_wrong_names(gcms):
    wrong_files_info = gcms.create_files_info()
    wrong_files_info.index = ["Wrong_filename"] + wrong_files_info.index.tolist()[1:]
    gcms.files_info = wrong_files_info
    with pytest.raises(FileNotFoundError):
        gcms.load_all_files()


def test_load_class_code_fractions(gcms, checked_load_class_code_fractions):
    to_check = gcms.load_class_code_frac()
    assert_frame_equal(
        to_check,
        checked_load_class_code_fractions,
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_load_calibrations(
    gcms,
    checked_load_calibrations,
):
    files_info = gcms.create_files_info()
    calib_to_check, is_calib_deriv_to_check = gcms.load_calibrations()
    for to_check, checked in zip(calib_to_check, checked_load_calibrations):
        assert to_check == checked

        for to_check, checked in zip(
            calib_to_check.values(), checked_load_calibrations.values()
        ):
            assert_frame_equal(
                to_check, checked, check_exact=False, atol=1e-5, rtol=1e-5
            )

        assert is_calib_deriv_to_check == False


def test_list_of_all_compounds(gcms, checked_list_of_all_compounds):
    to_check = gcms.create_list_of_all_compounds()
    assert to_check.sort() == checked_list_of_all_compounds.sort()


# def test_list_of_all_deriv_compounds(gcms, checked_list_of_all_deriv_compounds):
#     to_check = gcms.create_list_of_all_deriv_compounds()
#     assert to_check.sort() == checked_list_of_all_deriv_compounds.sort()


@pytest.mark.slow
def test_create_compounds_properties(gcms, checked_compounds_properties):
    to_check = gcms.create_compounds_properties()
    assert_frame_equal(
        to_check.sort_index(),
        checked_compounds_properties.sort_index(),
        check_exact=False,
        check_dtype=False,
        atol=1e-5,
        rtol=1e-5,
    )


# @pytest.mark.slow
# def test_create_deriv_compounds_properties(gcms, checked_deriv_compounds_properties):
#     to_check = gcms.create_deriv_compounds_properties()
#     assert_frame_equal(
#         to_check.sort_index(),
#         checked_deriv_compounds_properties.sort_index(),
#         check_exact=False,
#         atol=1e-3,
#         rtol=1e-3,
#     )


def test_load_compounds_properties(gcms, checked_compounds_properties):
    to_check = gcms.load_compounds_properties()
    assert_frame_equal(
        to_check.sort_index(),
        checked_compounds_properties.sort_index(),
        check_exact=False,
        atol=1e-3,
        rtol=1e-3,
    )


# def test_load_deriv_compounds_properties(gcms, checked_deriv_compounds_properties):
#     to_check = gcms.load_deriv_compounds_properties()
#     assert_frame_equal(
#         to_check.sort_index(),
#         checked_deriv_compounds_properties.sort_index(),
#         check_exact=False,
#         atol=1e-3,
#         rtol=1e-3,
#     )


def test_create_samples_info(gcms, checked_samples_info, checked_samples_info_std):
    to_check, to_check_std = gcms.create_samples_info()
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
def test_files_param_reports(gcms, checked_files_param_reports, parameter):
    to_check = gcms.create_files_param_report(param=parameter)
    checked_report = checked_files_param_reports[parameter]
    assert_frame_equal(
        to_check, checked_report, check_exact=False, atol=1e-5, rtol=1e-5
    )


def test_files_param_reports_exception(gcms):
    with pytest.raises(ValueError):
        gcms.create_files_param_report(param="wrong_parameter")


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
def test_files_param_aggrreps(gcms, checked_files_param_aggrreps, parameter):
    to_check = gcms.create_files_param_aggrrep(param=parameter)
    checked_report = checked_files_param_aggrreps[parameter]
    assert_frame_equal(
        to_check, checked_report, check_exact=False, atol=1e-5, rtol=1e-5
    )


def test_files_param_aggreps_exception(gcms):
    with pytest.raises(ValueError):
        gcms.create_files_param_aggrrep(param="wrong_parameter")


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
    gcms, checked_samples_param_reports, checked_samples_param_reports_std, parameter
):
    to_check, to_check_std = gcms.create_samples_param_report(param=parameter)
    checked_report = checked_samples_param_reports[parameter]
    checked_report_std = checked_samples_param_reports_std[parameter]
    assert_frame_equal(
        to_check, checked_report, check_exact=False, atol=1e-5, rtol=1e-5
    )
    assert_frame_equal(
        to_check_std, checked_report_std, check_exact=False, atol=1e-5, rtol=1e-5
    )


def test_samples_param_reports_exception(gcms):
    with pytest.raises(ValueError):
        gcms.create_samples_param_report(param="wrong_parameter")


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
    gcms, checked_samples_param_aggrreps, checked_samples_param_aggrreps_std, parameter
):
    to_check, to_check_std = gcms.create_samples_param_aggrrep(param=parameter)
    checked_report = checked_samples_param_aggrreps[parameter]
    checked_report_std = checked_samples_param_aggrreps_std[parameter]
    assert_frame_equal(
        to_check, checked_report, check_exact=False, atol=1e-5, rtol=1e-5
    )
    assert_frame_equal(
        to_check_std, checked_report_std, check_exact=False, atol=1e-5, rtol=1e-5
    )


def test_samples_param_aggrreps_exception(gcms):
    with pytest.raises(ValueError):
        gcms.create_samples_param_aggrrep(param="wrong_parameter")
