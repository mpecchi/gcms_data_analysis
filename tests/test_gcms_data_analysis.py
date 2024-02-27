import pytest
from pandas.testing import assert_frame_equal


def test_load_files_info(gcms, checked_files_info):
    to_check = gcms.load_files_info()
    assert_frame_equal(to_check, checked_files_info,
        check_exact=False, atol=1e-5, rtol=1e-5)


def test_create_files_info(gcms, checked_created_files_info):
    to_check = gcms.create_files_info()
    assert_frame_equal(to_check, checked_created_files_info,
        check_exact=False, atol=1e-5, rtol=1e-5)


def test_load_all_files(gcms, checked_files, checked_is_files_deriv):

    files_to_check, is_deriv_files_to_check = gcms.load_all_files()

    for filename_to_check, checked_filename in zip(files_to_check, checked_files):
        assert filename_to_check == checked_filename

    for file_to_check, checked_file in zip(files_to_check.values(),
                                           checked_files.values()):
        assert_frame_equal(file_to_check, checked_file,
            check_exact=False, atol=1e-5, rtol=1e-5)

    assert is_deriv_files_to_check == checked_is_files_deriv


def test_load_all_files_wrong_names(gcms):
    wrong_files_info = gcms.create_files_info()
    wrong_files_info.index = \
        ['Wrong_filename'] + wrong_files_info.index.tolist()[1:]
    gcms.files_info = wrong_files_info
    with pytest.raises(FileNotFoundError):
        gcms.load_all_files()


def test_load_class_code_fractions(gcms, checked_load_class_code_fractions):
    to_check = gcms.load_class_code_frac()
    assert_frame_equal(to_check, checked_load_class_code_fractions,
        check_exact=False, atol=1e-5, rtol=1e-5)


def test_load_calibrations(gcms, checked_load_calibrations,
                           checked_is_calibrations_deriv):
    files_info = gcms.create_files_info()
    calib_to_check, is_calib_deriv_to_check = gcms.load_calibrations()
    for to_check, checked in zip(calib_to_check, checked_load_calibrations):
        assert to_check == checked

        for to_check, checked in zip(calib_to_check.values(),
                                     checked_load_calibrations.values()):
            assert_frame_equal(to_check, checked,
                check_exact=False, atol=1e-5, rtol=1e-5)

        assert is_calib_deriv_to_check == checked_is_calibrations_deriv

@pytest.mark.parametrize("calibration_name", ['calibration', 'deriv_calibration'])
def test_load_calibrations_only_one_file(gcms, checked_load_calibrations,
                                         checked_is_calibrations_deriv, calibration_name):
    checked_load_calibrations_only_underiv = \
        {k: checked_load_calibrations[k] for k in [calibration_name]}
    checked_is_calibrations_deriv_only_underiv = \
        {k: checked_is_calibrations_deriv[k] for k in [calibration_name]}
    files_info = gcms.create_files_info()
    gcms.files_info = files_info[files_info['calibration_file'] == calibration_name]
    calib_to_check, is_calib_deriv_to_check = gcms.load_calibrations()
    for to_check, checked in zip(calib_to_check, checked_load_calibrations_only_underiv):
        assert to_check == checked

        for to_check, checked in zip(calib_to_check.values(),
                                     checked_load_calibrations_only_underiv.values()):
            assert_frame_equal(to_check, checked,
                check_exact=False, atol=1e-5, rtol=1e-5)

        assert is_calib_deriv_to_check == checked_is_calibrations_deriv_only_underiv

@pytest.mark.parametrize("calibration_name", [False, None])
def test_load_calibrations_false_or_none(gcms, calibration_name):
    _ = gcms.create_files_info()
    gcms.files_info['calibration_file'] = calibration_name
    calib_to_check, is_calib_deriv_to_check = gcms.load_calibrations()
    assert calib_to_check == {}
    assert is_calib_deriv_to_check == {}
    assert gcms.calibrations_loaded is True
    assert gcms.calibrations_not_present is True

def test_list_of_all_compounds(gcms, checked_list_of_all_compounds):
    to_check = gcms.create_list_of_all_compounds()
    assert to_check == checked_list_of_all_compounds

def test_list_of_all_deriv_compounds(gcms, checked_list_of_all_deriv_compounds):
    to_check = gcms.create_list_of_all_deriv_compounds()
    assert to_check == checked_list_of_all_deriv_compounds

@pytest.mark.slow
def test_create_compounds_properties(gcms, checked_compounds_properties):
    to_check = gcms.create_compounds_properties()
    assert_frame_equal(to_check, checked_compounds_properties,
        check_exact=False, atol=1e-5, rtol=1e-5)

@pytest.mark.slow
def test_create_deriv_compounds_properties(gcms, checked_deriv_compounds_properties):
    to_check = gcms.create_deriv_compounds_properties()
    assert_frame_equal(to_check, checked_deriv_compounds_properties,
        check_exact=False, atol=1e-5, rtol=1e-5)

def test_load_compounds_properties(gcms, checked_compounds_properties):
    to_check = gcms.load_compounds_properties()
    assert_frame_equal(to_check, checked_compounds_properties,
        check_exact=False, atol=1e-5, rtol=1e-5)

def test_load_deriv_compounds_properties(gcms, checked_deriv_compounds_properties):
    to_check = gcms.load_deriv_compounds_properties()
    assert_frame_equal(to_check, checked_deriv_compounds_properties,
        check_exact=False, atol=1e-5, rtol=1e-5)

def test_add_iupac_to_calibrations(gcms, checked_calibrations_added_iupac_only_iupac_and_mw,
                                   checked_is_calibrations_deriv):
    calib_to_check, is_calib_deriv_to_check = gcms.add_iupac_to_calibrations()
    for to_check, checked in zip(calib_to_check,
                                 checked_calibrations_added_iupac_only_iupac_and_mw):
        assert to_check == checked

        for to_check, checked in zip(calib_to_check.values(),
                                     checked_calibrations_added_iupac_only_iupac_and_mw.values()):
            assert_frame_equal(to_check.loc[:, ['iupac_name', 'MW']], checked,
                check_exact=False, atol=1e-5, rtol=1e-5)

        assert is_calib_deriv_to_check == checked_is_calibrations_deriv

def test_add_iupac_to_files(gcms, checked_files_added_iupac_only_iupac_and_time,
                            checked_is_files_deriv):
    files_to_check, is_files_deriv_to_check = gcms.add_iupac_to_files()
    for to_check, checked in zip(files_to_check, checked_files_added_iupac_only_iupac_and_time):
        assert to_check == checked

        for to_check, checked in zip(files_to_check.values(),
                                     checked_files_added_iupac_only_iupac_and_time.values()):
            assert_frame_equal(to_check.loc[:, ['iupac_name', 'retention_time']], checked,
                check_exact=False, atol=1e-5, rtol=1e-5)

        assert is_files_deriv_to_check == checked_is_files_deriv

def test_apply_calibration_to_files(gcms, checked_files_applied_calibration,
                                     checked_is_files_deriv):

    files_to_check, is_deriv_files_to_check = gcms.apply_calibration_to_files()

    for filename_to_check, checked_filename in zip(files_to_check, checked_files_applied_calibration):
        assert filename_to_check == checked_filename

    for file_to_check, checked_file in zip(files_to_check.values(),
                                           checked_files_applied_calibration.values()):
        assert_frame_equal(file_to_check, checked_file,
            check_exact=False, atol=1e-5, rtol=1e-5)

    assert is_deriv_files_to_check == checked_is_files_deriv

def test_add_stats_to_files_info(gcms, checked_files_info_added_stats):
    to_check = gcms.add_stats_to_files_info()
    assert_frame_equal(to_check, checked_files_info_added_stats,
        check_exact=False, atol=1e-5, rtol=1e-5)

def test_create_samples_info(gcms, checked_samples_info):
    to_check = gcms.create_samples_info()
    assert_frame_equal(to_check, checked_samples_info,
        check_exact=False, atol=1e-5, rtol=1e-5)

def test_add_stats_to_samples_info_no_calibrations(gcms, checked_samples_info_no_calibrations):
    _ = gcms.create_files_info()
    gcms.files_info['calibration_file'] = False
    gcms.load_calibrations()
    _ = gcms.add_stats_to_files_info()
    to_check = gcms.add_stats_to_samples_info()
    assert_frame_equal(to_check, checked_samples_info_no_calibrations,
        check_exact=False, atol=1e-5, rtol=1e-5)

def test_add_stats_to_samples_info_applied_calibration(gcms, checked_samples_info_applied_calibration):
    _ = gcms.add_stats_to_files_info()
    to_check = gcms.add_stats_to_samples_info()
    assert_frame_equal(to_check, checked_samples_info_applied_calibration,
        check_exact=False, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("parameter", ['height', 'area', 'area_if_undiluted', 'conc_vial_mg_L',
    'conc_vial_if_undiluted_mg_L', 'fraction_of_sample_fr', 'fraction_of_feedstock_fr'])
def test_files_param_reports(gcms, checked_files_param_reports, parameter):
    to_check =  gcms.create_files_param_report(param=parameter)
    checked_report = checked_files_param_reports[parameter]
    assert_frame_equal(to_check, checked_report,
        check_exact=False, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("parameter", ['height', 'area', 'area_if_undiluted', 'conc_vial_mg_L',
    'conc_vial_if_undiluted_mg_L', 'fraction_of_sample_fr', 'fraction_of_feedstock_fr'])
def test_files_param_aggrreps(gcms, checked_files_param_aggrreps, parameter):
    to_check =  gcms.create_files_param_aggrrep(param=parameter)
    checked_report = checked_files_param_aggrreps[parameter]
    assert_frame_equal(to_check, checked_report,
        check_exact=False, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("parameter", ['height', 'area', 'area_if_undiluted', 'conc_vial_mg_L',
    'conc_vial_if_undiluted_mg_L', 'fraction_of_sample_fr', 'fraction_of_feedstock_fr'])
def test_samples_param_reports(gcms, checked_samples_param_reports,
                             checked_samples_param_reports_std, parameter):
    to_check, to_check_std =  gcms.create_samples_param_report(param=parameter)
    checked_report = checked_samples_param_reports[parameter]
    checked_report_std = checked_samples_param_reports_std[parameter]
    assert_frame_equal(to_check, checked_report,
        check_exact=False, atol=1e-5, rtol=1e-5)
    assert_frame_equal(to_check_std, checked_report_std,
        check_exact=False, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("parameter", ['height', 'area', 'area_if_undiluted', 'conc_vial_mg_L',
    'conc_vial_if_undiluted_mg_L', 'fraction_of_sample_fr', 'fraction_of_feedstock_fr'])
def test_samples_param_aggrreps(gcms, checked_samples_param_aggrreps,
                             checked_samples_param_aggrreps_std, parameter):
    to_check, to_check_std =  gcms.create_samples_param_aggrrep(param=parameter)
    checked_report = checked_samples_param_aggrreps[parameter]
    checked_report_std = checked_samples_param_aggrreps_std[parameter]
    assert_frame_equal(to_check, checked_report,
        check_exact=False, atol=1e-5, rtol=1e-5)
    assert_frame_equal(to_check_std, checked_report_std,
        check_exact=False, atol=1e-5, rtol=1e-5)