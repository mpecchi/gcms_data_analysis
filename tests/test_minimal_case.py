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
    gcms, checked_load_calibrations, checked_is_calibrations_deriv
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

        assert is_calib_deriv_to_check == checked_is_calibrations_deriv
