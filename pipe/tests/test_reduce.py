import os
from tempfile import TemporaryDirectory
from shutil import copy


def cleanup(tempdir, also_delete):
    """
    Needed to cleanup the TemporaryDirectory on Windows
    """
    fits = [
        'SCI_RAW_Attitude',
        'SCI_RAW_HkExtended',
        'REF_APP_GainCorrection',
        'SCI_CAL_SubArray',
        'SCI_COR_SubArray',
        'RAW_SubArray',
        'PIP_COR_PixelFlagMapSubArray',
        'SCI_COR_Lightcurve-DEFAULT',
        'SCI_COR_Lightcurve-OPTIMAL',
        'SCI_COR_Lightcurve-RINF',
        'SCI_COR_Lightcurve-RSUP',
        'EXT_PRE_StarCatalogue',
        'SCI_RAW_Imagette'
    ] + also_delete

    for file in fits:
        path_to_delete = os.path.join(tempdir, file + ".fits")
        if os.path.exists(path_to_delete):
            os.remove(path_to_delete)


def test_end_to_end():
    """
    Run locally with:
    python -c "from pipe.tests.test_reduce import test_end_to_end as f; f()"
    """
    psflib = os.path.join(
        os.path.dirname(__file__), os.path.pardir,
        'data', 'eigenlib_815_281_70_0.pkl'
    )
    with TemporaryDirectory() as tempdir:
        from .generate_fake_data import generate_fits_files

        generate_fits_files(tempdir, psflib)

        from ..pipe_param import PipeParam
        from ..pipe_control import PipeControl

        copy(
            os.path.join(os.path.dirname(__file__), os.path.pardir, 'data',
                         'CH_TU2020-02-18T06-15-13_REF_APP_GainCorrection_V0109.fits'),
            os.path.join(tempdir, 'CH_TU2020-02-18T06-15-13_REF_APP_GainCorrection_V0109.fits')
        )

        copy(
            os.path.join(os.path.dirname(__file__), os.path.pardir, 'data',
                         'nonlin.txt'),
            os.path.join(tempdir, 'nonlin.txt')
        )

        pps = PipeParam(
            'example', '101', datapath=tempdir, calibpath=tempdir
        )
        pps.darksub = False
        pps.mask_badpix = False
        pps.smear_corr = False
        pps.ccdsize = (200, 200)
        pps.psflib = psflib
        pps.gain = 1.9
        pps.non_lin = False
        pps.save_maskcube = False
        pps.save_resid_cube = False

        pc = PipeControl(pps)

        pc.process_eigen()

        assert pc.pp.mad_im < 5
        assert pc.pp.mad_sa < 1

        also_delete = [
            'CH_TU2020-02-18T06-15-13_REF_APP_GainCorrection_V0109.fits',
            'nonlin.txt'
        ]
        
        cleanup(tempdir, also_delete)