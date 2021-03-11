import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
from pickle import load


def generate_fits_files(tmpdir, psflib):
    np.random.seed(42)

    # Generate fake time axis
    exptime = 60  # sec
    times = np.arange(0, 10 / 60 / 24, 1 / 60 / 24) + 59283
    bjd = 2400000.5 + times

    # Generate fake DRP fluxes
    fluxes = np.ones_like(times)

    # Generate fake subarray images:
    n_pixels = 200
    psf_lib = load(open(psflib, 'rb'))
    x = np.linspace(-100, 99, 200)
    xx, yy = np.mgrid[0:200, 0:200]
    mask = (xx - 100) ** 2 + (yy - 100) ** 2 < 70 ** 2
    sa = mask * psf_lib[0](x, x)
    image_cube = np.repeat(1e6 * sa[None, :, :], len(times), axis=0)
    image_cube += 1e-5 * np.random.randn(*image_cube.shape) + np.ones(
        image_cube.shape)
    unit_vector = np.ones_like(times)

    data = {
        'SCI_RAW_Attitude': {
            'header':
                {
                    'TEXPTIME': exptime,
                    'EXPTIME': exptime,
                    'NAXIS2': len(times),
                },
            'data':
                {
                    'SC_ROLL_ANGLE': np.degrees(
                        np.cos(2 * np.pi * times / 100 / 60 / 24)),
                    'SC_DEC': unit_vector,
                    'SC_RA': unit_vector,
                    'MJD_TIME': times,
                    'BJD_TIME': bjd,
                },
            'ext': 1
        },
        'SCI_RAW_HkExtended': {
            'header':
                {
                    'TEXPTIME': exptime,
                    'EXPTIME': exptime,
                },
            'data':
                {
                    'VOLT_FEE_VOD': unit_vector,
                    'VOLT_FEE_VRD': unit_vector,
                    'VOLT_FEE_VOG': unit_vector,
                    'VOLT_FEE_VSS': unit_vector,
                    'VOLT_FEE_CCD': unit_vector,
                    'thermFront_2': unit_vector
                },
            'ext': 9
        },
        'REF_APP_GainCorrection': {
            'header':
                {
                    'TEXPTIME': exptime,
                    'EXPTIME': exptime,
                    'VOD_OFF': 22,
                    'VRD_OFF': 9,
                    'VOG_OFF': -5.75,
                    'VSS_OFF': 8.8,
                    'TEMP_OFF': -40,
                    'GAIN_NOM': 0.52,
                },
            'data':
                {
                    'FACTOR': unit_vector - 1,
                    'EXP_VOD': unit_vector,
                    'EXP_VRD': unit_vector,
                    'EXP_VOG': unit_vector,
                    'EXP_VSS': unit_vector,
                    'EXP_TEMP': unit_vector
                },
            'ext': 1
        },
        'SCI_CAL_SubArray': {
            'data':
                {
                    'MJD_TIME': times,
                    'BJD_TIME': bjd,
                    'RON': 3.4 * unit_vector,
                    'BIAS': 0 * unit_vector
                },
            'header': {
                'X_WINOFF': 0,
                'Y_WINOFF': 0,
                'V_STRT_M': 0,
                'NEXP': len(times),
                'TEXPTIME': exptime,
                'EXPTIME': exptime,
            },
            'ext': 2
        },
        'SCI_COR_SubArray': {
            'data': image_cube,
            'header': {
                'X_WINOFF': 0,
                'Y_WINOFF': 0,
                'V_STRT_M': 0,
                'NEXP': len(times),
                'TEXPTIME': exptime,
                'EXPTIME': exptime,
            },
            'ext': 1
        },
        'RAW_SubArray': {
            'data':
                {
                    'MJD_TIME': times,
                    'BJD_TIME': bjd
                },
            'header': {
                'NEXP': len(times),
                'TEXPTIME': exptime,
                'EXPTIME': exptime,
            },
            'ext': 2
        },
        'PIP_COR_PixelFlagMapSubArray': {
            'data': image_cube,
            'ext': 1
        },
        'SCI_COR_Lightcurve-DEFAULT': {
            'data': fluxes
        },
        'SCI_COR_Lightcurve-OPTIMAL': {
            'data': fluxes
        },
        'SCI_COR_Lightcurve-RINF': {
            'data': fluxes
        },
        'SCI_COR_Lightcurve-RSUP': {
            'data': fluxes
        },
        'EXT_PRE_StarCatalogue': {
            'data': {
                'T_EFF': [5800],
                'distance': [10],
                'MAG_CHEOPS': [10],
                'RA': [1.0],
                'DEC': [1.0]
            },
        },
        "SCI_RAW_Imagette": {
            'data': image_cube
        },
    }

    default_header = fits.Header({
        'X_WINOFF': n_pixels // 2,
        'Y_WINOFF': n_pixels // 2,
        'V_STRT_M': 0,
        'NEXP': len(times),
        'TEXPTIME': exptime,
        'EXPTIME': exptime,
        'RO_FREQU': 1,
    })

    sci_raw_table = {
        "X_OFF_FULL_ARRAY": 0 * unit_vector,
        "Y_OFF_FULL_ARRAY": 0 * unit_vector,
        "X_OFF_SUB_ARRAY": 0 * unit_vector,
        "Y_OFF_SUB_ARRAY": 0 * unit_vector,
        "MJD_TIME": times,
        "BJD_TIME": bjd
    }

    for filename in data:
        path = os.path.join(tmpdir, filename + '.fits')
        ext = data[filename].get('ext', 1)
        hdu = fits.HDUList(
            [fits.PrimaryHDU(image_cube, header=default_header)] + (ext) * [
                fits.ImageHDU(image_cube)])
        if ext == 1:
            if filename == 'SCI_RAW_Imagette':
                hdu.append(
                    fits.TableHDU(Table(sci_raw_table).to_pandas().to_records()))
            elif filename == 'PIP_COR_PixelFlagMapSubArray':
                hdu.append(fits.ImageHDU(image_cube))
            else:
                hdu.append(fits.TableHDU(Table(
                    {'MJD_TIME': times, 'BJD_TIME': bjd}).to_pandas().to_records()))
        elif ext > 2:
            hdu[1] = fits.TableHDU(
                Table(data['SCI_RAW_HkExtended']['data']).to_pandas().to_records())
            hdu[2] = fits.TableHDU(Table(
                {'MJD_TIME': times, 'BJD_TIME': bjd}).to_pandas().to_records())

        if isinstance(data[filename]['data'], dict):
            table_rec = Table(data[filename]['data']).to_pandas().to_records()
            header = fits.Header(data[filename].get('header', ""))
            hdu[ext] = fits.TableHDU(table_rec, header=header)
        elif isinstance(data[filename]['data'], np.ndarray):
            header = fits.Header(data[filename].get('header', ""))
            hdu[ext] = fits.ImageHDU(data[filename]['data'], header=header)

        if filename == 'RAW_SubArray':
            hdu.extend((9 - len(hdu) + 1) * [
                fits.TableHDU(Table(sci_raw_table).to_pandas().to_records())])
            table_rec = Table(
                data['SCI_RAW_HkExtended']['data']).to_pandas().to_records()
            hdu[9] = fits.TableHDU(table_rec)
        hdu.writeto(path, overwrite=True)

    teffs = [2450., 2500., 2650., 2800., 3070., 3200., 3310., 3370.,
             3420., 3470., 3520., 3580., 3650., 3720., 3790., 3850.,
             4060., 4205., 4350., 4590., 4730., 4900., 5080., 5250.,
             5410., 5520., 5630., 5700., 5770., 5800., 5830., 5860.,
             5945., 6030., 6115., 6200., 6280., 6360., 6440., 6590.,
             6740., 6890., 7050., 7200., 7440., 7500., 7800., 8000.,
             8080., 8270., 8550., 8840., 9200., 9700., 10700., 12500.,
             14000., 14500., 15700., 16700., 17000., 20600., 26000., 31500.,
             32500., 34500., 36500.]

    flats = np.ones((len(teffs), 1024, 1024))

    flat_rec = Table({'T_EFF': teffs,
                      'DATA_TYPE': len(teffs) * ['FLAT FIELD']}
    ).to_pandas(index=False).to_records(
        index_dtypes="<i8",
        column_dtypes={'DATA_TYPE': "S16"}
    )

    flat_path = os.path.join(tmpdir, 'flats.fits')
    hdu = fits.HDUList([fits.PrimaryHDU(np.ones((1024, 1024))),
                        fits.ImageHDU(flats),
                        fits.TableHDU.from_columns(flat_rec)])

    hdu.writeto(flat_path)