import AFQ.tractography as aft
import AFQ.registration as reg
import AFQ.data as afd

import tempfile
import numpy as np
import nibabel as nib
import boto3
from boto.s3.connection import S3Connection

import os
import os.path as op
import AFQ.registration as reg

import numpy as np

import dipy.reconst.dti as dti
import dipy.reconst.dki as dki
import dipy.core.gradients as dpg
import dipy.reconst.cross_validation as xval


def exists(path, bucket_name):
    paths = []
    conn = S3Connection(profile_name='cirrus')
    bucket = conn.get_bucket(bucket_name)
    for key in bucket.list():
        paths.append(key.name)

    if path in paths:
        return True
    else:
        return False


def setup_boto():
    boto3.setup_default_session(profile_name='hcp')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')
    return bucket


def save_wm_mask(subject):
    s3 = boto3.resource('s3')
    boto3.setup_default_session(profile_name='cirrus')
    bucket = s3.Bucket('hcp-dki')
    path = '%s/%s_white_matter_mask.nii.gz' % (subject, subject)
    if not exists(path, bucket.name):
        bucket = setup_boto()
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                dwi_file = op.join(tempdir, 'data.nii.gz')
                seg_file = op.join(tempdir, 'aparc+aseg.nii.gz')
                data_files = {}
                data_files[dwi_file] = \
                    'HCP_900/%s/T1w/Diffusion/data.nii.gz' % subject
                data_files[seg_file] = \
                    'HCP_900/%s/T1w/aparc+aseg.nii.gz' % subject
                for k in data_files.keys():
                    if not op.exists(k):
                        bucket.download_file(data_files[k], k)

                seg_img = nib.load(seg_file)
                dwi_img = nib.load(dwi_file)
                seg_data_orig = seg_img.get_data()
                # Corpus callosum labels:
                cc_mask = ((seg_data_orig == 251) |
                           (seg_data_orig == 252) |
                           (seg_data_orig == 253) |
                           (seg_data_orig == 254) |
                           (seg_data_orig == 255))

                # Cerebral white matter in both hemispheres + corpus callosum
                wm_mask = ((seg_data_orig == 41) | (seg_data_orig == 2) |
                           (cc_mask))
                dwi_data = dwi_img.get_data()
                resamp_wm = np.round(reg.resample(wm_mask, dwi_data[..., 0],
                                     seg_img.affine,
                                     dwi_img.affine)).astype(int)
                wm_file = op.join(tempdir, 'wm.nii.gz')
                nib.save(nib.Nifti1Image(resamp_wm.astype(int),
                                         dwi_img.affine),
                         wm_file)
                boto3.setup_default_session(profile_name='cirrus')
                s3 = boto3.resource('s3')
                s3.meta.client.upload_file(
                        wm_file,
                        'hcp-dki',
                        path)
                return subject, True
            except Exception as err:
                return subject, err.args
    else:
        return subject, True


def compare_models(subject):
    s3 = boto3.resource('s3')
    boto3.setup_default_session(profile_name='cirrus')
    bucket = s3.Bucket('hcp-dki')
    path_dki = '%s/%s_cod_dki.nii.gz' % (subject, subject)
    path_dti = '%s/%s_cod_dti.nii.gz' % (subject, subject)
    if not (exists(path_dki, bucket.name) and exists(path_dti, bucket.name)):
        print("Files don't exist - going ahead")
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                bucket = setup_boto()
                dwi_file = op.join(tempdir, 'data.nii.gz')
                bvec_file = op.join(tempdir, 'data.bvec')
                bval_file = op.join(tempdir, 'data.bval')

                data_files = {}

                data_files[dwi_file] = \
                    'HCP_900/%s/T1w/Diffusion/data.nii.gz' % subject
                data_files[bvec_file] = \
                    'HCP_900/%s/T1w/Diffusion/bvecs' % subject
                data_files[bval_file] = \
                    'HCP_900/%s/T1w/Diffusion/bvals' % subject
                for k in data_files.keys():
                    if not op.exists(k):
                        bucket.download_file(data_files[k], k)

                wm_file = op.join(tempdir, 'wm.nii.gz')
                boto3.setup_default_session(profile_name='cirrus')
                s3 = boto3.resource('s3')
                s3.meta.client.download_file(
                    'hcp-dki',
                    '%s/%s_white_matter_mask.nii.gz' % (subject, subject),
                    wm_file)
                wm_mask = nib.load(wm_file).get_data().astype(bool)
                dwi_img = nib.load(dwi_file)
                data = dwi_img.get_data()
                bvals = np.loadtxt(bval_file)
                bvecs = np.loadtxt(bvec_file)
                gtab = dpg.gradient_table(bvals, bvecs,
                                          b0_threshold=10)
                s3 = boto3.resource('s3')
                boto3.setup_default_session(profile_name='cirrus')
                bucket = s3.Bucket('hcp-dki')
                for model_object, method in zip([dti.TensorModel,
                                                 dki.DiffusionKurtosisModel],
                                                ['dti', 'dki']):
                    path_method = '%s/%s_cod_%s.nii.gz' % (subject,
                                                           subject,
                                                           method)
                    if not (exists(path_method, bucket.name)):
                        print("No %s file - fitting" % method)

                        print("1")
                        model = model_object(gtab)
                        print("2")
                        pred = xval.kfold_xval(model, data, 5, mask=wm_mask)
                        print("3")
                        cod = xval.coeff_of_determination(pred, data)
                        cod_file = op.join(tempdir, 'cod_%s.nii.gz' % method)
                        print("4")
                        nib.save(nib.Nifti1Image(cod, dwi_img.affine),
                                 cod_file)
                        print("5")
                        s3.meta.client.upload_file(
                            cod_file,
                            'hcp-dki',
                            path_method)
                return subject, True
            except Exception as err:
                return subject, err.args
    else:
        return subject, True



def calc_cod_dti1000(subject):
    s3 = boto3.resource('s3')
    boto3.setup_default_session(profile_name='cirrus')
    bucket = s3.Bucket('hcp-dki')
    path_dti = '%s/%s_cod_dti_1000.nii.gz' % (subject, subject)
    if not exists(path_dti, bucket.name):
        print("File doesn't exist - going ahead")
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                bucket = setup_boto()
                dwi_file = op.join(tempdir, 'data.nii.gz')
                bvec_file = op.join(tempdir, 'data.bvec')
                bval_file = op.join(tempdir, 'data.bval')

                data_files = {}

                data_files[dwi_file] = \
                    'HCP_900/%s/T1w/Diffusion/data.nii.gz' % subject
                data_files[bvec_file] = \
                    'HCP_900/%s/T1w/Diffusion/bvecs' % subject
                data_files[bval_file] = \
                    'HCP_900/%s/T1w/Diffusion/bvals' % subject
                for k in data_files.keys():
                    if not op.exists(k):
                        bucket.download_file(data_files[k], k)

                wm_file = op.join(tempdir, 'wm.nii.gz')
                boto3.setup_default_session(profile_name='cirrus')
                s3 = boto3.resource('s3')
                s3.meta.client.download_file(
                    'hcp-dki',
                    '%s/%s_white_matter_mask.nii.gz' % (subject, subject),
                    wm_file)
                wm_mask = nib.load(wm_file).get_data().astype(bool)
                dwi_img = nib.load(dwi_file)
                data = dwi_img.get_data()
                bvals = np.loadtxt(bval_file)
                bvecs = np.loadtxt(bvec_file)

                idx = bvals < 1985
                data = data[..., idx]
                gtab = dpg.gradient_table(bvals[idx], bvecs[:, idx].squeeze(),
                                          b0_threshold=10)

                print("1")
                model = dti.TensorModel(gtab)
                print("2")
                pred = xval.kfold_xval(model, data, 5, mask=wm_mask)
                print("3")
                cod = xval.coeff_of_determination(pred, data)
                cod_file = op.join(tempdir, 'cod_dti_1000.nii.gz')
                print("4")
                nib.save(nib.Nifti1Image(cod, dwi_img.affine),
                         cod_file)
                print("5")
                s3.meta.client.upload_file(
                    cod_file,
                    'hcp-dki',
                    path_dti)

                return subject, True
            except Exception as err:
                return subject, err.args
    else:
        return subject, True
