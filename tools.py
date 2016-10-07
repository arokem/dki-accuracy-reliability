import AFQ.tractography as aft
import AFQ.registration as reg
import AFQ.data as afd

import tempfile
import numpy as np
import nibabel as nib
import boto3
import os
import os.path as op
import AFQ.registration as reg

import numpy as np

import dipy.reconst.dti as dti
import dipy.reconst.dki as dki
import dipy.core.gradients as dpg
import dipy.reconst.cross_validation as xval

def setup_boto():
    boto3.setup_default_session(profile_name='hcp')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')
    return bucket


def save_wm_mask(subject):
    bucket = setup_boto()
    with tempfile.TemporaryDirectory() as temp_dir:
        dwi_file = op.join(temp_dir, 'data.nii.gz')
        seg_file = op.join(temp_dir, 'aparc+aseg.nii.gz')
        data_files = {}
        data_files[dwi_file] = \
            'HCP/%s/T1w/Diffusion/data.nii.gz' % subject
        data_files[seg_file] = \
            'HCP/%s/T1w/aparc+aseg.nii.gz' % subject
        for k in data_files.keys():
            if not op.exists(k):
                bucket.download_file(data_files[k], k)

        seg_img = nib.load(seg_file)
        dwi_img = nib.load(dwi_file)
        seg_data_orig = seg_img.get_data()
        # Corpus callosum labels:
        cc_mask = ((seg_data_orig==251) | 
                   (seg_data_orig==252) |
                   (seg_data_orig==253) |
                   (seg_data_orig==254) |
                   (seg_data_orig==255))

        # Cerebral white matter in both hemispheres + corpus callosum
        wm_mask = (seg_data_orig==41) | (seg_data_orig==2) | (cc_mask)
        dwi_data = dwi_img.get_data()
        resamp_wm = np.round(reg.resample(wm_mask, dwi_data[..., 0], seg_img.affine, dwi_img.affine)).astype(int)
        wm_file = op.join(temp_dir, 'wm.nii.gz')
        nib.save(nib.Nifti1Image(resamp_wm.astype(int), dwi_img.affine), wm_file) 
        boto3.setup_default_session(profile_name='cirrus')
        s3 = boto3.resource('s3')
        s3.meta.client.upload_file(wm_file, 
                                   'hcp-dki', 
                                   '%s/%s_white_matter_mask.nii.gz'%(subject, subject))

    return (subject, data_files)


def compare_models(subject):
    bucket = setup_boto()
    with tempfile.TemporaryDirectory() as temp_dir:
        dwi_file = op.join(temp_dir, 'data.nii.gz')
        bvec_file = op.join(temp_dir, 'data.bvec')
        bval_file = op.join(temp_dir, 'data.bval')

        data_files = {}
        
        data_files[dwi_file] = \
            'HCP/%s/T1w/Diffusion/data.nii.gz' % subject
        data_files[bvec_file] = \
            'HCP/%s/T1w/Diffusion/bvecs' % subject
        data_files[bval_file] = \
            'HCP/%s/T1w/Diffusion/bvals' % subject
        for k in data_files.keys():
            if not op.exists(k):
                bucket.download_file(data_files[k], k)

        wm_file = op.join(temp_dir, 'wm.nii.gz')
        boto3.setup_default_session(profile_name='cirrus')
        s3 = boto3.resource('s3')
        s3.meta.client.download_file('hcp-dki', '%s/%s_white_matter_mask.nii.gz'%(subject, subject), wm_file)
        wm_mask = nib.load(wm_file).get_data().astype(bool)
        dwi_img = nib.load(dwi_file)
        data = dwi_img.get_data()
        gtab = dpg.gradient_table(bval_file, bvec_file, b0_threshold=10)
        for model_object, method in zip([dti.TensorModel, dki.DiffusionKurtosisModel],
                                        ['dti', 'dki']):
            
            print("1")
            model = model_object(gtab)
            print("2")
            if method == 'dti':
                pred = xval.kfold_xval(model, data, 5, mask=wm_mask, step=1000000)
            else: 
                pred = xval.kfold_xval(model, data, 5, mask=wm_mask)

            print("3")
            cod = xval.coeff_of_determination(pred, data)
            cod_file = op.join(temp_dir, 'cod_%s.nii.gz'%method)
            print("4")
            nib.save(nib.Nifti1Image(cod, dwi_img.affine), cod_file)
            print("5")
            s3.meta.client.upload_file(cod_file, 
                                       'hcp-dki', 
                                       '%s/%s_cod_%s.nii.gz'%(subject, subject, method))

        
        