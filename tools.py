import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineMap
import ipywidgets as wdg
import IPython.display as display
import matplotlib.pyplot as plt

def resample_volume(moving, static):
    """ 
    Resample a nifti image into the space of another nifti image
    
    Parameters
    ----------
    moving : Nifti1Image
        The 'source' image.
    static : Nifti1Image
        The 'target' image.
        
    Returns
    -------
    resampled_img : Nifti1Image
       The source data in the target space, with the target affine
    """
    affine_map = AffineMap(np.eye(4),
                           static.shape[:3], static.affine, 
                           moving.shape, moving.affine)
    
    resampled = affine_map.transform(moving.get_data())
    return nib.Nifti1Image(resampled, static.get_affine())


def make_widget(data, cmap='bone', dims=4, contours=False):
    """Create an ipython widget for displaying 3D/4D data."""
    def plot_rgb_image(z=data.shape[-2]//2):
        fig, ax = plt.subplots(1)
        im = ax.imshow(data[:, :, z])
        fig.set_size_inches([10, 10])
        plt.show()

    pb_widget = wdg.interactive(plot_rgb_image,
                                    z=wdg.IntSlider(min=0,
                                                    max=data.shape[-2]-1,
                                                    value=data.shape[-2]//2),
                                    b=wdg.IntSlider(min=0,
                                                    max=data.shape[-1]-1,
                                                    value=0))
    display.display(pb_widget)
