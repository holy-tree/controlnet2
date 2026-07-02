# Utils for image quality metrics (PSNR/SSIM)
# Original openset_utils / get_mAP / get_PR functions were for image tagging
# and are not used in ControlNet training. They have been removed to avoid
# dependency on `clip` module which may not be installed.