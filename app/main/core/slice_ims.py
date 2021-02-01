import os

import cv2
import skimage.io
import time
import shutil


def split_test_im(im_root_with_ext, testims_dir_tot, results_dir,
                  log_file,
                  slice_sizes=[416],
                  slice_overlap=0.2,
                  test_slice_sep='__',
                  zero_frac_thresh=0.5,
                  ):
    """
    Split files for test step
    Assume input string has no path, but does have extension (e.g:, 'pic.png')

    1. get image path (args.test_image_tmp) from image root name
            (args.test_image_tmp)
    2. slice test image and move to results dir
    """

    # get image root, make sure there is no extension
    im_root = im_root_with_ext.split('.')[0]
    im_path = os.path.join(testims_dir_tot, im_root_with_ext)

    # slice test plot into manageable chunks

    # slice (if needed)
    if slice_sizes[0] > 0:
        # if len(args.slice_sizes) > 0:
        # create test_splitims_locs_file
        # set test_dir as in results_dir
        test_split_dir = os.path.join(results_dir,  im_root + '_split' + '/')
        test_dir_str = '"test_split_dir: ' + test_split_dir + '\n"'
        print("test_dir:", test_dir_str[1:-2])
        os.system('echo ' + test_dir_str + ' >> ' + log_file)
        # print "test_split_dir:", test_split_dir

        # clean out dir, and make anew
        if os.path.exists(test_split_dir):
            if (not test_split_dir.startswith(results_dir)) \
                    or len(test_split_dir) < len(results_dir) \
                    or len(test_split_dir) < 10:
                print("test_split_dir too short!!!!:", test_split_dir)
                return
            shutil.rmtree(test_split_dir, ignore_errors=True)
        os.mkdir(test_split_dir)

        # slice
        for s in slice_sizes:
            slice_im(im_path, im_root,
                              test_split_dir, s, s,
                              zero_frac_thresh=zero_frac_thresh,
                              overlap=slice_overlap,
                              slice_sep=test_slice_sep)
            test_files = [os.path.join(test_split_dir, f) for
                          f in os.listdir(test_split_dir)]
        n_files_str = '"Num files: ' + str(len(test_files)) + '\n"'
        print(n_files_str[1:-2])
        os.system('echo ' + n_files_str + ' >> ' + log_file)

    else:
        test_files = [im_path]
        test_split_dir = os.path.join(results_dir, 'nonsense')

    return test_files, test_split_dir


def slice_im(image_path, out_name, out_dir, sliceHeight=256, sliceWidth=256,
             zero_frac_thresh=0.2, overlap=0.2, slice_sep='|',
             out_ext='.png', verbose=False):

    """
    Slice a large image into smaller windows

    Arguments
    ---------
    image_path : str
        Location of image to slice
    out_name : str
        Root name of output files (coordinates will be appended to this)
    out_dir : str
        Output directory
    sliceHeight : int
        Height of each slice.  Defaults to ``256``.
    sliceWidth : int
        Width of each slice.  Defaults to ``256``.
    zero_frac_thresh : float
        Maximum fraction of window that is allowed to be zeros or null.
        Defaults to ``0.2``.
    overlap : float
        Fractional overlap of each window (e.g. an overlap of 0.2 for a window
        of size 256 yields an overlap of 51 pixels).
        Default to ``0.2``.
    slice_sep : str
        Character used to separate outname from coordinates in the saved
        windows.  Defaults to ``|``
    out_ext : str
        Extension of saved images.  Defaults to ``.png``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``

    Returns
    -------
    None
    """

    use_cv2 = True
    # read in image, cv2 fails on large files
    print("Read in image:", image_path)
    try:
        # convert to rgb (cv2 reads in bgr)
        img_cv2 = cv2.imread(image_path, 1)
        # print ("img_cv2.shape:", img_cv2.shape)
        image0 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    except:
        image0 = skimage.io.imread(
            image_path, as_grey=False).astype(np.uint8)  # [::-1]
        use_cv2 = False
    print("image.shape:", image0.shape)
    # image0 = cv2.imread(image_path, 1)  # color

    if len(out_ext) == 0:
        ext = '.' + image_path.split('.')[-1]
    else:
        ext = out_ext

    win_h, win_w = image0.shape[:2]

    # if slice sizes are large than image, pad the edges
    pad = 0
    if sliceHeight > win_h:
        pad = sliceHeight - win_h
    if sliceWidth > win_w:
        pad = max(pad, sliceWidth - win_w)
    # pad the edge of the image with black pixels
    if pad > 0:
        border_color = (0, 0, 0)
        image0 = cv2.copyMakeBorder(image0, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=border_color)

    win_size = sliceHeight*sliceWidth

    t0 = time.time()
    n_ims = 0
    n_ims_nonull = 0
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    # for y0 in xrange(0, image0.shape[0], dy):#sliceHeight):
    #    for x0 in xrange(0, image0.shape[1], dx):#sliceWidth):
    for y0 in range(0, image0.shape[0], dy):  # sliceHeight):
        for x0 in range(0, image0.shape[1], dx):  # sliceWidth):
            n_ims += 1

            if (n_ims % 50) == 0:
                print(n_ims)

            # make sure we don't have a tiny image on the edge
            if y0+sliceHeight > image0.shape[0]:
                y = image0.shape[0] - sliceHeight
            else:
                y = y0
            if x0+sliceWidth > image0.shape[1]:
                x = image0.shape[1] - sliceWidth
            else:
                x = x0

            # extract image
            window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
            # get black and white image
            if use_cv2:
                window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)
            else:
                window = cv2.cvtColor(window_c, cv2.COLOR_RGB2GRAY)

            # find threshold that's not black
            # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=threshold
            ret, thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
            non_zero_counts = cv2.countNonZero(thresh1)
            zero_counts = win_size - non_zero_counts
            zero_frac = float(zero_counts) / win_size
            # print "zero_frac", zero_fra
            # skip if image is mostly empty
            if zero_frac >= zero_frac_thresh:
                if verbose:
                    print("Zero frac too high at:", zero_frac)
                continue
            # else save
            else:
                # outpath = os.path.join(out_dir, out_name + \
                # '|' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) +\
                # '_' + str(pad) + ext)
                outpath = os.path.join(
                    out_dir,
                    out_name + slice_sep + str(y) + '_' + str(x) + '_'
                    + str(sliceHeight) + '_' + str(sliceWidth)
                    + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                    + ext)

                # outpath = os.path.join(out_dir, 'slice_' + out_name + \
                # '_' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) +\
                # '_' + str(pad) + '.jpg')
                if verbose:
                    print("outpath:", outpath)
                # if large image, convert to bgr prior to saving
                if not use_cv2:
                    skimage.io.imsave(outpath, cv2.cvtColor(window_c, cv2.BGR_RGB))
                else:
                    cv2.imwrite(outpath, cv2.cvtColor(window_c, cv2.cv2.COLOR_BGR2RGB))
                n_ims_nonull += 1

    print("Num slices:", n_ims, "Num non-null slices:", n_ims_nonull,
          "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
    print("Time to slice", image_path, time.time()-t0, "seconds")

    return