import numpy as np
from scipy.ndimage import gaussian_laplace
import math
from math import sqrt
from scipy import spatial
from skimage.util import img_as_float
from skimage.feature import peak_local_max


"""This basic blob detection algorithm is based on the blob_log 
function from https://scikit-image.org/"""


def _compute_disk_overlap(d, r1, r2):
    """
    Compute fraction of surface overlap between two disks of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first disk.
    r2 : float
        Radius of the second disk.

    Returns
    -------
    fraction: float
        Fraction of area of the overlap between the two disks.
    """

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = math.acos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 -
            0.5 * sqrt(abs(a * b * c * d)))
    return area / (math.pi * (min(r1, r2) ** 2))


def _compute_sphere_overlap(d, r1, r2):
    """
    Compute volume overlap fraction between two spheres of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first sphere.
    r2 : float
        Radius of the second sphere.

    Returns
    -------
    fraction: float
        Fraction of volume of the overlap between the two spheres.

    Notes
    -----
    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    for more details.
    """
    vol = (math.pi / (12 * d) * (r1 + r2 - d)**2 *
           (d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2))
    return vol / (4./3 * math.pi * min(r1, r2) ** 3)


def _blob_overlap(blob1, blob2, sigma_dim=1):
    """Finds the overlapping area fraction between two blobs.

    Returns a float representing fraction of overlapped area. Note that 0.0
    is *always* returned for dimension greater than 3.

    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    sigma_dim : int, optional
        The dimensionality of the sigma value. Can be 1 or the same as the
        dimensionality of the blob space (2 or 3).

    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).
    """
    ndim = len(blob1) - sigma_dim
    if ndim > 3:
        return 0.0
    root_ndim = sqrt(ndim)

    # we divide coordinates by sigma * sqrt(ndim) to rescale space to isotropy,
    # giving spheres of radius = 1 or < 1.
    if blob1[-1] > blob2[-1]:
        max_sigma = blob1[-sigma_dim:]
        r1 = 1
        r2 = blob2[-1] / blob1[-1]
    else:
        max_sigma = blob2[-sigma_dim:]
        r2 = 1
        r1 = blob1[-1] / blob2[-1]
    pos1 = blob1[:ndim] / (max_sigma * root_ndim)
    pos2 = blob2[:ndim] / (max_sigma * root_ndim)

    d = np.sqrt(np.sum((pos2 - pos1)**2))
    if d > r1 + r2:  # centers farther than sum of radii, so no overlap
        return 0.0

    # one blob is inside the other
    if d <= abs(r1 - r2):
        return 1.0

    if ndim == 2:
        return _compute_disk_overlap(d, r1, r2)

    else:  # ndim=3 http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        return _compute_sphere_overlap(d, r1, r2)


def _prune_blobs(blobs_array, overlap, sigma_dim=1):
    """Eliminated blobs with area overlap.

    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 (or 4) values,
        ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
        where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
        and ``sigma`` is the standard deviation of the Gaussian kernel which
        detected the blob.
        This array must not have a dimension of size 0.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    sigma_dim : int, optional
        The number of columns in ``blobs_array`` corresponding to sigmas rather
        than positions.

    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """
    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if _blob_overlap(blob1, blob2, sigma_dim=sigma_dim) > overlap:
                # note: this test works even in the anisotropic case because
                # all sigmas increase together.
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.array([b for b in blobs_array if b[-1] > 0])


def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=.2,
             overlap=.5, log_scale=False, exclude_border=False):
    r"""Finds blobs in the given grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : 2D or 3D ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        the minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    exclude_border : int or bool, optional
        If nonzero int, `exclude_border` excludes blobs from
        within `exclude_border`-pixels of the border of the image.

    Returns
    -------
    A : (n, image.ndim + sigma) ndarray
        A 2d array with each row representing 2 coordinate values for a 2D
        image, and 3 coordinate values for a 3D image, plus the sigma(s) used.
        When a single sigma is passed, outputs are:
        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
        deviation of the Gaussian kernel which detected the blob. When an
        anisotropic gaussian is used (sigmas per dimension), the detected sigma
        is returned for each dimension.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian

    Examples
    --------
    >>> from skimage import data, feature, exposure
    >>> img = data.coins()
    >>> img = exposure.equalize_hist(img)  # improves detection
    >>> feature.blob_log(img, threshold = .3)
    array([[266.        , 115.        ,  11.88888889],
           [263.        , 302.        ,  17.33333333],
           [263.        , 244.        ,  17.33333333],
           [260.        , 174.        ,  17.33333333],
           [198.        , 155.        ,  11.88888889],
           [198.        , 103.        ,  11.88888889],
           [197.        ,  44.        ,  11.88888889],
           [194.        , 276.        ,  17.33333333],
           [194.        , 213.        ,  17.33333333],
           [185.        , 344.        ,  17.33333333],
           [128.        , 154.        ,  11.88888889],
           [127.        , 102.        ,  11.88888889],
           [126.        , 208.        ,  11.88888889],
           [126.        ,  46.        ,  11.88888889],
           [124.        , 336.        ,  11.88888889],
           [121.        , 272.        ,  17.33333333],
           [113.        , 323.        ,   1.        ]])

    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """
    image = img_as_float(image)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = (
        True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False
    )

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float)

    # Convert sequence types to array
    min_sigma = np.asarray(min_sigma, dtype=float)
    max_sigma = np.asarray(max_sigma, dtype=float)

    if log_scale:
        # for anisotropic data, we use the "highest resolution/variance" axis
        standard_axis = np.argmax(min_sigma)
        start = np.log10(min_sigma[standard_axis])
        stop = np.log10(max_sigma[standard_axis])
        scale = np.logspace(start, stop, num_sigma)[:, np.newaxis]
        sigma_list = scale * min_sigma / np.max(min_sigma)
    else:
        scale = np.linspace(0, 1, num_sigma)[:, np.newaxis]
        sigma_list = scale * (max_sigma - min_sigma) + min_sigma

    print("Calculating Laplacian of Gaussian")
    # computing gaussian laplace
    # average s**2 provides scale invariance
    gl_images = [-gaussian_laplace(image, s) * np.mean(s) ** 2
                 for s in sigma_list]

    image_cube = np.stack(gl_images, axis=-1)
    print("Max log value: %.3f"%np.max(image_cube))
    print("Min log value: %.3f"%np.min(image_cube))

    print("Finding peaks")
    min_threshold = min(threshold)
    local_maxima = peak_local_max(image_cube, threshold_abs=min_threshold,
                                  footprint=np.ones((3,) * (image.ndim + 1)),
                                  threshold_rel=0.0,
                                  exclude_border=exclude_border)
    print("Peaks found")
    
    new_local_maxima = []
    for p in local_maxima:
        log_val = image_cube[p[0], p[1], p[2], p[3]]
        if log_val>=threshold[p[3]]:
            new_local_maxima.append(p)
    local_maxima = np.array(new_local_maxima)
    

    # Catch no peaks
    if local_maxima.size == 0:
        return [], 0, 0

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]
    print("Removing close blobs")

    return lm, overlap, sigma_dim
