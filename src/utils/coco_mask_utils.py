from rasterio.crs import CRS
import pandas as pd
import geopandas as gpd
import shapely
from affine import Affine
import os
import numpy as np
import pyproj
import logging
from shapely.geometry import box, Polygon, MultiLineString, MultiPolygon, mapping, shape
from shapely.geometry.base import BaseGeometry
import rasterio
import json
import skimage
from rasterio.warp import transform_bounds
from rasterio import features
from tqdm.auto import tqdm
from looseversion import LooseVersion
import imagecodecs


def _check_do_transform(df, reference_im, affine_obj):
    """Check whether or not a transformation should be performed."""
    try:
        crs = getattr(df, 'crs')
    except AttributeError:
        return False  # if it doesn't have a CRS attribute

    if not crs:
        return False  # return False for do_transform if crs is falsey
    elif crs and (reference_im is not None or affine_obj is not None):
        # if the input has a CRS and another obj was provided for xforming
        return True


def _check_crs(input_crs, return_rasterio=False):
    """Convert CRS to the ``pyproj.CRS`` object passed by ``solaris``."""
    if not isinstance(input_crs, pyproj.CRS) and input_crs is not None:
        out_crs = pyproj.CRS(input_crs)
    else:
        out_crs = input_crs

    if return_rasterio:
        if LooseVersion(rasterio.__gdal_version__) >= LooseVersion("3.0.0"):
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt())
        else:
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt("WKT1_GDAL"))

    return out_crs


def _check_gdf_load(gdf):
    """Check if `gdf` is already loaded in, if not, load from geojson."""
    if isinstance(gdf, str):
        # as of geopandas 0.6.2, using the OGR CSV driver requires some add'nal
        # kwargs to create a valid geodataframe with a geometry column. see
        # https://github.com/geopandas/geopandas/issues/1234
        if gdf.lower().endswith('csv'):
            return gpd.read_file(gdf, GEOM_POSSIBLE_NAMES="geometry",
                                 KEEP_GEOM_COLUMNS="NO")
        try:
            return gpd.read_file(gdf)
        except Exception:
            print(f"GeoDataFrame couldn't be loaded: either {gdf} isn't a valid"
                 " path or it isn't a valid vector file. Returning an empty"
                 " GeoDataFrame.")
            return gpd.GeoDataFrame()
    elif isinstance(gdf, gpd.GeoDataFrame):
        return gdf
    else:
        raise ValueError(f"{gdf} is not an accepted GeoDataFrame format.")


def _check_rasterio_im_load(im):
    """Check if `im` is already loaded in; if not, load it in."""
    if isinstance(im, str):
        return rasterio.open(im)
    elif isinstance(im, rasterio.DatasetReader):
        return im
    else:
        raise ValueError(
            "{} is not an accepted image format for rasterio.".format(im))


def _get_fname_list(p, recursive=False, extension='.tif'):
    """Get a list of filenames from p, which can be a dir, fname, or list."""
    if isinstance(p, list):
        return p
    elif isinstance(p, str):
        if os.path.isdir(p):
            return get_files_recursively(p, traverse_subdirs=recursive,
                                         extension=extension)
        elif os.path.isfile(p):
            return [p]
        else:
            raise ValueError("If a string is provided, it must be a valid"
                             " path.")
    else:
        raise ValueError("{} is not a string or list.".format(p))


def _get_logging_level(level_int):
    """Convert a logging level integer into a log level."""
    if isinstance(level_int, bool):
        level_int = int(level_int)
    if level_int < 0:
        return logging.CRITICAL + 1  # silence all possible outputs
    elif level_int == 0:
        return logging.WARNING
    elif level_int == 1:
        return logging.INFO
    elif level_int == 2:
        return logging.DEBUG
    elif level_int in [10, 20, 30, 40, 50]:  # if user provides the logger int
        return level_int
    elif isinstance(level_int, int):  # if it's an int but not one of the above
        return level_int
    else:
        raise ValueError(f"logging level set to {level_int}, "
                         "but it must be an integer <= 2.")


def remove_multipolygons(gdf):
    """
    Filters out rows of a geodataframe containing MultiPolygons and GeometryCollections.

    This function is optionally used in geojson2coco. For instance segmentation, where
    objects are composed of single polygons, multi part geometries need to be either removed or
    inspected manually to be resolved as a single geometry.
    """
    mask = (gdf.geom_type == "MultiPolygon") | (gdf.geom_type == "GeometryCollection")
    if mask.any():
        return gdf.drop(gdf[mask].index).reset_index(drop=True)
    else:
        return gdf


def geojson_to_px_gdf(geojson, im_path, geom_col='geometry', precision=None,
                      output_path=None, override_crs=False):
    """Convert a geojson or set of geojsons from geo coords to px coords.

    Arguments
    ---------
    geojson : str
        Path to a geojson. This function will also accept a
        :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` with a
        column named ``'geometry'`` in this argument.
    im_path : str
        Path to a georeferenced image (ie a GeoTIFF) that geolocates to the
        same geography as the `geojson`(s). This function will also accept a
        :class:`osgeo.gdal.Dataset` or :class:`rasterio.DatasetReader` with
        georeferencing information in this argument.
    geom_col : str, optional
        The column containing geometry in `geojson`. If not provided, defaults
        to ``"geometry"``.
    precision : int, optional
        The decimal precision for output geometries. If not provided, the
        vertex locations won't be rounded.
    output_path : str, optional
        Path to save the resulting output to. If not provided, the object
        won't be saved to disk.
    override_crs: bool, optional
        Useful if the geojsons generated by the vector tiler or otherwise were saved
        out with a non EPSG code projection. True sets the gdf crs to that of the
        image, the inputs should have the same underlying projection for this to work.
        If False, and the gdf does not have an EPSG code, this function will fail.
    Returns
    -------
    output_df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` with all geometries in `geojson` that
        overlapped with the image at `im_path` converted to pixel coordinates.
        Additional columns are included with the filename of the source
        geojson (if available) and images for reference.

    """
    # get the bbox and affine transforms for the image
    im = _check_rasterio_im_load(im_path)
    if isinstance(im_path, rasterio.DatasetReader):
        im_path = im_path.name
    # make sure the geo vector data is loaded in as geodataframe(s)
    gdf = _check_gdf_load(geojson)

    if len(gdf):  # if there's at least one geometry
        if override_crs:
            gdf.crs = im.crs
        overlap_gdf = get_overlapping_subset(gdf, im)
    else:
        overlap_gdf = gdf

    affine_obj = im.transform
    transformed_gdf = affine_transform_gdf(overlap_gdf, affine_obj=affine_obj,
                                           inverse=True, precision=precision,
                                           geom_col=geom_col)
    transformed_gdf['image_fname'] = os.path.split(im_path)[1]

    if output_path is not None:
        if output_path.lower().endswith('json'):
            transformed_gdf.to_file(output_path, driver='GeoJSON')
        else:
            transformed_gdf.to_csv(output_path, index=False)
    return transformed_gdf


def get_overlapping_subset(gdf, im=None, bbox=None, bbox_crs=None):
    """Extract a subset of geometries in a GeoDataFrame that overlap with `im`.

    Notes
    -----
    This function uses RTree's spatialindex, which is much faster (but slightly
    less accurate) than direct comparison of each object for overlap.

    Arguments
    ---------
    gdf : :class:`geopandas.GeoDataFrame`
        A :class:`geopandas.GeoDataFrame` instance or a path to a geojson.
    im : :class:`rasterio.DatasetReader` or `str`, optional
        An image object loaded with `rasterio` or a path to a georeferenced
        image (i.e. a GeoTIFF).
    bbox : `list` or :class:`shapely.geometry.Polygon`, optional
        A bounding box (either a :class:`shapely.geometry.Polygon` or a
        ``[bottom, left, top, right]`` `list`) from an image. Has no effect
        if `im` is provided (`bbox` is inferred from the image instead.) If
        `bbox` is passed and `im` is not, a `bbox_crs` should be provided to
        ensure correct geolocation - if it isn't, it will be assumed to have
        the same crs as `gdf`.
    bbox_crs : int, optional
        The coordinate reference system that the bounding box is in as an EPSG
        int. If not provided, it's assumed that the CRS is the same as `im`
        (if provided) or `gdf` (if not).

    Returns
    -------
    output_gdf : :class:`geopandas.GeoDataFrame`
        A :class:`geopandas.GeoDataFrame` with all geometries in `gdf` that
        overlapped with the image at `im`.
        Coordinates are kept in the CRS of `gdf`.

    """
    if im is None and bbox is None:
        raise ValueError('Either `im` or `bbox` must be provided.')
    gdf = _check_gdf_load(gdf)
    sindex = gdf.sindex
    if im is not None:
        im = _check_rasterio_im_load(im)
        # currently, convert CRSs to WKT strings here to accommodate rasterio.
        bbox = transform_bounds(im.crs, _check_crs(gdf.crs, return_rasterio=True),
                                *im.bounds)
        bbox_crs = im.crs
    # use transform_bounds in case the crs is different - no effect if not
    if isinstance(bbox, Polygon):
        bbox = bbox.bounds
    if bbox_crs is None:
        try:
            bbox_crs = _check_crs(gdf.crs, return_rasterio=True)
        except AttributeError:
            raise ValueError('If `im` and `bbox_crs` are not provided, `gdf`'
                             'must provide a coordinate reference system.')
    else:
        bbox_crs = _check_crs(bbox_crs, return_rasterio=True)
    # currently, convert CRSs to WKT strings here to accommodate rasterio.
    bbox = transform_bounds(bbox_crs,
                            _check_crs(gdf.crs, return_rasterio=True),
                            *bbox)
    try:
        intersectors = list(sindex.intersection(bbox))
    except RTreeError:
        intersectors = []

    return gdf.iloc[intersectors, :]


def geojson2coco(image_src, label_src, output_path=None, image_ext='.tif',
                 matching_re=None, category_attribute=None, score_attribute=None,
                 preset_categories=None, include_other=True, info_dict=None,
                 license_dict=None, recursive=False, override_crs=False,
                 explode_all_multipolygons=False, remove_all_multipolygons=False,
                 verbose=0):
    """Generate COCO-formatted labels from one or multiple geojsons and images.

    This function ingests optionally georegistered polygon labels in geojson
    format alongside image(s) and generates .json files per the
    `COCO dataset specification`_ . Some models, like
    many Mask R-CNN implementations, require labels to be in this format. The
    function assumes you're providing image file(s) and geojson file(s) to
    create the dataset. If the number of images and geojsons are both > 1 (e.g.
    with a SpaceNet dataset), you must provide a regex pattern to extract
    matching substrings to match images to label files.

    .. _COCO dataset specification: http://cocodataset.org/

    Arguments
    ---------
    image_src : :class:`str` or :class:`list` or :class:`dict`
        Source image(s) to use in the dataset. This can be::

            1. a string path to an image,
            2. the path to a directory containing a bunch of images,
            3. a list of image paths,
            4. a dictionary corresponding to COCO-formatted image records, or
            5. a string path to a COCO JSON containing image records.

        If a directory, the `recursive` flag will be used to determine whether
        or not to descend into sub-directories.
    label_src : :class:`str` or :class:`list`
        Source labels to use in the dataset. This can be a string path to a
        geojson, the path to a directory containing multiple geojsons, or a
        list of geojson file paths. If a directory, the `recursive` flag will
        determine whether or not to descend into sub-directories.
    output_path : str, optional
        The path to save the JSON-formatted COCO records to. If not provided,
        the records will only be returned as a dict, and not saved to file.
    image_ext : str, optional
        The string to use to identify images when searching directories. Only
        has an effect if `image_src` is a directory path. Defaults to
        ``".tif"``.
    matching_re : str, optional
        A regular expression pattern to match filenames between `image_src`
        and `label_src` if both are directories of multiple files. This has
        no effect if those arguments do not both correspond to directories or
        lists of files. Will raise a ``ValueError`` if multiple files are
        provided for both `image_src` and `label_src` but no `matching_re` is
        provided.
    category_attribute : str, optional
        The name of an attribute in the geojson that specifies which category
        a given instance corresponds to. If not provided, it's assumed that
        only one class of object is present in the dataset, which will be
        termed ``"other"`` in the output json.
    score_attribute : str, optional
        The name of an attribute in the geojson that specifies the prediction
        confidence of a model
    preset_categories : :class:`list` of :class:`dict`s, optional
        A pre-set list of categories to use for labels. These categories should
        be formatted per
        `the COCO category specification`_.
        example:
        [{'id': 1, 'name': 'Fighter Jet', 'supercategory': 'plane'},
        {'id': 2, 'name': 'Military Bomber', 'supercategory': 'plane'}, ... ]
    include_other : bool, optional
        If set to ``True``, and `preset_categories` is provided, objects that
        don't fall into the specified categories will not be removed from the
        dataset. They will instead be passed into a category named ``"other"``
        with its own associated category ``id``. If ``False``, objects whose
        categories don't match a category from `preset_categories` will be
        dropped.
    info_dict : dict, optional
        A dictonary with the following key-value pairs::

            - ``"year"``: :class:`int` year of creation
            - ``"version"``: :class:`str` version of the dataset
            - ``"description"``: :class:`str` string description of the dataset
            - ``"contributor"``: :class:`str` who contributed the dataset
            - ``"url"``: :class:`str` URL where the dataset can be found
            - ``"date_created"``: :class:`datetime.datetime` when the dataset
                was created

    license_dict : dict, optional
        A dictionary containing the licensing information for the dataset, with
        the following key-value pairs::

            - ``"name": :class:`str` the name of the license.
            -  ``"url": :class:`str` a link to the dataset's license.

        *Note*: This implementation assumes that all of the data uses one
        license. If multiple licenses are provided, the image records will not
        be assigned a license ID.
    recursive : bool, optional
        If `image_src` and/or `label_src` are directories, setting this flag
        to ``True`` will induce solaris to descend into subdirectories to find
        files. By default, solaris does not traverse the directory tree.
    explode_all_multipolygons : bool, optional
        Explode the multipolygons into individual geometries using sol.utils.geo.split_multi_geometries.
        Be sure to inspect which geometries are multigeometries, each individual geometries within these
        may represent artifacts rather than true labels.
    remove_all_multipolygons : bool, optional
        Filters MultiPolygons and GeometryCollections out of each tile geodataframe. Alternatively you
        can edit each polygon manually to be a polygon before converting to COCO format.
    verbose : int, optional
        Verbose text output. By default, none is provided; if ``True`` or
        ``1``, information-level outputs are provided; if ``2``, extremely
        verbose text is output.

    Returns
    -------
    coco_dataset : dict
        A dictionary following the `COCO dataset specification`_ . Depending
        on arguments provided, it may or may not include license and info
        metadata.
    """

    # first, convert both image_src and label_src to lists of filenames
    logger = logging.getLogger(__name__)
    logger.setLevel(_get_logging_level(int(verbose)))
    logger.debug('Preparing image filename: image ID dict.')
    # pdb.set_trace()
    if isinstance(image_src, str):
        if image_src.endswith('json'):
            logger.debug('COCO json provided. Extracting fname:id dict.')
            with open(image_src, 'r') as f:
                image_ref = json.load(f)
                image_ref = {image['file_name']: image['id']
                             for image in image_ref['images']}
        else:
            image_list = _get_fname_list(image_src, recursive=recursive,
                                         extension=image_ext)
            image_ref = dict(zip(image_list,
                                 list(range(1, len(image_list) + 1))
                                 ))
    elif isinstance(image_src, dict):
        logger.debug('image COCO dict provided. Extracting fname:id dict.')
        if 'images' in image_src.keys():
            image_ref = image_src['images']
        else:
            image_ref = image_src
        image_ref = {image['file_name']: image['id']
                     for image in image_ref}
    else:
        logger.debug('Non-COCO formatted image set provided. Generating '
                     'image fname:id dict with arbitrary ID integers.')
        image_list = _get_fname_list(image_src, recursive=recursive,
                                     extension=image_ext)
        image_ref = dict(zip(image_list, list(range(1, len(image_list) + 1))))

    logger.debug('Preparing label filename list.')
    label_list = _get_fname_list(label_src, recursive=recursive,
                                 extension='json')

    logger.debug('Checking if images and vector labels must be matched.')
    do_matches = len(image_ref) > 1 and len(label_list) > 1
    if do_matches:
        logger.info('Matching images to label files.')
        im_names = pd.DataFrame({'image_fname': list(image_ref.keys())})
        label_names = pd.DataFrame({'label_fname': label_list})
        logger.debug('Getting substrings for matching from image fnames.')
        if matching_re is not None:
            im_names['match_substr'] = im_names['image_fname'].str.extract(
                matching_re)
            logger.debug('Getting substrings for matching from label fnames.')
            label_names['match_substr'] = label_names[
                'label_fname'].str.extract(matching_re)
        else:
            logger.debug('matching_re is none, getting full filenames '
                         'without extensions for matching.')
            im_names['match_substr'] = im_names['image_fname'].apply(
                lambda x: os.path.splitext(os.path.split(x)[1])[0])
            im_names['match_substr'] = im_names['match_substr'].astype(
                str)
            label_names['match_substr'] = label_names['label_fname'].apply(
                lambda x: os.path.splitext(os.path.split(x)[1])[0])
            label_names['match_substr'] = label_names['match_substr'].astype(
                str)
        match_df = im_names.merge(label_names, on='match_substr', how='inner')

    logger.info('Loading labels.')
    label_df = pd.DataFrame({'label_fname': [],
                             'category_str': [],
                             'geometry': []})
    for gj in tqdm(label_list):
        logger.debug('Reading in {}'.format(gj))
        curr_gdf = gpd.read_file(gj)

        if remove_all_multipolygons is True and explode_all_multipolygons is True:
            raise ValueError("Only one of remove_all_multipolygons or explode_all_multipolygons can be set to True.")
        if remove_all_multipolygons is True and explode_all_multipolygons is False:
            curr_gdf = remove_multipolygons(curr_gdf)
        elif explode_all_multipolygons is True:
            curr_gdf = split_multi_geometries(curr_gdf)

        curr_gdf['label_fname'] = gj
        curr_gdf['image_fname'] = ''
        curr_gdf['image_id'] = np.nan
        if category_attribute is None:
            logger.debug('No category attribute provided. Creating a default '
                         '"other" category.')
            curr_gdf['category_str'] = 'other'  # add arbitrary value
            tmp_category_attribute = 'category_str'
        else:
            tmp_category_attribute = category_attribute
        if do_matches:  # multiple images: multiple labels
            logger.debug('do_matches is True, finding matching image')
            logger.debug('Converting to pixel coordinates.')
            if len(curr_gdf) > 0:  # if there are geoms, reproj to px coords
                curr_gdf = geojson_to_px_gdf(
                    curr_gdf,
                    override_crs=override_crs,
                    im_path=match_df.loc[match_df['label_fname'] == gj,
                                         'image_fname'].values[0])
                curr_gdf['image_id'] = image_ref[match_df.loc[
                    match_df['label_fname'] == gj, 'image_fname'].values[0]]
        # handle case with multiple images, one big geojson
        elif len(image_ref) > 1 and len(label_list) == 1:
            logger.debug('do_matches is False. Many images:1 label detected.')
            raise NotImplementedError('one label file: many images '
                                      'not implemented yet.')
        elif len(image_ref) == 1 and len(label_list) == 1:
            logger.debug('do_matches is False. 1 image:1 label detected.')
            logger.debug('Converting to pixel coordinates.')
            # match the two images
            curr_gdf = geojson_to_px_gdf(curr_gdf,
                                         override_crs=override_crs,
                                         im_path=list(image_ref.keys())[0])
            curr_gdf['image_id'] = list(image_ref.values())[0]
        curr_gdf = curr_gdf.rename(
            columns={tmp_category_attribute: 'category_str'})
        if score_attribute is not None:
            curr_gdf = curr_gdf[['image_id', 'label_fname', 'category_str',
                                 score_attribute, 'geometry']]
        else:
            curr_gdf = curr_gdf[['image_id', 'label_fname', 'category_str',
                                 'geometry']]
        label_df = pd.concat([label_df, curr_gdf], axis='index',
                             ignore_index=True, sort=False)

    logger.info('Finished loading labels.')
    logger.info('Generating COCO-formatted annotations.')
    coco_dataset = df_to_coco_annos(label_df,
                                    geom_col='geometry',
                                    image_id_col='image_id',
                                    category_col='category_str',
                                    score_col=score_attribute,
                                    preset_categories=preset_categories,
                                    include_other=include_other,
                                    verbose=verbose)

    logger.info('Generating COCO-formatted image and license records.')
    if license_dict is not None:
        logger.debug('Getting license ID.')
        if len(license_dict) == 1:
            logger.debug('Only one license present; assuming it applies to '
                         'all images.')
            license_id = 1
        else:
            logger.debug('Zero or multiple licenses present. Not trying to '
                         'match to images.')
            license_id = None
        logger.info('Adding licenses to dataset.')
        coco_licenses = []
        license_idx = 1
        for license_name, license_url in license_dict.items():
            coco_licenses.append({'name': license_name,
                                  'url': license_url,
                                  'id': license_idx})
            license_idx += 1
        coco_dataset['licenses'] = coco_licenses
    else:
        logger.debug('No license information provided, skipping for image '
                     'COCO records.')
        license_id = None
    coco_image_records = make_coco_image_dict(image_ref, license_id)
    coco_dataset['images'] = coco_image_records

    logger.info('Adding any additional information provided as arguments.')
    if info_dict is not None:
        coco_dataset['info'] = info_dict

    if output_path is not None:
        with open(output_path, 'w') as outfile:
            json.dump(coco_dataset, outfile)

    return coco_dataset


def footprint_mask(df, out_file=None, reference_im=None, geom_col='geometry',
                   do_transform=None, affine_obj=None, shape=(900, 900),
                   out_type='int', burn_value=255, burn_field=None):
    """Convert a dataframe of geometries to a pixel mask.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to ``None``, in which case the function
        attempts to infer whether or not a transformation is required based on
        the presence or absence of a CRS in `df`. If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    out_type : 'float' or 'int'
    burn_value : `int` or `float`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`. Ignored if `burn_field` is provided.
    burn_field : str, optional
        Name of a column in `df` that provides values for `burn_value` for each
        independent object. If provided, `burn_value` is ignored.

    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`.

    """
    # start with required checks and pre-population of values
    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')
    df = _check_df_load(df)

    if len(df) == 0 and not out_file:
        return np.zeros(shape=shape, dtype='uint8')

    if do_transform is None:
        # determine whether or not transform should be done
        do_transform = _check_do_transform(df, reference_im, affine_obj)

    df[geom_col] = df[geom_col].apply(_check_geom)  # load in geoms if wkt
    if not do_transform:
        affine_obj = Affine(1, 0, 0, 0, 1, 0)  # identity transform

    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
        shape = reference_im.shape
        if do_transform:
            affine_obj = reference_im.transform

    # extract geometries and pair them with burn values
    if burn_field:
        if out_type == 'int':
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('uint8')))
        else:
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('float32')))
    else:
        feature_list = list(zip(df[geom_col], [burn_value]*len(df)))

    if len(df) > 0:
        output_arr = features.rasterize(shapes=feature_list, out_shape=shape,
                                        transform=affine_obj)
    else:
        output_arr = np.zeros(shape=shape, dtype='uint8')
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        if out_type == 'int':
            meta.update(dtype='uint8')
            meta.update(nodata=0)
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr


def affine_transform_gdf(gdf, affine_obj=None, inverse=False, geom_col="geometry",
                         precision=None):
    """Perform an affine transformation on a GeoDataFrame.

    Arguments
    ---------
    gdf : :class:`geopandas.GeoDataFrame`, :class:`pandas.DataFrame`, or `str`
        A GeoDataFrame, pandas DataFrame with a ``"geometry"`` column (or a
        different column containing geometries, identified by `geom_col` -
        note that this column will be renamed ``"geometry"`` for ease of use
        with geopandas), or the path to a saved file in .geojson or .csv
        format.
    affine_obj : list or :class:`affine.Affine`
        An affine transformation to apply to `geom` in the form of an
        ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
    inverse : bool, optional
        Use this argument to perform the inverse transformation.
    geom_col : str, optional
        The column in `gdf` corresponding to the geometry. Defaults to
        ``'geometry'``.
    precision : int, optional
        Decimal precision to round the geometries to. If not provided, no
        rounding is performed.
    """
    if isinstance(gdf, str):  # assume it's a geojson
        if gdf.lower().endswith('json'):
            gdf = gpd.read_file(gdf)
        elif gdf.lower().endswith('csv'):
            gdf = pd.read_csv(gdf)
        else:
            raise ValueError(
                "The file format is incompatible with this function.")
    if 'geometry' not in gdf.columns:
        gdf = gdf.rename(columns={geom_col: 'geometry'})
    if not isinstance(gdf['geometry'][0], shapely.geometry.polygon.Polygon):
        gdf['geometry'] = gdf['geometry'].apply(shapely.wkt.loads)
    gdf["geometry"] = gdf["geometry"].apply(convert_poly_coords,
                                            affine_obj=affine_obj,
                                            inverse=inverse)
    if precision is not None:
        gdf['geometry'] = gdf['geometry'].apply(
            _reduce_geom_precision, precision=precision)

    # the CRS is no longer valid - remove it
    gdf.crs = None

    return gdf


def convert_poly_coords(geom, raster_src=None, affine_obj=None, inverse=False,
                        precision=None):
    """Georegister geometry objects currently in pixel coords or vice versa.

    Arguments
    ---------
    geom : :class:`shapely.geometry.shape` or str
        A :class:`shapely.geometry.shape`, or WKT string-formatted geometry
        object currently in pixel coordinates.
    raster_src : str, optional
        Path to a raster image with georeferencing data to apply to `geom`.
        Alternatively, an opened :class:`rasterio.Band` object or
        :class:`osgeo.gdal.Dataset` object can be provided. Required if not
        using `affine_obj`.
    affine_obj: list or :class:`affine.Affine`
        An affine transformation to apply to `geom` in the form of an
        ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
        Required if not using `raster_src`.
    inverse : bool, optional
        If true, will perform the inverse affine transformation, going from
        geospatial coordinates to pixel coordinates.
    precision : int, optional
        Decimal precision for the polygon output. If not provided, rounding
        is skipped.

    Returns
    -------
    out_geom
        A geometry in the same format as the input with its coordinate system
        transformed to match the destination object.
    """

    if not raster_src and not affine_obj:
        raise ValueError("Either raster_src or affine_obj must be provided.")

    if raster_src is not None:
        affine_xform = get_geo_transform(raster_src)
    else:
        if isinstance(affine_obj, Affine):
            affine_xform = affine_obj
        else:
            # assume it's a list in either gdal or "standard" order
            # (list_to_affine checks which it is)
            if len(affine_obj) == 9:  # if it's straight from rasterio
                affine_obj = affine_obj[0:6]
            affine_xform = list_to_affine(affine_obj)

    if inverse:  # geo->px transform
        affine_xform = ~affine_xform

    if isinstance(geom, str):
        # get the polygon out of the wkt string
        g = shapely.wkt.loads(geom)
    elif isinstance(geom, shapely.geometry.base.BaseGeometry):
        g = geom
    else:
        raise TypeError('The provided geometry is not an accepted format. '
                        'This function can only accept WKT strings and '
                        'shapely geometries.')

    xformed_g = shapely.affinity.affine_transform(g, [affine_xform.a,
                                                      affine_xform.b,
                                                      affine_xform.d,
                                                      affine_xform.e,
                                                      affine_xform.xoff,
                                                      affine_xform.yoff])
    if isinstance(geom, str):
        # restore to wkt string format
        xformed_g = shapely.wkt.dumps(xformed_g)
    if precision is not None:
        xformed_g = _reduce_geom_precision(xformed_g, precision=precision)

    return xformed_g


def df_to_coco_annos(df, output_path=None, geom_col='geometry',
                     image_id_col=None, category_col=None, score_col=None,
                     preset_categories=None, supercategory_col=None,
                     include_other=True, starting_id=1, verbose=0):
    """Extract COCO-formatted annotations from a pandas ``DataFrame``.

    This function assumes that *annotations are already in pixel coordinates.*
    If this is not the case, you can transform them using
    :func:`solaris.vector.polygon.geojson_to_px_gdf`.

    Note that this function generates annotations formatted per the COCO object
    detection specification. For additional information, see
    `the COCO dataset specification`_.

    .. _the COCO dataset specification: http://cocodataset.org/#format-data

    Arguments
    ---------
    df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` containing geometries to store as annos.
    image_id_col : str, optional
        The column containing image IDs. If not provided, it's assumed that
        all are in the same image, which will be assigned the ID of ``1``.
    geom_col : str, optional
        The name of the column in `df` that contains geometries. The geometries
        should either be shapely :class:`shapely.geometry.Polygon` s or WKT
        strings. Defaults to ``"geometry"``.
    category_col : str, optional
        The name of the column that specifies categories for each object. If
        not provided, all objects will be placed in a single category named
        ``"other"``.
    score_col : str, optional
        The name of the column that specifies the ouptut confidence of a model.
        If not provided, will not be output.
    preset_categories : :class:`list` of :class:`dict`s, optional
        A pre-set list of categories to use for labels. These categories should
        be formatted per
        `the COCO category specification`_.
    starting_id : int, optional
        The number to start numbering annotation IDs at. Defaults to ``1``.
    verbose : int, optional
        Verbose text output. By default, none is provided; if ``True`` or
        ``1``, information-level outputs are provided; if ``2``, extremely
        verbose text is output.

    .. _the COCO category specification: http://cocodataset.org/#format-data

    Returns
    -------
    output_dict : dict
        A dictionary containing COCO-formatted annotation and category entries
        per the `COCO dataset specification`_
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(_get_logging_level(int(verbose)))
    logger.debug('Checking that df is loaded.')
    df = _check_df_load(df)
    temp_df = df.copy()  # for manipulation
    if preset_categories is not None and category_col is None:
        logger.debug('preset_categories has a value, category_col is None.')
        raise ValueError('category_col must be specified if using'
                         ' preset_categories.')
    elif preset_categories is not None and category_col is not None:
        logger.debug('Both preset_categories and category_col have values.')
        logger.debug('Getting list of category names.')
        category_dict = _coco_category_name_id_dict_from_list(
            preset_categories)
        category_names = list(category_dict.keys())
        if not include_other:
            logger.info('Filtering out objects not contained in '
                        ' preset_categories')
            temp_df = temp_df.loc[temp_df[category_col].isin(category_names),
                                  :]
        else:
            logger.info('Setting category to "other" for objects outside of '
                        'preset category list.')
            temp_df.loc[~temp_df[category_col].isin(category_names),
                        category_col] = 'other'
            if 'other' not in category_dict.keys():
                logger.debug('Adding "other" to category_dict.')
                other_id = np.array(list(category_dict.values())).max() + 1
                category_dict['other'] = other_id
                preset_categories.append({'id': other_id,
                                          'name': 'other',
                                          'supercategory': 'other'})
    elif preset_categories is None and category_col is not None:
        logger.debug('No preset_categories, have category_col.')
        logger.info(f'Collecting unique category names from {category_col}.')
        category_names = list(temp_df[category_col].unique())
        logger.info('Generating category ID numbers arbitrarily.')
        category_dict = {k: v for k, v in zip(category_names,
                                              range(1, len(category_names)+1))}
    else:
        logger.debug('No category column or preset categories.')
        logger.info('Setting category to "other" for all objects.')
        category_col = 'category_col'
        temp_df[category_col] = 'other'
        category_names = ['other']
        category_dict = {'other': 1}

    if image_id_col is None:
        temp_df['image_id'] = 1
    else:
        temp_df.rename(columns={image_id_col: 'image_id'})
    logger.debug('Checking geometries.')
    temp_df[geom_col] = temp_df[geom_col].apply(_check_geom)
    logger.info('Getting area of geometries.')
    temp_df['area'] = temp_df[geom_col].apply(lambda x: x.area)
    logger.info('Getting geometry bounding boxes.')
    temp_df['bbox'] = temp_df[geom_col].apply(
        lambda x: bbox_corners_to_coco(x.bounds))
    temp_df['category_id'] = temp_df[category_col].map(category_dict)
    temp_df['annotation_id'] = list(range(starting_id,
                                          starting_id + len(temp_df)))
    if score_col is not None:
        temp_df['score'] = df[score_col]

    def _row_to_coco(row, geom_col, category_id_col, image_id_col, score_col):
        "get a single annotation record from a row of temp_df."
        if score_col is None:

            return {'id': row['annotation_id'],
                    'image_id': int(row[image_id_col]),
                    'category_id': int(row[category_id_col]),
                    'segmentation': [polygon_to_coco(row[geom_col])],
                    'area': row['area'],
                    'bbox': row['bbox'],
                    'iscrowd': 0}
        else:
            return {'id': row['annotation_id'],
                    'image_id': int(row[image_id_col]),
                    'category_id': int(row[category_id_col]),
                    'segmentation': [polygon_to_coco(row[geom_col])],
                    'score': float(row[score_col]),
                    'area': row['area'],
                    'bbox': row['bbox'],
                    'iscrowd': 0}

    coco_annotations = temp_df.apply(_row_to_coco, axis=1, geom_col=geom_col,
                                     category_id_col='category_id',
                                     image_id_col=image_id_col,
                                     score_col=score_col).tolist()
    coco_categories = coco_categories_dict_from_df(
        temp_df, category_id_col='category_id',
        category_name_col=category_col,
        supercategory_col=supercategory_col)

    output_dict = {'annotations': coco_annotations,
                   'categories': coco_categories}

    if output_path is not None:
        with open(output_path, 'w') as outfile:
            json.dump(output_dict, outfile)

    return output_dict


def _check_df_load(df):
    """Check if `df` is already loaded in, if not, load from file."""
    if isinstance(df, str):
        if df.lower().endswith('json'):
            return _check_gdf_load(df)
        else:
            return pd.read_csv(df)
    elif isinstance(df, pd.DataFrame):
        return df
    else:
        raise ValueError(f"{df} is not an accepted DataFrame format.")


def _check_geom(geom):
    """Check if a geometry is loaded in.

    Returns the geometry if it's a shapely geometry object. If it's a wkt
    string or a list of coordinates, convert to a shapely geometry.
    """
    if isinstance(geom, BaseGeometry):
        return geom
    elif isinstance(geom, str):  # assume it's a wkt
        return loads(geom)
    elif isinstance(geom, list) and len(geom) == 2:  # coordinates
        return Point(geom)


def bbox_corners_to_coco(bbox):
    """Convert bbox from ``[minx, miny, maxx, maxy]`` to coco format.

    COCO formats bounding boxes as ``[minx, miny, width, height]``.

    Arguments
    ---------
    bbox : :class:`list`-like of numerics
        A 4-element list of the form ``[minx, miny, maxx, maxy]``.

    Returns
    -------
    coco_bbox : list
        ``[minx, miny, width, height]`` shape.
    """

    return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]


def polygon_to_coco(polygon):
    """Convert a geometry to COCO polygon format."""
    if isinstance(polygon, Polygon):
        coords = polygon.exterior.coords.xy
    elif isinstance(polygon, str):  # assume it's WKT
        coords = loads(polygon).exterior.coords.xy
    elif isinstance(polygon, MultiPolygon):
        raise ValueError("You have MultiPolygon types in your label df. Remove, explode, or fix these to be Polygon geometry types.")
    else:
        raise ValueError('polygon must be a shapely geometry or WKT.')
    # zip together x,y pairs
    coords = list(zip(coords[0], coords[1]))
    coords = [item for coordinate in coords for item in coordinate]

    return coords


def coco_categories_dict_from_df(df, category_id_col, category_name_col,
                                 supercategory_col=None):
    """Extract category IDs, category names, and supercat names from df.

    Arguments
    ---------
    df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` of records to filter for category info.
    category_id_col : str
        The name for the column in `df` that contains category IDs.
    category_name_col : str
        The name for the column in `df` that contains category names.
    supercategory_col : str, optional
        The name for the column in `df` that contains supercategory names,
        if one exists. If not provided, supercategory will be left out of the
        output.

    Returns
    -------
    :class:`list` of :class:`dict` s
        A :class:`list` of :class:`dict` s that contain category records per
        the `COCO dataset specification`_ .
    """
    cols_to_keep = [category_id_col, category_name_col]
    rename_dict = {category_id_col: 'id',
                   category_name_col: 'name'}
    if supercategory_col is not None:
        cols_to_keep.append(supercategory_col)
        rename_dict[supercategory_col] = 'supercategory'
    coco_cat_df = df[cols_to_keep]
    coco_cat_df = coco_cat_df.rename(columns=rename_dict)
    coco_cat_df = coco_cat_df.drop_duplicates()

    return coco_cat_df.to_dict(orient='records')


def make_coco_image_dict(image_ref, license_id=None):
    """Take a dict of ``image_fname: image_id`` pairs and make a coco dict.

    Note that this creates a relatively limited version of the standard
    `COCO image record format`_ record, which only contains the following
    keys::

        * id ``(int)``
        * width ``(int)``
        * height ``(int)``
        * file_name ``(str)``
        * license ``(int)``, optional

    .. _COCO image record format: http://cocodataset.org/#format-data

    Arguments
    ---------
    image_ref : dict
        A dictionary of ``image_fname: image_id`` key-value pairs.
    license_id : int, optional
        The license ID number for the relevant license. If not provided, no
        license information will be included in the output.

    Returns
    -------
    coco_images : list
        A list of COCO-formatted image records ready for export to json.
    """

    image_records = []
    for image_fname, image_id in image_ref.items():
        with rasterio.open(image_fname) as f:
            width = f.width
            height = f.height
        im_record = {'id': image_id,
                     'file_name': os.path.split(image_fname)[1],
                     'width': width,
                     'height': height}
        if license_id is not None:
            im_record['license'] = license_id
        image_records.append(im_record)

    return image_records