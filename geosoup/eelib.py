import ee
import math


class EEFunc(object):

    @staticmethod
    def expand_image_meta(img_meta):
        """
        Function to expand the metadata associated with an ee.Image object
        :param img_meta: Retrieved ee.Image metadata dictionary using getInfo() method
        :return: String
        """
        if type(img_meta) != dict:
            if type(img_meta).__name__ == 'Image':
                img_meta = img_meta.getInfo()
            else:
                raise RuntimeError('Unsupported EE object')

        out_str = ''
        for k, y in img_meta.items():
            if k == 'bands':
                for _y in y:
                    out_str += 'Band: {} : {}\n'.format(_y['id'], str(_y))
            elif k == 'properties':
                for _k, _y in y.items():
                    out_str += 'Property: {} : {}\n'.format(_k, str(_y))
            else:
                out_str += '{} : {}\n'.format(str(k), str(y))
        return out_str

    @staticmethod
    def expand_feature_meta(feat_meta):
        """
        Function to expand the metadata associated with an ee.Feature object
        :param feat_meta: Retrieved ee.Feature metadata dictionary using getInfo() method
        :return: String
        """
        if type(feat_meta) != dict:
            if type(feat_meta).__name__ == 'Feature':
                feat_meta = feat_meta.getInfo()
            else:
                raise RuntimeError('Unsupported EE object')

        out_str = ''
        for k, y in feat_meta.items():
            if k == 'geometry':
                for _k, _y in y.items():
                    out_str += '{}: {}\n'.format(str(_k), str(_y))

            elif k == 'properties':
                for _k, _y in y.items():
                    out_str += 'Property: {} : {}\n'.format(_k, str(_y))
            else:
                out_str += '{} : {}\n'.format(str(k), str(y))
        return out_str

    @staticmethod
    def expand_feature_coll_meta(feat_coll_meta):
        """
        Function to expand the metadata associated with an ee.FeatureCollection object
        :param feat_coll_meta: Retrieved ee.FeatureCollection metadata dictionary using getInfo() method
        :return: String
        """
        if type(feat_coll_meta) != dict:
            if type(feat_coll_meta).__name__ == 'FeatureCollection':
                feat_coll_meta = feat_coll_meta.getInfo()
            else:
                raise RuntimeError('Unsupported EE object')

        out_str = '---------------------\n'
        for k, y in feat_coll_meta.items():
            if k == 'features':
                for feat in y:
                    out_str += EEFunc.expand_feature_meta(feat) + '---------------------\n'

            elif k == 'properties':
                for _k, _y in y.items():
                    out_str += 'Property: {} : {}\n'.format(_k, str(_y))
            else:
                out_str += '{} : {}\n'.format(str(k), str(y))
        return out_str

    @staticmethod
    def ndvi_calc(img, scale_factor=10000):
        """ Normalized difference vegetation index"""
        return img.normalizedDifference(['NIR', 'RED']).select([0], ['NDVI']).multiply(scale_factor).toInt16()

    @staticmethod
    def vari_calc(img, scale_factor=10000):
        """ Visible Atmospherically Resistant Index"""
        return (img.select(['RED']).subtract(img.select(['GREEN'])))\
            .divide(img.select(['RED']).add(img.select(['GREEN'])).subtract(img.select(['BLUE'])))\
            .select([0], ['VARI']).multiply(scale_factor).toInt16()

    @staticmethod
    def ndwi_calc(img, scale_factor=10000):
        """ Normalized difference wetness index"""
        return img.normalizedDifference(['NIR', 'SWIR2']).select([0], ['NDWI']).multiply(scale_factor).toInt16()

    @staticmethod
    def nbr_calc(img, scale_factor=10000):
        """ Normalized burn ratio"""
        return img.normalizedDifference(['NIR', 'SWIR1']).select([0], ['NBR']).multiply(scale_factor).toInt16()

    @staticmethod
    def savi_calc(img, const=0.5, scale_factor=10000):
        """ Soil adjusted vegetation index"""
        return (img.select(['NIR']).subtract(img.select(['RED'])).multiply(1 + const))\
            .divide(img.select(['NIR']).add(img.select(['RED'])).add(const))\
            .select([0], ['SAVI']).multiply(scale_factor).toInt16()

    @staticmethod
    def add_indices(in_image, const=0.5, scale_factor=10000):
        """ Function to add indices to an image:  NDVI, NDWI, VARI, NBR, SAVI"""
        temp_image = in_image.float().divide(scale_factor)
        return in_image.addBands(EEFunc.ndvi_calc(temp_image, scale_factor))\
            .addBands(EEFunc.ndwi_calc(temp_image, scale_factor))\
            .addBands(EEFunc.vari_calc(temp_image, scale_factor))\
            .addBands(EEFunc.nbr_calc(temp_image, scale_factor))\
            .addBands(EEFunc.savi_calc(temp_image, const, scale_factor))

    @staticmethod
    def add_suffix(in_image, suffix_str):
        """ Add suffix to all band names"""
        bandnames = in_image.bandNames().map(lambda elem: ee.String(elem).toLowerCase().cat('_').cat(suffix_str))
        nb = bandnames.length()
        return in_image.select(ee.List.sequence(0, ee.Number(nb).subtract(1)), bandnames)

    @staticmethod
    def ls8_sr_corr(img):
        """ Method to correct Landsat 8 based on Landsat 7 reflectance.
            This method scales the SR reflectance values to match LS7 reflectance
            The returned values are generally lower than input image
            based on roy et al 2016"""
        return img.select(['B2'], ['BLUE']).float().multiply(0.8850).add(183).int16()\
            .addBands(img.select(['B3'], ['GREEN']).float().multiply(0.9317).add(123).int16())\
            .addBands(img.select(['B4'], ['RED']).float().multiply(0.9372).add(123).int16())\
            .addBands(img.select(['B5'], ['NIR']).float().multiply(0.8339).add(448).int16())\
            .addBands(img.select(['B6'], ['SWIR1']).float().multiply(0.8639).add(306).int16())\
            .addBands(img.select(['B7'], ['SWIR2']).float().multiply(0.9165).add(116).int16())\
            .addBands(img.select(['pixel_qa'], ['PIXEL_QA']).int16())\
            .addBands(img.select(['radsat_qa'], ['RADSAT_QA']).int16())\
            .copyProperties(img)\
            .copyProperties(img, ['system:time_start', 'system:time_end', 'system:index', 'system:footprint'])

    @staticmethod
    def ls5_sr_corr(img):
        """ Method to correct Landsat 5 based on Landsat 7 reflectance.
            This method scales the SR reflectance values to match LS7 reflectance
            The returned values are generally lower than input image
            based on sulla-menashe et al 2016"""
        return img.select(['B1'], ['BLUE']).float().multiply(0.91996).add(37).int16()\
            .addBands(img.select(['B2'], ['GREEN']).float().multiply(0.92764).add(84).int16())\
            .addBands(img.select(['B3'], ['RED']).float().multiply(0.8881).add(98).int16())\
            .addBands(img.select(['B4'], ['NIR']).float().multiply(0.95057).add(38).int16())\
            .addBands(img.select(['B5'], ['SWIR1']).float().multiply(0.96525).add(29).int16())\
            .addBands(img.select(['B7'], ['SWIR2']).float().multiply(0.99601).add(20).int16())\
            .addBands(img.select(['pixel_qa'], ['PIXEL_QA']).int16())\
            .addBands(img.select(['radsat_qa'], ['RADSAT_QA']).int16())\
            .copyProperties(img)\
            .copyProperties(img, ['system:time_start', 'system:time_end', 'system:index', 'system:footprint'])

    @staticmethod
    def ls_sr_band_correction(img):
        """ This method renames LS5, LS7, and LS8 bands and corrects LS5 and LS8 bands
            this method should be used with SR only"""
        return \
            ee.Algorithms.If(
                ee.String(img.get('SATELLITE')).compareTo('LANDSAT_8'),
                ee.Algorithms.If(
                    ee.String(img.get('SATELLITE')).compareTo('LANDSAT_5'),
                    ee.Image(img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa', 'radsat_qa'],
                                        ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'PIXEL_QA', 'RADSAT_QA'])
                             .int16()
                             .copyProperties(img)
                             .copyProperties(img,
                                             ['system:time_start',
                                              'system:time_end',
                                              'system:index',
                                              'system:footprint'])),
                    ee.Image(EEFunc.ls5_sr_corr(img))
                ),
                ee.Image(EEFunc.ls8_sr_corr(img))
            )

    @staticmethod
    def ls_sr_only_clear(image):
        """ Method to calcluate clear mask based on pixel_qa and radsat_qa bands"""
        clearbit = 1
        clearmask = math.pow(2, clearbit)
        qa = image.select('PIXEL_QA')
        qa_mask = qa.bitwiseAnd(clearmask)

        ra = image.select('RADSAT_QA')
        ra_mask = ra.eq(0)

        return ee.Image(image.updateMask(qa_mask).updateMask(ra_mask))

    @staticmethod
    def maxval_comp_ndvi(collection, pctl=50, index='NDVI'):
        """ function to make pctl th value composite"""
        index_band = collection.select(index).reduce(ee.Reducer.percentile([pctl]))
        with_dist = collection.map(lambda image: image.addBands(image.select(index)
                                                                 .subtract(index_band).abs().multiply(-1)
                                                                 .rename('quality')))
        return with_dist.qualityMosaic('quality')

    @staticmethod
    def interval_mean(collection, min_pctl, max_pctl, internal_bands):
        """ function to make interval mean composite"""
        temp_img = collection.reduce(ee.Reducer.intervalMean(min_pctl, max_pctl))
        return temp_img.select(ee.List.sequence(0, internal_bands.length().subtract(1)), internal_bands)