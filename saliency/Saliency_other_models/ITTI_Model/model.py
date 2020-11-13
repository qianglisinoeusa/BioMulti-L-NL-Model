from multiprocessing import Pool
import numpy as np
import skimage

from . import saliency, utils

try:
    import tensorflow as tf
except Exception:
    import warnings
    warnings.warn("Could not import tensorflow. DeepGaze models will not be runnable.")


class IttyKoch:

    """ Python Implementation of the Itty Koch Saliency Model
    """

    def __init__(self,
                 mapwidth = 64,
                 gabor_wavelength = 3.5,
                 n_gabor_angles = 4,
                 center_bias = 1.5,
                 blur_radius = 0.04,
                 gabor_gamma = 1,
                 border_size = 10,
                 surround_sig = [1, 3],
                 logtransform = False,
                 smooting_final = 2,
                 top_down = 'peakiness',
                 n_jobs=1):

        self.__dict__.update(locals())

    def predict(self, img, return_chanmaps = False):
        """ Compute and return the saliency map

        img : The input image. Should have dimensions (W,H,C)
        return_chanmaps : if True, also return the intermediate saliency maps

        Returns: saliency map, [channel maps]
        """
        print("processing")
        batch_mode = (isinstance(img, list))
        if not batch_mode: batch_mode = (img.ndim == 4)
        if batch_mode:
            if self.n_jobs == 1:
                return [self.predict(i,return_chanmaps) for i in img]
            with Pool(self.n_jobs) as p:
                result = p.map(self.predict,img)
            return result
        if img.ndim == 2:
            img = img[:,:,np.newaxis]

        # compute saliency maps
        maps = saliency.collect_maps(saliency.resize(img, self.mapwidth))
        #salmaps = saliency.attenuate_borders(salmaps, self.border_size)

        salmaps = np.stack([saliency.saliency(maps[...,i], self.surround_sig)
                            for i in range(maps.shape[2])], axis=-1)

        if self.top_down == 'peakiness':
            weights = np.array([saliency.peakiness(salmaps[...,i]) for i in range(salmaps.shape[2])])
        else:
            weights = np.array([1.0] + [1/24.]*24 + [1.0]*3)
            weights /= weights.sum()

        final_map = (salmaps*weights).sum(axis=-1)

        if self.smooting_final is not None:
            final_map = skimage.filters.gaussian(final_map,         \
                                         sigma=self.smooting_final, \
                                         truncate=2, mode='reflect')

        if self.center_bias is not None:
            final_map = saliency.center_bias(final_map, length = self.center_bias)

        if self.logtransform:
            final_map = -np.log(1 + 1e-5 - final_map)

        final_map = utils.minmaxnorm(final_map, axis=(0,1))

        # optimize for NSS score by enforcing sharp maxima
        #if self.smooting_final is not None:
    #        final_map = skimage.filters.gaussian(final_map,         \
#                                         sigma=2, \
                                         #truncate=2, mode='reflect')

        if return_chanmaps:
            return final_map, salmaps#, maps
        return final_map



class TensorflowModel:

    def __init__(self,
                 batch_size = 10,
                 check_point = 'DeepGazeII.ckpt'):

        self.__dict__.update(locals())

        tf.reset_default_graph()
        self.new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))

        self.input_tensor = tf.get_collection('input_tensor')[0]
        self.centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
        self.log_density = tf.get_collection('log_density')[0]
        self.log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]

    def predict(self,X, verbose=False):
        """ Compute log probability density
        """
        centerbias_data = np.zeros((1, X.shape[1], X.shape[2], 1))
        log_density_prediction = np.zeros(X.shape[:3] + (1,))

        with tf.Session() as sess:

            self.new_saver.restore(sess, self.check_point)

            for i in range(0, len(X), self.batch_size):

                idc = slice(i, min(i+self.batch_size, len(X)))
                bX = X[idc]

                log_density_prediction[idc] = sess.run(self.log_density, {
                    self.input_tensor: bX,
                    self.centerbias_tensor: centerbias_data,
                })

        return log_density_prediction

class DeepGazeII(TensorflowModel):
    """ Implementation of the Deep Gaze II model

    Adapted from https://deepgaze.bethgelab.org/


    """

    def __init__(self, *args, **kwargs):
        super(self.__class__).__init__(self, *args,
                                       check_point = 'DeepGazeII.ckpt',
                                       **kwargs)

class ICF(TensorflowModel):
    """ Implementation of the Deep Gaze II model

    Adapted from https://deepgaze.bethgelab.org/


    """
    def __init__(self, *args, **kwargs):
        super(self.__class__).__init__(self, *args,
                                       check_point = 'ICF.ckpt',
                                       **kwargs)
