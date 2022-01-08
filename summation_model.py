import numpy as np
import pyrtools as pyr

def summation_model(dists,ind,expfreq,expspat):

  # SUMMATION_MODEL pools the differences in the wavelet domain according to
  # some Minkowski spatial and frequency summation exponents.
  # A 4 scales orthogonal QMF domain is assumed.
  # 
  # SUMMATON_MODEL applies the pooling in two different ways:
  #
  #         (1) First spatial pooling in each subband and then frequency
  #         pooling (d_ef)
  #
  #         (2) First frequency pooling across scales and orientations and
  #         then spatial pooling (d_fe)
  #
  # These pooling strategies give rise to the same result when using 2-norm
  # exponents.
  #
  # Syntax:
  #
  # [d_ef,d_fe,spatial_dist_map]=summation_model(dis,ind,expfreq,expspat)
  #
  #    Input:
  #        dist = distortions in the wavelet domain in the YUV channels
  #        expfreq = exponent for the frequency pooling 
  #        expspat = exponent for the spatial pooling
  #
  #    Output:
  #        d_ef = distortion pooled across the spatial dimension and then
  #               accross the frequency dimension
  #        d_fe = distortion pooled across the frequency dimension and then
  #               accross the spatial dimension
  #        spatial_dist_map = distortion map in the spatial domain (after
  #                           frequency pooling and before spatial pooling
  #
  #

  ef=expfreq
  es=expspat

  tam=2*ind[1,:]

  p = pt.pyramids.SteerablePyramidSpace(np.zeros((tam[1],tam[2]),4))
  
  distorsion_ef = 0
  distorsion_fe = 0
              
  diso = np.reshape(dists[:],len(dists[:])/3,3)
  dis_c = np.zeros((np.prod(ind[-1,:]),3))
  dis = np.abs([[diso], [dis_c]])
  
  dis_e    = np.power(dis, es) 
  dis_frec = np.zeros((len(ind)-1,3))
    
  for capa in range(1, 4):
    for band in range(1, length(ind)-1+1)
        banda=pyr.pyr_coeffs(dis_e[capa,band])
        dis_frec[capa,band]=np.power(np.sum(np.sum(banda)), (1/es))
      
  distorsion_ef = (np.sum(np.power((np.sum(np.power(dis_frec, ef))), (1/ef)))/np.numel(diso)


  dis_f=np.power(dis, ef)
  mapa_esp=np.zeros((tam[1],tam[2],3))

  for capa in range(1,4):               
    sum_esc=np.zeros((ind[-1,1],ind[-1,2]))                      
    for esc in reversed(range(1,5)): 
      banda=[]
      for ori in range(1,4):
        band=(esc-1)*3+or
        bb=pyr.pyr_coeffs(dis_e[capa,band])+sum_esc/3
        banda[:,:,ori]=extrapola_2d(bb)/4            
      sum_esc=[]
      sum_esc[:,:]=np.sum(banda,3)
    mapa_esp[:,:,capa]=np.power((sum_esc), (1/ef))
  distorsion_fe= (np.sum(np.power(np.sum(np.sum(np.power(mapa_esp, es))), (1/es)))/np.numel(diso)  

  return distorsion_ef, distorsion_fe, mapa_esp
                    