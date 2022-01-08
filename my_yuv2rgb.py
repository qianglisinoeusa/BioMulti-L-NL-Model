import numpy as np

def my_yuv2rgb(YUV):

	# MY_YUV2RGB calcula la imagen true-color RGB a partir de la true-color YUV
	# aplicando la matriz:
	#
	#   Myuv_rgb =
	#
	#    1.0000    0.0000    1.3707
	#    1.0000   -0.3365   -0.6982
	#    1.0000    1.7324    0.0000
	#
	# Su inversa MY_RGB2YUV usa la inversa:
	#
	# Mrgb_yuv =
	#
	#    0.2990    0.5870    0.1140
	#   -0.1726   -0.3388    0.5114
	#    0.5114   -0.4282   -0.0832
	#
	# USO: RGB=my_yuv2rgb(YUV);

	Mrgb_yuv =np.array([[0.2990,    0.5870,    0.1140],[-0.1726,   -0.3388,    0.5114],[0.5114,   -0.4282,   -0.0832]])
	print(Mrgb_yuv)
	Myuv_rgb = np.linalg.inv(Mrgb_yuv)
	print(Myuv_rgb)
	
	YUV=np.double(YUV)
	RGB=YUV

	RGB[:,:,0]=Myuv_rgb[0,0]*YUV[:,:,0]+Myuv_rgb[0,1]*YUV[:,:,1]+Myuv_rgb[0,2]*YUV[:,:,2]
	RGB[:,:,1]=Myuv_rgb[1,0]*YUV[:,:,0]+Myuv_rgb[1,1]*YUV[:,:,1]+Myuv_rgb[1,2]*YUV[:,:,2]
	RGB[:,:,2]=Myuv_rgb[2,0]*YUV[:,:,0]+Myuv_rgb[2,1]*YUV[:,:,1]+Myuv_rgb[2,2]*YUV[:,:,2]
	#RGB[:,:,0]-=179.45477266423404
	#RGB[:,:,1]+=135.45870971679688
	#RGB[:,:,2]-=226.8183044444304
	return RGB

