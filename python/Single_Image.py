import RCA

FOV = 25
source = "/home/liqiang/CVP/BioMulti-L-NL-Model/imgs/clown.png"

RCA = RCA.RetinalCompression()
dIm = RCA.single(fov=FOV)
dIm = RCA.single(image=source, fov=FOV)
rIm = RCA.single(image=dIm, fov=FOV, decomp=1)