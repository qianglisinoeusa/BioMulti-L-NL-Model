function  EyeParametersInfo()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%THE COD FOR RECORDING HUMAN EYE PARAMETERS INFO
%DEPENDING TOOLBOX Psychtoolbox
%Download IT AND ADD MATLABPATH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%addpath('~/Psychtoolbox')

%Set default values for photoreceptors structure.
photoreceptors = DefaultPhotoreceptors('LivingHumanFovea');


%Human eyes parameters info
photoreceptors = FillInPhotoreceptors(photoreceptors)


%EyeLength:  Return estimate of distance between nodal point and retina.
%eyeLengthMM = EyeLength(species,source)

%Options
%LeGrand (Human, 16.6832 mm, Default)
%Rodieck (Human, 16.1 mm)
%Default Human
eyeLengthMM = EyeLength()
 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%THE COD FOR RECORDING NUMBER IN THE NEUROSCIENCE
%
%MODIFIED FROM  https://alivelearn.net/?p=1416
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print out info
fprintf('--------------------------\n');
fprintf('Number of neurons and synapses \n');
fprintf('Number of neurons: %d \n', 10^11);
fprintf('Number of synapses: %d \n', 10^14);


%Soure Images In Local Dir 
Dir='~/QiangLi/Matlab_Utils_Functional/matlab_human-vision-model_utils/Retina-LGN-V1-models-OnGoing/Images/ParaNeuro/';
cd (Dir)


a=imread('F1.jpg'); figure, imagesc(a);
b=imread('F2.png'); figure, imagesc(b);
c=imread('F3.png'); figure, imagesc(c);
d=imread('F4.png'); figure, imagesc(d);
e=imread('F5.png'); figure, imagesc(e);
f=imread('F6.png'); figure, imagesc(f);

tile

end