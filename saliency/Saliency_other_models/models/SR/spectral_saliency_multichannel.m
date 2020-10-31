function S=spectral_saliency_multichannel(I,imsize,multichannel_method,smap_smoothing_filter_params,cmap_smoothing_filter_params,cmap_normalization,do_figures,do_channel_image_mattogrey)
  % SPECTRAL_SALIENCY_MULTICHANNEL provides implementations of several
  %   spectral (FFT,DCT) saliency algorithms for images.
  %
  %   The selected image size (imsize) at which the saliency is calculated 
  %   is the most important parameter. Just try different sizes and you 
  %   will see ...
  %
  %   There are several methods (multichannel_method) to calculate the 
  %   multichannel saliency:
  %   'fft':          by default the same as 'fft:whitening'
  %   'fft:whitening' Uses spectral whitening to calculate the saliency of
  %                   each channel separately and then averages the result.
  %   'fft:residual'  Uses the spectral residual to calculate saliency of 
  %                   each channel separately and then averages the result.
  %   'dct'           Uses DCT-based image signatures to calculate saliency
  %                   of each channel separately and then averages the 
  %                   result.
  %   'quat:fft':     Converts the image into a quaternion-based 
  %                   representation, uses quaternion FFT/IFFT operations.
  %   'quat:dct'      Converts the image into a quaternion-based 
  %                   representation, uses quaternion DCT/IDCT operations.
  %   'quat:dct:fast' Same as 'quad:dct', but with a fixed image 
  %                   resolution of 64x48 and uses optimized .mex files for
  %                   faster calculation.
  %   [...]           some others, e.g., Itti-Koch and GBVS for reference
  % 
  %   Usage examples:
  %   - spectral_saliency_multichannel(imread(..image path...))
  %     or as an example for other color spaces (e.g. ICOPP, Lab, ...)
  %   - spectral_saliency_multichannel(rgb2icopp(imread(..image path...)))
  %
  %   If you use any of this work in scientific research or as part of a 
  %   larger software system, you are requested to cite the use in any 
  %   related publications or technical documentation. The work is based 
  %   upon:
  %
  %       B. Schauerte, and R. Stiefelhagen, "Quaternion DCT Spectral 
  %       Saliency: Predicting Human Gaze using Quaternion DCT Image 
  %       Signatures and Face Detection," in IEEE Workshop on Applications
  %       of Computer Vision (WACV), 2012.
  %
  %   Notes:
  %   - The implementation of the quaternion-based approach requires the
  %     quaternion toolbox (QTFM) for Matlab.
  %   - I kept the implementations as focused and simple as possible and
  %     thus they lack more advanced functionality, e.g. more complex 
  %     normalizations. However, I think that the provided functionality is
  %     more than sufficient for (a) people who want to get started in the
  %     field of visual attention (especially students), (b) practitioners
  %     who have heard about the spectral approach and want to try it, and
  %     (c) people who just need a fast, reliable, well-established visual 
  %     saliency algorithm (with a simple interface and not too many
  %     parameters) for their applications.
  %   - GBVS and Itti require the original GBVS Matlab implementation by
  %     J. Harel (see http://www.klab.caltech.edu/~harel/share/gbvs.php)
  %
  %   For more details on the method see:
  %   [1] X. Hou and L. Zhang, "Saliency Detection: A Spectral Residual
  %       Approach", in CVPR, 2007.
  %       (original paper)
  %   [2] C. Guo, Q. Ma, and L. Zhang, "Spatio-temporal saliency detection
  %       using phase spectrum of quaternion fourier transform," in CVPR, 
  %       2008.
  %       (extension to quaternions; importance of the residual)
  %   [3] X. Hou, J. Harel, and C. Koch, "Image Signature: Highlighting 
  %       sparse salient regions," in PAMI, 2011.
  %       (uses DCT-based "image signatures")
  %   [4] B. Schauerte, and R. Stiefelhagen, "Quaternion DCT Spectral 
  %       Saliency: Predicting Human Gaze using Quaternion DCT Image 
  %       Signatures and Face Detection," in IEEE Workshop on Applications
  %       of Computer Vision (WACV) / IEEE Winter Vision Meetings, 2012.
  %       (extension to quaternions; spectral saliency and face detection;
  %        evaluation of spectral saliency approaches on eye-tracking data;
  %        achieved the currently best reported results on the CERF/FIFA
  %        eye-tracking data set and Toronto/Bruce-Tsotsos data set)
  %
  %   It has been applied quite a lot during the last years, e.g., see:
  %   [5] B. Schauerte, B. Kuehn, K. Kroschel, R. Stiefelhagen, "Multimodal 
  %       Saliency-based Attention for Object-based Scene Analysis," in 
  %       IROS, 2011.
  %       ("simple" multi-channel and quaternion-based; Isophote-based
  %        saliency map segmentation)
  %   [6] B. Schauerte, J. Richarz, G. A. Fink,"Saliency-based 
  %       Identification and Recognition of Pointed-at Objects," in IROS,
  %       2010.
  %       (uses multi-channel on intensity, blue-yellow/red-green opponent)
  %   [7] B. Schauerte, G. A. Fink, "Focusing Computational Visual 
  %       Attention in Multi-Modal Human-Robot Interaction," in Proc. ICMI,
  %       2010
  %       (extended to a multi-scale and neuron-based approach that allows
  %        to incorporate information about the visual search target)
  %
  %   However, the underlying principle has been addressed long before:
  %   [9] A. Oppenheim and J. Lim, "The importance of phase in signals,"
  %       in Proc. IEEE, vol. 69, pp. 529-541, 1981.
  % 
  % @author: B. Schauerte
  % @date:   2009-2011
  % @url:    http://cvhci.anthropomatik.kit.edu/~bschauer/
  
  % Copyright 2009-2011 B. Schauerte. All rights reserved.
  % 
  % Redistribution and use in source and binary forms, with or without 
  % modification, are permitted provided that the following conditions are 
  % met:
  % 
  %    1. Redistributions of source code must retain the above copyright 
  %       notice, this list of conditions and the following disclaimer.
  % 
  %    2. Redistributions in binary form must reproduce the above copyright 
  %       notice, this list of conditions and the following disclaimer in 
  %       the documentation and/or other materials provided with the 
  %       distribution.
  % 
  % THIS SOFTWARE IS PROVIDED BY B. SCHAUERTE ''AS IS'' AND ANY EXPRESS OR 
  % IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
  % WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
  % DISCLAIMED. IN NO EVENT SHALL B. SCHAUERTE OR CONTRIBUTORS BE LIABLE 
  % FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
  % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
  % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
  % BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
  % WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  % OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
  % ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  % 
  % The views and conclusions contained in the software and documentation
  % are those of the authors and should not be interpreted as representing 
  % official policies, either expressed or implied, of B. Schauerte.

  %if nargin<2, imsize=1; end % use original image size
  %if nargin<2, imsize=[48 64]; end
  if nargin<2, imsize=[64 64]; end
  if nargin<3, multichannel_method='quat:dct:fast'; end
  %if nargin<4, smap_smoothing_filter_params={'disk',3}; end
  if nargin<4, smap_smoothing_filter_params={'gaussian',9,2.5}; end % {'gaussian',11,2.5}
  if nargin<4, cmap_smoothing_filter_params={}; end
  if nargin<5, cmap_normalization=0; end
  if nargin<6, do_figures=false; end
  if nargin<7, do_channel_image_mattogrey=false; end
  
  do_force_double_image_type=true;
  
  if ~isfloat(I) && do_force_double_image_type
    I=im2double(I);
  end
  imorigsize=size(I);
  IR=imresize(I, imsize, 'bicubic'); % @note: the resizing method has an influence on the results, take care!
  
  nchannels=size(IR,3);
  channel_saliency=zeros(size(IR));
  channel_saliency_smoothed=zeros(size(IR));
  
  switch multichannel_method
    % "simple" single-channel and averaging
    case {'fft','fft:whitening','fft:residual'}
      channel_phase=zeros(size(IR));
      channel_magnitude=zeros(size(IR));
      
      residual_filter_length=0;   % don't use the residual (whitening)
      if strcmp(multichannel_method,'fft:residual')
        residual_filter_length=3; % use the spectral residual (default value)
      end
      
      % calculate "saliency" for each channel
      for i=1:1:nchannels
        % calculate the channel ssaliency / conspicuity map
        [channel_saliency(:,:,i),channel_phase(:,:,i),channel_magnitude(:,:,i)]=spectral_saliency(IR(:,:,i),residual_filter_length);
        
        % filter the conspicuity maps
        if ~isempty(cmap_smoothing_filter_params)
          channel_saliency_smoothed(:,:,i)=imfilter(channel_saliency(:,:,i), fspecial(cmap_smoothing_filter_params{:}));
        else
          channel_saliency_smoothed(:,:,i)=channel_saliency(:,:,i);
        end
        
        % normalize the conspicuity maps
        switch cmap_normalization % simple (range) normalization
          case {1}
            % simply normalize the value range
            cmin=min(channel_saliency_smoothed(:));
            cmax=max(channel_saliency_smoothed(:));
            if (cmin - cmax) > 0
              channel_saliency_smoothed=(channel_saliency_smoothed - cmin) / (cmax - cmin);
            end

          case {0}
            % do nothing
            
          otherwise
            error('unsupported normalization')
        end
      end
          
      % uniform linear combination of the channels
      S=mean(channel_saliency_smoothed,3);
      
      % filter the saliency map
      if ~isempty(smap_smoothing_filter_params)
        S=imfilter(S, fspecial(smap_smoothing_filter_params{:}));
      end
          
      if do_figures
        figure('name','Saliency / Channel');
        for i=1:1:nchannels
          subplot(4,nchannels,0*nchannels+i);
          if do_channel_image_mattogrey
            subimage(mat2gray(IR(:,:,i))); 
          else
            subimage(IR(:,:,i));
          end
          title(['Channel ' int2str(i)]);
        end
        for i=1:1:nchannels
          subplot(4,nchannels,1*nchannels+i);
          subimage(mat2gray(channel_saliency_smoothed(:,:,i))); 
          title(['Channel ' int2str(i) ' Saliency']);
        end
        for i=1:1:nchannels
          subplot(4,nchannels,2*nchannels+i);
          subimage(mat2gray(channel_phase(:,:,i))); 
          title(['Channel ' int2str(i) ' Phase']);
        end
        for i=1:1:nchannels
          subplot(4,nchannels,3*nchannels+i);
          subimage(mat2gray(channel_magnitude(:,:,i))); 
          title(['Channel ' int2str(i) ' Magnitude']);
        end
      end

    % "simple" single-channel and averaging
    case {'dct'}
      channel_isi=zeros(size(IR));
      channel_di=zeros(size(IR));
      
      % calculate "saliency" for each channel
      for i=1:1:nchannels
        % calculate the channel ssaliency / conspicuity map
        [channel_saliency(:,:,i),channel_isi(:,:,i),channel_di(:,:,i)]=spectral_dct_saliency(IR(:,:,i));
        
        % filter the conspicuity maps
        if ~isempty(cmap_smoothing_filter_params)
          channel_saliency_smoothed(:,:,i)=imfilter(channel_saliency(:,:,i), fspecial(cmap_smoothing_filter_params{:})); % @note: smooth each channel vs. smooth the aggregated/summed map
        else
          channel_saliency_smoothed(:,:,i)=channel_saliency(:,:,i);
        end
        
        % normalize the conspicuity maps
        switch cmap_normalization % simple (range) normalization
          case {1}
            % simply normalize the value range
            cmin=min(channel_saliency_smoothed(:));
            cmax=max(channel_saliency_smoothed(:));
            if (cmin - cmax) > 0
              channel_saliency_smoothed=(channel_saliency_smoothed - cmin) / (cmax - cmin);
            end

          case {0}
            % do nothing
            
          otherwise
            error('unsupported normalization')
        end
      end
          
      % uniform linear combination of the channels
      S=mean(channel_saliency_smoothed,3);
      
      % filter the saliency map
      if ~isempty(smap_smoothing_filter_params)
        S=imfilter(S, fspecial(smap_smoothing_filter_params{:}));
      end
          
      if do_figures
        figure('name','Saliency / Channel');
        for i=1:1:nchannels
          subplot(4,nchannels,0*nchannels+i);
          if do_channel_image_mattogrey
            subimage(mat2gray(IR(:,:,i))); 
          else
            subimage(IR(:,:,i));
          end
          title(['Channel ' int2str(i)]);
        end
        for i=1:1:nchannels
          subplot(4,nchannels,1*nchannels+i);
          subimage(mat2gray(channel_saliency_smoothed(:,:,i))); 
          title(['Channel ' int2str(i) ' Saliency']);
        end
        for i=1:1:nchannels
          subplot(4,nchannels,2*nchannels+i);
          subimage(mat2gray(channel_isi(:,:,i))); 
          title(['Channel ' int2str(i) ' Image Signature']);
        end
        for i=1:1:nchannels
          subplot(4,nchannels,3*nchannels+i);
          subimage(mat2gray(channel_di(:,:,i))); 
          title(['Channel ' int2str(i) ' DCT']);
        end
      end
  
    % quaternion-based spectral whitening
    case {'quat:fft','quaternion:fft'}
      [S,FQIR,IFQIR]=spectral_saliency_quaternion(IR);
      if ~isempty(smap_smoothing_filter_params)
        S=imfilter(S, fspecial(smap_smoothing_filter_params{:}));
      end
      
      if do_figures
        visualize_quaternion_image(FQIR)
        visualize_quaternion_image(IFQIR)
      end

    % quaternion-based DCT image signatures
    case {'quat:dct','quaternion:dct'}
      %tic;
      [S,DCTIR,IDCTIR]=spectral_dct_saliency_quaternion(IR);
      %toc
      
      if ~isempty(smap_smoothing_filter_params)
        S=imfilter(S, fspecial(smap_smoothing_filter_params{:}));
      end

      if do_figures
        visualize_quaternion_image(sign(DCTIR));
        visualize_quaternion_image(IDCTIR);
      end
      
    % quaternion-based DCT image signatures (highly-optimized implementation)
    case {'quat:dct:fast','quaternion:dct:fast'}
      assert(size(IR,1) == 48 && size(IR,2) == 64);
      
      if ~exist('qdct_saliency_48_64','file')
        addpath(genpath('qdct_impl')); % add the path to the implementation
        if ~exist('qdct_saliency_48_64','file')
          fprintf('Can not find qdct_saliency_48_64 .mex-file. Trying to compile/build.\n');
          run('qdct_impl/build.m'); % compile/build the hard-coded interface
        end
        addpath(genpath('qdct_impl')); % add the path to the implementation
        % check for success
        if ~exist('qdct_saliency_48_64','file')
          error('Can not find/build qdct_saliency_48_64');
        end
      end
      
      %tic
      S=qdct_saliency_48_64(IR);
      %toc
      
      if ~isempty(smap_smoothing_filter_params)
        S=imfilter(S, fspecial(smap_smoothing_filter_params{:}));
      end
      
    %
    % REFERENCE IMPLEMENTATIONS FOR COMPARISON
    % 
    
    % classic Itti-Koch saliency map (comes with J. Harel's GBVS package)
    case {'itti'}
      gbvs_params=makeGBVSParams;
      gbvs_params.useIttiKochInsteadOfGBVS=true;
      gbvs_params.blurfrac=0.000; % we do the blurring
      S=getfield(gbvs(I,gbvs_params),'master_map');
      %S=getfield(ittikochmap(I),'master_map');
      S=imresize(S,imsize);
      if ~isempty(smap_smoothing_filter_params)
        S=imfilter(S, fspecial(smap_smoothing_filter_params{:}));
      end
      
    % Graph-based Visual Saliency (GBVS)
    case {'gbvs'}
      % @note: always make sure to use the correct color space with GBVS (i.e., rgb)
      gbvs_params=makeGBVSParams; % create default GBVS parameters
      gbvs_params.salmapmaxsize = 50; % you can get better results with higher values (e.g., 60), but it gets really slow!
      gbvs_params.channels      = 'DIOR';
      gbvs_params.blurfrac      = 0.000; % we do the blurring afterwards
      gbvs_params.levels        = [2 3 4]; 
      %gbvs_params.cyclic_type   = 1; % important to set this when calculating/evaluating center-bias corrected AUCs
      %gbvs_params.activationType=2;
      %fprintf('Internal GBVS Parameters: gbvs_params=%s\n',struct2str(gbvs_params));
      S=getfield(gbvs(I,gbvs_params),'master_map');
      S=imresize(S,imsize);
      if ~isempty(smap_smoothing_filter_params)
        S=imfilter(S, fspecial(smap_smoothing_filter_params{:}));
      end
      
    %
    % DUMMY IMPLEMENTATIONS
    % 
    case {'ones'}
      S=double(ones(imsize));
      
    case {'zeros'}
      S=double(ones(imsize));
      
    case {'random'}
      S=rand(imsize);
      
    %
    % FAIL
    % 
    otherwise
      error('unsupported multichannel saliency calculation mode')
  end
  
  if do_figures
    figure('name',['Saliency (' multichannel_method ')']);
    subplot(1,2,1); imshow(I);
    subplot(1,2,2); imshow(mat2gray(imresize(S, imorigsize(1:2))));
  end
