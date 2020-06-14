function [yy,xim1,J]=stage_L_NL_c(xi,param,Jcomp)

% STAGE_L_NL_c computes the basic linear+nonlinear transform (and derivatives!) in feedforward neural networks.
% It uses a general (not necessarily convolutional) linear transform and a divisive-normalization (DN) nonlinearity.
%
%  [yim1,xim1,J] = stage_L_NL_c(xi,param,computJ)
%
%           Li        Ni
%       xi ---> yim1 ---> xim1
%
% Depending on the parameters this routine may compute each of the transforms mentioned in deep_model_DN_isomorph.m and
% described in parameters_DN_isomorph.m. Possiblities include: 
%       (1) Stage 1: Brightness Divisive Normalization. luminance-to-brightness transform  
%       (2) Stage 2: Contrast Divisive Normalization. brightness-to-contrast transform
%       (3) Stage 3: Generic Divisive Normalization - spatial. CSF + masking in the spatial domain
%       (4) Stage 4: Generic Divisive Normalization - wavelet. Wavelets + masking in the wavelet domain
%
% INPUT:
%  xi = set of N input vectors (given as d*N matrix -each stimulus is a column vector-)
%  param = structure with the parameters of the transform
%       param.general      % GENERIC DIV. NORM. (signed signal) this flag = 1 for generic DN (either in space or in wavelets)
%       param.s_wavelet    % WAVELET DIV. NORM. this flag = 1 for DN in case of wavelet-like sensors
%       param.ns              Number of scales (for wavelet case)
%       param.no              Number of orientations (for wavelet case)
%       param.tw              Transition between scales (convenient default is 1. See matlabpyrtools)
%       param.ind             Index variable of matlabpyrtools
%       param.brightness   % BRIGHTNESS DIV. NORM. (unsigned signal) this flag = 1 for the specific luminance-to-brighness case
%       param.contrast     % CONTRAST DIV. NORM. (unsigned input - signed output) this flag = 1 for the contrast computation DN
%       param.L            % LINEAR TRANSFORM. Forward matrix
%       param.iL               Inverse matrix (not required in the wavelet case)
%       param.Ls               Sigma of the local brightness kernel (only for contrast DN)
%       param.Lc               Amplitude of the local brightness kernel (only for contrast DN)
%       param.scale            Global scaling constrant of the brightness transform 
%       param.g            % NONLINEAR TRANSFORM PARAMETERS. Excitation-inhibition exponent
%       param.b                Semisaturation constant (global scale)
%       param.H                Divisive interaction kernel (matrix)
%       param.Hx               Divisive interaction kernel (matrix). Spatial part (wavelet case). 
%       param.Hsc              Divisive interaction kernel (matrix). Scale part (wavelet case).
%       param.Ho               Divisive interaction kernel (matrix). Orientation part (wavelet case).
%       param.Cff              Divisive interaction kernel (matrix). Relative inter-subband weight (wavelet case).
%       param.autonorm         Normalization of scale based on the current image: if param.autonorm == 1 -> current image, if autonorm == 0 -> it uses an average from natural images = param.e_average 
%       param.e_average        Average amplitude in the subbands of the energy y4^g4 from natural images   
%       param.kappa            Linear gain applied to the normalized subbands
%       param.beta             Weight of the adaptation term in the DN kernel of the brghtness transform
%       param.Hs               Spatial sigma of the gaussian DN kernel. 
%                              Individual sigma (per coefficient, per subband or per scale) can be defined for the 
%                              Gaussian DN kernels in the wavelet case.
%       param.Hss              Scale Sigma of the gaussian DN kernel (in octaves). 
%                              Individual sigma (per coefficient, per subband or per scale) can be defined for the 
%                              Gaussian DN kernels in the wavelet case.
%       param.Hso              Orientation Sigma of the gaussian DN kernel (in octaves). 
%                              Individual sigma (per coefficient, per subband or per scale) can be defined for the 
%                              Gaussian DN kernels in the wavelet case.
%       param.Hc               Amplifude of the gaussian DN kernel.
%                              Individual amplitude (per coefficient, subband or scale) can be defined for the 
%                              Gaussian DN kernels in the wavelet case.
%       param.Hw               Weight of the gaussian DN kernel.
%                              Individual amplitude (per coefficient, subband or scale) can be defined for the 
%                              Gaussian DN kernels in the wavelet case.
%
%  computJ = structure of binary flags to select which Jacobian is computed (seting the corresponding field to 1 -compute- or 0 -don't compute-).
%       computJ.lx   Jacobian of the linear part with regard to x1 (the nonlinear input from the previous layer)
%       computJ.ny   Jacobian of the nonlinear part with regard to y2 (the linear input)
%       computJ.sx   Jacobian of the response (linear+nonlinear) with regard to x1 (the nonlinear input from the previous layer)
%
%       computJ.L    Jacobian of the linear response with regard to the linear matrix
%       computJ.L_compact   Jacobian of the linear response with regard to the linear matrix
%       computJ.Ls   (in the contrast case in which L = (I-H) with Gaussian H) Jacobian of the linear response with regard to the widths of the Gaussians (allows for different widths per sensor, but this can be restricted to single width by imposing the corresponding structure in the Jacobian using the all-ones vector)
%       computJ.Lc   (in the contrast case in which L = (I-H) with Gaussian H) Jacobian of the linear response with regard to the amplitudes of the Gaussians (same amplitude for all)
%       computJ.b    Jacobian of the nonlinear response with regard to the scale of vector b (with fixed profile).
%       computJ.g    Jacobian of the nonlinear response with regard to the parameter g.
%       computJ.H    Jacobian of the nonlinear response with regard to H (full matrix, non-parametric kernel)
%       computJ.H_compact    Jacobian of the nonlinear response with regard to H (full matrix, non-parametric kernel)
%       computJ.Hs   Jacobian of the nonlinear response with regard to the sigmas of the spatial Gaussian kernel H
%                    Only one sigma per subband is allowed. This can be easily generalized by removing the binary matrix "sub_struct" hardcoded in J.Hs = put_diags(M,v)*sub_struct
%       computJ.Hss  Jacobian of the nonlinear response with regard to the sigmas of the scale Gaussian kernel H (only in wavelets)
%                    Only one sigma per subband is allowed. This can be easily generalized by removing the binary matrix "sub_struct" hardcoded in J.Hss = put_diags(M,v)*sub_struct
%       computJ.Hso  Jacobian of the nonlinear response with regard to the sigmas of the orientation Gaussian kernel H (only in wavelets)
%                    Only one sigma per subband is allowed. This can be easily generalized by removing the binary matrix "sub_struct" hardcoded in J.Hso = put_diags(M,v)*sub_struct
%       computJ.Hc   Jacobian of the nonlinear response with regard to the amplitudes of Gaussian kernel H.
%                    Only one amplitude per subband is allowed. This can be easily generalized by removing the binary matrix "sub_struct" hardcoded in J.Hc = put_diags(M,v)*sub_struct
%       computJ.Hw   Jacobian of the nonlinear response with regard to the amplitudes of Gaussian kernel H.
%                    Only one amplitude per subband is allowed. This can be easily generalized by removing the binary matrix "sub_struct" hardcoded in J.Hc = put_diags(M,v)*sub_struct
%       computJ.scale Jacobian of the brightness transform wrt the global scale of the nonlinearity
%       computJ.beta (in the restricted brightness case H = (beta*ones(d,d)/d + I)) Jacobian with regard to the constant beta
%
% OUTPUT:
%  yim1 = set of linear response vectors (given as d*N matrix -each response is a column vector-)
%  xim1 = set of nonlinear response vectors (given as d*N matrix -each response is a column vector-)
%  J    = structure with all the computed Jacobians, with fields:
%       J.lx   Jacobian of the linear part with regard to x1 (the nonlinear input from the previous layer). d*d matrix
%       J.ny   Jacobian of the nonlinear part with regard to y2 (the linear input).
%              This is stored as a (d*N)*d matrix. It is just a convenient way to store the actual block-diagonal matrix
%              (of size (d*N)*(d*N)) stacking the non-interacting image-dependent blocks in a column.
%              The i-th image-dependent Jacobian of size d*d can be accessed using Jim = sacafot(J.ny',d,d,i)';
%       J.sx   Jacobian of the response (linear+nonlinear) with regard to x1 (the nonlinear input from the previous layer)
%              Stored (and accesible) in the same way as J.ny
%       J.L    Jacobian of the linear response with regard to the linear matrix.
%              Huge block diagonal matrices are involved. See the more convenient compact format below instead.
%       J.L_matrix   Jacobian of the linear response with regard to the linear matrix (matrix part).
%                    See blk_diagJ_times_deltaH.m and deltaS_times_blk_diagJ.m
%       J.L_vector   Jacobian of the linear response with regard to the linear matrix (vector part).
%                    See blk_diagJ_times_deltaH.m and deltaS_times_blk_diagJ.m
%       J.Ls   (in the restricted case in which L = (I-H) with Gaussian H)
%              Jacobian of the linear response with regard to the widths of the Gaussians. N*d vector (fixed width is assumed)
%       J.Lc   (in the restricted case in which L = (I-H) with Gaussian H)
%              Jacobian of the linear response with regard to the amplitudes of the Gaussians. N*d vector (fixed amplitude is assumed)
%       J.b   Jacobian of the nonlinear response with regard to the vector b.
%                    It is given as a (d*N)*d matrix (N diagonal matrices stacked in a column).
%                    If certain structure is assumed in the vector (e.g. fixed semisaturation per subband),
%                    the general matrix can be right-multiplied by a binary matrix imposing the desired structure.
%       J.g    Jacobian of the nonlinear response with regard to the parameter g. (N*d)*1 column vector.
%       J.H    Jacobian of the nonlinear response with regard to H (full matrix, non-parametric kernel).
%              Huge d*(d*d) matrix -> not available for multiple images
%       J.H_matrix    Jacobian of the nonlinear response with regard to H (full matrix, non-parametric kernel). Not available for multiple images either.
%                     See blk_diagJ_times_deltaH.m and deltaS_times_blk_diagJ.m
%                     In the wavelet case the derivative has two terms (given the actual nonlinearity and the scaling constant)
%                     Therefore, we have J.H_matrix1 and J.H_matrix2
%       J.H_vector    Jacobian of the nonlinear response with regard to H (full matrix, non-parametric kernel). Not available for multiple images either.
%                     See blk_diagJ_times_deltaH.m and deltaS_times_blk_diagJ.m
%                     In the wavelet case the derivative has two terms (given the actual nonlinearity and the scaling constant)
%                     Therefore, we have J.H_vector1 and J.H_vector2
%       J.Hs   Jacobian of the nonlinear response with regard to the spatial sigmas of the Gaussian kernel H.
%              In the wavelet case one sigma per subband is allowed so the result is given in a matrix (d*N)*n_subbands
%              If a single sigma is assumed, the result reduces to a (d*N)*1 column vector (which is also what you obtain through right-multiplication with an all ones vector)
%       J.Hss   Jacobian of the nonlinear response with regard to the scale sigmas of the Gaussian kernel H.
%              In the wavelet case one sigma per subband is allowed so the result is given in a matrix (d*N)*n_subbands
%              If a single sigma is assumed, the result reduces to a (d*N)*1 column vector (which is also what you obtain through right-multiplication with an all ones vector)
%       J.Hso   Jacobian of the nonlinear response with regard to the orientation sigmas of the Gaussian kernel H.
%              In the wavelet case one sigma per subband is allowed so the result is given in a matrix (d*N)*n_subbands
%              If a single sigma is assumed, the result reduces to a (d*N)*1 column vector (which is also what you obtain through right-multiplication with an all ones vector)
%       J.Hc   Jacobian of the nonlinear response with regard to the amplitudes of Gaussian kernel H. 
%              In the wavelet case one amplitude per coefficient, subband or scale is allowed, so the result is given in matrices 
%              of size (d*N)*d, (d*N)*n_subbands or (d*N)*n_scales.
%              If a single amplitude is assumed, the result reduces to a (d*N)*1 column vector (which is also what you obtain through right-multiplication with an all ones vector)
%       J.Hw   Jacobian of the nonlinear response with regard to the weight vector that right-multiplies the Gaussian kernel H (only in the wavelet case). 
%              In this wavelet case one weight per coefficient, subband or scale is allowed, so the result is given in matrices 
%              of size (d*N)*d, (d*N)*n_subbands or (d*N)*n_scales.
%              If a single weight is assumed, the result reduces to a (d*N)*1 column vector (which is also what you obtain through right-multiplication with an all ones vector)
%       J.beta (in the restricted brightness case H = (beta*ones(d,d)/d + I)) Jacobian with regard to the constant beta
%       J.scale (in the restricted brightness case H = (beta*ones(d,d)/d + I)) Jacobian of the brightness transform wrt the global scale of the nonlinearity
%
% SYNTAX  [yim1,xim1,J] = stage_L_NL_c(xi,param,computJ);
%
% NOTE: stage_L_NL_c.m computes the responses of a single layer of the network,
% therefore only the corresponding layer in the "parameters" and "computJ"
% structures has to be passed to this routine.
%
 

% Added to allow simpler use of Jcomp (passing Jcomp = 0 when no Jacobian is required)
if ~isstruct(Jcomp)
    Jcomp = struct;
    Jcomp.sx = 0;
end
Npatch=size(xi,2); % Number of patches
N=size(xi,1);   % Size of the patch
d=param.d;      % Dim of the stage
H = param.H;

if isfield(param,'channel') == 0
   param.channel = 1;
end

% parameters can have different sizes
SizeB = length(param.b(:,1));
if param.s_wavelet==1 
    [B_struct,b]=structure(param.ind,d,param.b);
    [Hs_struct,~]=structure(param.ind,d,param.Hs);
    [Hss_struct,~]=structure(param.ind,d,param.Hss);
    [Hso_struct,~]=structure(param.ind,d,param.Hso);
    [Hc_struct,c_h]=structure(param.ind,d,param.Hc);
    % c_h = repmat(c_h,[1 Npatch]);
    [Hw_struct,w_h]=structure(param.ind,d,param.Hw);
    %w_h = repmat(w_h,[1 Npatch]);
    [kappa_struct,kappa]=structure(param.ind,d,param.kappa(:,param.channel));
    kappa = repmat(kappa,[1 Npatch]);
    gamma = param.g;
elseif  param.s_wavelet==0 && param.general==1      
    b = param.b(:,param.channel); 
    if  SizeB==1 %  si sigma es escalar
        b=b*ones(d,1);       
    end
    c_h=param.Hc; 
    if  length(c_h)==1 %  si sigma es escalar
        c_h=c_h*ones(d,1);       
    end
    c_h=repmat(c_h,[1 Npatch]);
    gamma = param.g;
else
    b = param.b(:,param.channel); 
    if  SizeB==1 %  si sigma es escalar
        b=b*ones(d,1);       
    end
    if param.contrast == 1
       gamma = param.g; 
    else
       gamma = param.g(param.channel);  
    end 
end

b = repmat(b,[1 Npatch]);
%------------------------------------------------------------------
% GENERAL DIVISIVE NORMALIZATION (separated sign and absolute value)
%------------------------------------------------------------------

if param.general == 1
    % LINEAR -------------------------------------------------------
    if param.s_wavelet == 0
        L = param.L(:,:,param.channel);
        yy = L*xi;         
    else
        L = param.L;
        yy=zeros(d, Npatch);
        for i=1:Npatch
            xaux=xi(:,i);
            yy(:,i) = buildSFpyr(reshape(xaux,[sqrt(N) sqrt(N)]),param.ns,param.no-1,param.tw);
        end
    end
    % NONLINEAR DN -------------------------------------------------------
    y = yy;    
    sy=sign(y);
    y = abs(y);
    e = y.^gamma;
    denom = b + H*e;
    elog=log(y);
    if param.s_wavelet == 0
       x = sy.*e./denom;
       xim1=x;
    else
       if param.autonorm == 1
           e_a = average_deviation_wavelet(e,param.ind);   
           K = kappa.*(b+H*e_a).*(1./e_a);
       else
           e_a = param.e_average;
           e_a = repmat(e_a,[1 Npatch]);
           K = kappa.*(b+H*e_a).*(1./e_a);
       end    
       xp = sy.*e./denom; 
       x = K.*xp;
       %[K x]
       xim1=x;
       % xim1(end)
    end
    
    % Derivative Analytic -------------------------------------
    
    % Derivatives wrt SIGNAL--------------------------------------
    if (isfield(Jcomp,'sx') && Jcomp.sx==1) || (isfield(Jcomp,'L') && Jcomp.L==1) || (isfield(Jcomp,'L_compact') && Jcomp.L_compact==1)
        Jcomp.ny = 1;
    end
    if isfield(Jcomp,'ny') && Jcomp.ny==1        
        % NOTE: this is not using the efficient kroneker-repmat notation to avoid building the diagonals
        Positive = sy(:).*((gamma*y(:).^(gamma-1))./denom(:)).*sy(:);
        a = spalloc(d*Npatch,d,Npatch*d);
        Vaux2=sy(:).*e(:)./denom(:).^2;
        Vaux1=(gamma* y.^(gamma-1)).*sy;
        Negative = repmat(Vaux2,[1 d]).*repmat(H,[ Npatch 1]).*kron(Vaux1,ones(1, d))';
        J.ny = put_diags(a,Positive)-Negative;        
        clear Negative Positive Vaux1 Vaux2 a
        
        if param.s_wavelet == 1
            if param.autonorm == 1
               % NOTE: this is not making full use of the efficient kroneker-repmat notation to avoid building the diagonals 
               K_a = average_deviation_kernel_wavelet(param.ind);
               KD = spalloc(d*Npatch,d,Npatch*d);
               v = kappa.*(1./e_a);
               KD = put_diags(KD,v(:));
               nabla_e_K = KD*H*K_a;  
               v = kappa.*(b+H*e_a).*(1./(e_a.^2));
               KD = put_diags(KD,v(:));
               nabla_e_K = nabla_e_K + KD*K_a;
               clear KD K_a
               nabla_K = nabla_e_K.*kron( gamma*y'.^(gamma-1) , ones(d,1) );
               clear nabla_e_K
               J.ny = repmat(K(:),[1 d]).*(J.ny) + repmat(xp(:),[1 d]).*nabla_K;
            else
               J.ny = repmat(K(:),[1 d]).*(J.ny);
            end
        end
    end
    
    J.lx=0;
    d=length(xi);
    if (isfield(Jcomp,'sx') && Jcomp.sx==1)
        J.sx = J.ny*L;
    end
    if (isfield(Jcomp,'lx') && Jcomp.lx==1)
        J.lx = L;
    end    
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Derivatives wrt PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if param.s_wavelet == 1
       d = length(b(:,1)); 
       if isfield(Jcomp,'kappa') && Jcomp.kappa==1
            aux = sy.*(b+H*e_a).*(1./e_a).*e./denom;
            a = spalloc(Npatch*d,d,Npatch*d);
            J.kappa = put_diags(a,aux(:))*kappa_struct;           
       end    
    end
    %% YA
    
    if isfield(Jcomp,'g') && Jcomp.g==1
       chu = H*(e.*elog); 
       if param.s_wavelet == 1
          if param.autonorm == 1
             chu_a = H*K_a*(e.*elog);  
             chu_b = K_a*(e.*elog);
             denom_a = (b+H*e_a);
             
             J.g = sy(:).*K(:).*(1./denom(:)).*( elog(:) - (1./denom(:)).*chu(:) ).*e(:) + ... 
                   sy(:).*(kappa(:)./denom_a(:)).*( chu_a(:) - (denom_a(:)./e_a(:)).*chu_b(:) ).*e(:)./denom(:);
             clear chu_a chu_b denom_a
          else
             J.g = sy(:).*K(:).*(1./denom(:)).*( elog(:) - (1./denom(:)).*chu(:) ).*e(:);  
          end
       else
          J.g = (sy(:)./denom(:)).*(elog(:) - (1./denom(:)).*(chu(:))).*e(:);   
       end 
       clear chu
    end
    %% YA
        
  % b  with all the possibles size -------------------
    if (isfield(Jcomp,'b') && Jcomp.b==1)
        
        if param.s_wavelet == 0
            aux = -sy(:).*e(:)./(denom(:).^2);
        else
            denom_a = (b + H*e_a);
            aux = ( sy(:).*kappa(:).*e(:)./e_a(:) ) .* (  1./denom(:) -  denom_a(:)./(denom(:).^2) );
        end
        if  SizeB==1 %  si be es escalar
            J.b = aux.*repmat(ones(d,1),[Npatch 1]);       
        elseif SizeB==d %  si be es vector talla d escalar
            a = spalloc(Npatch*d,d,Npatch*d);
            J.b = put_diags(a,aux);       
        else           
            a = spalloc(Npatch*d,d,Npatch*d);
            J.b = put_diags(a,aux)*B_struct;  
        end
        clear aux
    end
    %% YA
        
    if isfield(Jcomp,'H') && Jcomp.H==1
       if param.s_wavelet == 0 
            if Npatch==1
                Bde = block_diagonal(e',d);
                J.H= -diag(sy.*e)*diag(1./denom.^2)*Bde;
                clear Bde
            else
                disp(' Full derivative w.r.t. non-parametric kernel cannot be computed for multiple images (huge matrices)')
            end
       else
            if Npatch==1
                Bde = block_diagonal(e',d);
                J.H = -diag(  kappa.*(b+H*e_a).*(1./e_a).*sy.*e )*diag(1./denom.^2)*Bde;
                clear Bde
                Bdea = block_diagonal(e_a',d);
                J.H = J.H + diag( sy.*kappa.*e./(e_a.*denom) )*Bdea;
                clear Bdea
            else
                disp(' Full derivative w.r.t. non-parametric kernel cannot be computed for multiple images (huge matrices)')
            end
       end
    end
    %% YA
    
    if isfield(Jcomp,'H_compact') && Jcomp.H_compact==1
       if param.s_wavelet == 0 
          % Bde = block_diagonal(e',d);
          % J.H= -diag(sy.*e)*diag(1./denom.^2)*Bde;
          J.H_matrix= -sy.*e./denom.^2;
          J.H_vector= e;
       else
          J.H_matrix1 = -sy.*kappa.*(b+H*e_a).*(1./e_a).*e./denom.^2;
          J.H_vector1 = e;              
          J.H_matrix2 = sy.*e.*kappa./(e_a.*denom);
          J.H_vector2 = e_a;                        
       end
    end
    %% YA
    
    if isfield(Jcomp,'Hs') && Jcomp.Hs==1
        d = length(e(:,1));
        if param.s_wavelet == 0       
            [~,dHds3] = make_2d_gauss_kernel(param.fs,param.N,param.Hs);
            % M = repmat(e',[d 1])*dHds3;
            % J.Hs = -(c_h.*sy.*e./denom.^2).*diag(M);
            % J.Hs = -(c_h(:).*sy(:).*e(:)./denom(:).^2).*diag(M);
            dHds3 = repmat(c_h,[1 d]).*dHds3;
            % M = kron(e',ones(d ,1))*dHds3';
            M = (e'*dHds3')';
            clear dHds3;
            %v = -(c_h(:).*sy(:).*e(:)./denom(:).^2).*get_diags(M);
            %v = -(sy(:).*e(:)./denom(:).^2).*get_diags(M);
            v = -(sy(:).*e(:)./denom(:).^2).*M(:);
            clear M;
            M = spalloc(Npatch*d,d,Npatch*d);
            J.Hs = put_diags(M,v)*ones(d,1);
            clear M
            
        else
            
            % [~,dHxdsx,sub_struct] =kernel_s_wavelet_spatial(param.ind,param.fs,param.Hs)
            G = repmat(c_h,[1 d]).*(param.dHxdsx).*(param.Hsc).*(param.Ho).*(param.Cff).*repmat(w_h',[d 1]);
            %M = kron(e',ones(d ,1))*G';
            %v = -(sy(:).*K(:).*e(:)./denom(:).^2).*get_diags(M);
            M = (e'*G')';
            v = -(sy(:).*K(:).*e(:)./denom(:).^2).*M(:);
            clear M;
            %M = kron(e_a',ones(d ,1))*G';
            M = (e_a'*G')';
            clear G;
            % v = v + (sy(:).*kappa(:).*e(:)./(e_a(:).*denom(:))).*get_diags(M);
            v = v + (sy(:).*kappa(:).*e(:)./(e_a(:).*denom(:))).*M(:);
            clear M;
            M = spalloc(Npatch*d,d,Npatch*d);
            J.Hs = put_diags(M,v)*Hs_struct;
            clear M
            
        end
    end
    %% YA 
    
    if isfield(Jcomp,'Hss') && Jcomp.Hss==1
        d=length(e);
            
            G = repmat(c_h,[1 d]).*(param.dHsdss).*(param.Hx).*(param.Ho).*(param.Cff).*repmat(w_h',[d 1]);
            % M = kron(e',ones(d ,1))*G';
            % v = -(sy(:).*K(:).*e(:)./denom(:).^2).*get_diags(M);
            M = (e'*G')';
            v = -(sy(:).*K(:).*e(:)./denom(:).^2).*M(:);
            clear M;
            %M = kron(e_a',ones(d ,1))*G';
            M = (e_a'*G')';
            clear G;
            %v = v + (sy(:).*kappa(:).*e(:)./(e_a(:).*denom(:))).*get_diags(M);
            v = v + (sy(:).*kappa(:).*e(:)./(e_a(:).*denom(:))).*M(:);
            clear M;
            M = spalloc(Npatch*d,d,Npatch*d);
            J.Hss = put_diags(M,v)*Hss_struct;
            clear M
            
    end
    %% YA
    
    if isfield(Jcomp,'Hso') && Jcomp.Hso==1
        d = length(e);
            
            G = repmat(c_h,[1 d]).*(param.dHodso).*(param.Hx).*(param.Hsc).*(param.Cff).*repmat(w_h',[d 1]);
            %M = kron(e',ones(d ,1))*G';
            %v = -(sy(:).*K(:).*e(:)./denom(:).^2).*get_diags(M);
            M = (e'*G')';
            v = -(sy(:).*K(:).*e(:)./denom(:).^2).*M(:);
            clear M;
            % M = kron(e_a',ones(d ,1))*G';
            % clear G;
            M = (e_a'*G')';
            clear G;
            % v = v + (sy(:).*kappa(:).*e(:)./(e_a(:).*denom(:))).*get_diags(M);
            v = v + (sy(:).*kappa(:).*e(:)./(e_a(:).*denom(:))).*M(:);
            clear M;
            M = spalloc(Npatch*d,d,Npatch*d);
            J.Hso = put_diags(M,v)*Hso_struct;
            clear M
            
    end
    %% YA
    
    if isfield(Jcomp,'Hc') && Jcomp.Hc==1
        if param.s_wavelet == 0
            % J.H=  -diag(sy.*e)*diag(1./denom.^2)*Bde;
            [H3ay,~] = make_2d_gauss_kernel(param.fs,param.N,param.Hs);
            % M = kron(e',ones(d,1))*H3ay;
            % J.Hc = -(sy(:).*e(:)./denom(:).^2).*get_diags(M);
            M = (e'*H3ay)';
            J.Hc = -(sy(:).*e(:)./denom(:).^2).*M(:);
            clear M
            
        else
            d = length(e);
            G = param.H;
            G = G./repmat(c_h(:,1),[1 d]);
            %M = kron(e',ones(d,1))*G';
            %v = -(sy(:).*K(:).*e(:)./denom(:).^2).*get_diags(M);
            M = (e'*G')';                                                     %%%% Copy this way of multiplying when using repeated "e" and apply all over the place (forget about get_diags!!)
            v = -(sy(:).*K(:).*e(:)./denom(:).^2).*M(:);
            clear M
            %M = kron(e_a',ones(d ,1))*G';
            M = (e_a'*G')';
            clear G;
            %v = v + (sy(:).*kappa(:).*e(:)./(e_a(:).*denom(:))).*get_diags(M);
            v = v + (sy(:).*kappa(:).*e(:)./(e_a(:).*denom(:))).*M(:);
            clear M;
            M = spalloc(Npatch*d,d,Npatch*d);
            J.Hc = put_diags(M,v)*Hc_struct;
            clear M
        end
    end
    %% YA
    
    if isfield(Jcomp,'Hw') && Jcomp.Hw==1
            d = length(e);
            G = param.H;
            G = G./repmat(w_h',[d 1]);
            Ge = repmat(G,[Npatch 1]).*kron(e',ones(d,1));
            Gea = repmat(G,[Npatch 1]).*kron(e_a',ones(d,1));
            clear G
            Ja = repmat(-sy(:).*K(:).*e(:)./denom(:).^2,[1 d]).*Ge + repmat( sy(:).*kappa(:).*e(:)./(e_a(:).*denom(:)) , [1 d] ).*Gea;
            J.Hw = Ja*Hw_struct;
            clear Ja
    end
    %% YA    
    
    % computJ.L   Jacobian of the linear response with regard to the linear matrix
    if (isfield(Jcomp,'L') && Jcomp.L==1)
        if Npatch==1
            J.L = J.ny*block_diagonal(xi',d);
        else
            disp(' Full derivative w.r.t. non-parametric kernel cannot be computed for multiple images (huge matrices)')
        end
    end
    if (isfield(Jcomp,'L_compact') && Jcomp.L_compact==1)
        J.L_matrix = J.ny;
        J.L_vector = xi;
    end
    %% YA
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %------------------------------------------------------------------------
    % CONTRAST FROM BRIGTHNESS (particular always-positive denominator: no
    %need to separate sign and absolute value)
    %------------------------------------------------------------------------
    
elseif param.contrast == 1
    % LINEAR ------------------------------------------
    L = param.L;
    iL = param.iL;
    yy = L*xi;
    
    % NONLINEAR ------------------------------------------
    if (isfield(Jcomp,'sx') && Jcomp.sx==1) || (isfield(Jcomp,'L')...
            && Jcomp.L==1) || (isfield(Jcomp,'L_compact') && Jcomp.L_compact==1)
        Jcomp.ny = 1;
    end
    y = yy;
    c_h = param.Hc;
    c_h1 = param.Lc;
    denom = b + H*iL*y;
    x = y./denom;
    
    % Derivative Analytic ------------------------------------------
    
    if isfield(Jcomp,'ny') && Jcomp.ny==1
        Positive = 1./denom(:);
        a = spalloc(d*Npatch,d,Npatch*d);
        Vaux2 = y(:)./denom(:).^2;
        % Vaux1 = ones(length(y(:)),1);   
        % In the contrast case there is no extra right-multiplication...
        %(as opposed to the other cases) -> we can remove Vaux1
        Negative = repmat(Vaux2,[1 d]).*repmat(H*iL,[ Npatch 1]); %.*kron(Vaux1,ones(1, d))';
        J.ny = put_diags(a,Positive)-Negative;
        clear Negative Positive Vaux1 Vaux2 a
    end
    if isfield(Jcomp,'g') && Jcomp.g==1
        % J.g =sy.*(1./denom).*(elog -(1./denom).*(H*diag(e)*elog)).*e;
        % J.g= sy.*(1./denom).*(elog -(1./denom).*(H.*Ve'*elog)).*e;
        % J.g = (sy(:)./denom(:)).*(elog(:) -(1./denom(:)).*(chu(:))).*e(:);
        
        disp(' In the restricted div. norm., the excitatory exponent is fixed (gamma = 1)')
        J.g = [];
    end
    if (isfield(Jcomp,'b') && Jcomp.b==1) 
        aux= -y(:)./(denom(:).^2);
        if SizeB==1
            J.b=aux.*repmat(ones(d,1),[Npatch 1]);
        elseif SizeB==d
            a=spalloc(Npatch*d,d,Npatch*d);
            J.b=put_diags(a,aux);
        else
            disp('Error in the b size')
        end       
    end    
    if isfield(Jcomp,'H') && Jcomp.H==1
        if Npatch==1
            Bde = block_diagonal((iL*y)',d);
            J.H= -diag(y./denom.^2)*Bde;
        else
            disp(' Full derivative w.r.t. non-parametric kernel cannot be computed for multiple images (huge matrices)')
        end
    end
    
    if isfield(Jcomp,'H_compact') && Jcomp.H_compact==1
        % Bde = block_diagonal(e',d);
        % J.H= -diag(sy.*e)*diag(1./denom.^2)*Bde;
        J.H_matrix= -y./denom.^2;
        J.H_vector= iL*y;
    end
    
    if isfield(Jcomp,'Hs') && Jcomp.Hs==1
        % J.H=  -diag(sy.*e)*diag(1./denom.^2)*Bde;
        [~,dHds3] = make_2d_gauss_kernel(param.fs,param.N,param.Hs);
        % M = repmat(e',[d 1])*dHds3;
        %J.Hs = -(c_h.*sy.*e./denom.^2).*diag(M);
        %J.Hs = -(c_h(:).*sy(:).*e(:)./denom(:).^2).*diag(M);
        %M = kron((iL*y)',ones(d ,1))*dHds3;    
        %v = -(c_h(:).*y(:)./denom(:).^2).*get_diags(M);
        M = ((iL*y)'*dHds3)';
        v = -(c_h(:).*y(:)./denom(:).^2).*M(:);
        clear M;
        M = spalloc(Npatch*d,d,Npatch*d);
        J.Hs = put_diags(M,v)*ones(d,1);
        clear M
    end
    if isfield(Jcomp,'Hc') && Jcomp.Hc==1
        % J.H=  -diag(sy.*e)*diag(1./denom.^2)*Bde;
        [H3ay,~] = make_2d_gauss_kernel(param.fs,param.N,param.Hs);
        % M = kron((iL*y)',ones(d,1))*H3ay;  
        % J.Hc = -(y(:)./denom(:).^2).*get_diags(M);
        M = ((iL*y)'*H3ay)';
        J.Hc = -(y(:)./denom(:).^2).*M(:);        
        clear M
    end
    xim1=x;
    % Derivatives wrt signal -----------------------------------
    J.lx=0;
    if (isfield(Jcomp,'sx') && Jcomp.sx==1)
        J.sx = J.ny*L;
    end
    if (isfield(Jcomp,'lx') && Jcomp.lx==1)
        J.lx = L;
    end
    
    % Deriv wrt parameters (not computed in resp_DN)--------------
    % computJ.L   Jacobian of the linear response with regard to the linear matrix
    if (isfield(Jcomp,'L') && Jcomp.L==1)
        if Npatch==1
            J.L = J.ny*block_diagonal(xi',d);
        else
            disp(' Full derivative w.r.t. non-parametric kernel cannot be computed for multiple images (huge matrices)')
        end
    end
    if (isfield(Jcomp,'L_compact') && Jcomp.L_compact==1)
        J.L_matrix = J.ny;
        J.L_vector = xi;
    end
    
    if isfield(Jcomp,'Ls') && Jcomp.Ls==1
        % J.H=  -diag(sy.*e)*diag(1./denom.^2)*Bde;
        [~,dHds3] = make_2d_gauss_kernel(param.fs,param.N,param.Ls);
        % M = repmat(e',[d 1])*dHds3;
        %J.Hs = -(c_h.*sy.*e./denom.^2).*diag(M);
        %J.Hs = -(c_h(:).*sy(:).*e(:)./denom(:).^2).*diag(M);
        %M = kron(c_h1*xi',ones(d ,1))*dHds3;   
        %v = -get_diags(M);
        M = ((c_h1*xi')*dHds3)';
        v = -M(:);
        clear M;
        M = spalloc(Npatch*d,d,Npatch*d);
        v = put_diags(M,v)*ones(d,1);
        clear M
        v = reshape(v,d,Npatch);
        J.Ls = [];
        for i=1:Npatch
            Jac = sacafot(J.ny',d,d,i)';
            J.Ls = [J.Ls;Jac*v(:,i)];
        end
    end
    
    if isfield(Jcomp,'Lc') && Jcomp.Lc==1
        % J.H=  -diag(sy.*e)*diag(1./denom.^2)*Bde;
        [H3ay,~] = make_2d_gauss_kernel(param.fs,param.N,param.Ls);
        % M = kron(xi',ones(d,1))*H3ay;            
        % v = -get_diags(M);
        M = (xi'*H3ay)';
        v = -M(:);
        clear M
        v = reshape(v,d,Npatch);
        J.Lc = [];
        for i=1:Npatch
            Jac = sacafot(J.ny',d,d,i)';
            J.Lc = [J.Lc ; Jac*v(:,i)];
        end
    end
    % ---------------------------------------------------------------------
    % LUMINANCE TO BRIGHTNESS
    % ---------------------------------------------------------------------
else
    beta = param.beta(param.channel);
    aux=find(xi==0);
    xi(aux)=1e-6;
    
    % LINEAR--------------------------------------------------
    L = param.L;
    yy = L*xi;
    % NONLINEAR--------------------------------------------------
    if (isfield(Jcomp,'sx') && Jcomp.sx==1) || (isfield(Jcomp,'L') && Jcomp.L==1) || (isfield(Jcomp,'L_compact') && Jcomp.L_compact==1)
        Jcomp.ny = 1;
    end
    % RESP_DN --------------------------------------------------
    H = H(:,:,param.channel);
    y = yy;
    sy = sign(y);
    e = abs(y).^gamma;
    elog=log(abs(y));
    denom = b + H*e;
    all_1 = ones(d,d);
    %beta = beta(param.channel);
    K = param.scale(param.channel)*( b + (beta/d)*all_1*e + ones(d,Npatch));
    x = sy.*K.*e./denom;
    % Derivative Analytic --------------------------------
    
    if isfield(Jcomp,'ny') && Jcomp.ny==1
        % Derivative of the general DN
        % Positive=sy(:).*((gamma*y(:).^(gamma-1))./denom(:)).*sy(:);
        % a = spalloc(d*Npatch,d,Npatch*d);
        % Vaux2=sy(:).*e(:)./denom(:).^2;
        % Vaux1=(gamma* y.^(gamma-1)).*sy;
        % Negative=repmat(Vaux2,[1 d]).*repmat(H,[ Npatch 1]).*kron(Vaux1,ones(1, d))';
        % J.ny = put_diags(a,Positive)-Negative;
        
        a = spalloc(d*Npatch,d,Npatch*d);
        Positif = K(:).*((gamma*y(:).^(gamma-1))./denom(:));
        Positive = put_diags(a,Positif);
        clear Positif a
        
        Vaux2 = sy(:).*K(:).*e(:)./denom(:).^2;   %%%% added sign
        Vaux1 = (gamma*y(:).^(gamma-1)).*sy(:);      %%%% added sign
        Negative = repmat(Vaux2,[1 d]).*repmat(H,[ Npatch 1]).*kron(Vaux1,ones(1, d))';
        clear Vaux2
        
        % v = repmat((beta/d)*(e(:)./denom(:)),[1 d]).*kron((gamma* y.^(gamma-1)),ones(1,d))';
        % adenda_constant = repmat((beta/d)*(e(:)./denom(:)),[1 d]).*kron(Vaux1,ones(1,d))';
        % v = (beta/d)*all_1*(gamma*y.^(gamma-1));
        % adenda_constant = (e(:)./denom(:)).*v(:);
        % clear Vaux1 v
        % adenda_constant2 = put_diags(a,adenda_constant);
        adenda_constant2 = param.scale(param.channel)*(beta/d)*repmat((sy(:).*e(:)./denom(:)),[1 d]).*(kron(Vaux1,ones(1,d)))';  %%%% added sign
        J.ny = Positive - Negative + adenda_constant2;
        clear Negative Positive Vaux1 Vaux2 a
    end
    
    if isfield(Jcomp,'g') && Jcomp.g==1
        % J.g =sy.*(1./denom).*(elog -(1./denom).*(H*diag(e)*elog)).*e;
        % J.g= sy.*(1./denom).*(elog -(1./denom).*(H.*Ve'*elog)).*e;
        chu = H*(e.*elog);
        v = param.scale(param.channel)*beta/d*(e./denom).*(all_1*(elog.*e));
        J.g = (sy(:).*K(:)./denom(:)).*(elog(:) -(1./denom(:)).*(chu(:))).*e(:) +  sy(:).*v(:);   %%%%% added sign
    end
    if (isfield(Jcomp,'b') && Jcomp.b==1) 
          aux = -sy(:).*K(:).*e(:)./(denom(:).^2) + param.scale(param.channel)*sy(:).*e(:)./denom(:);  %%%%% added sign
        if SizeB==1%(isfield(Jcomp,'b') && Jcomp.b==1)
            J.b=aux.*repmat(ones(d,1),[Npatch 1]);
        elseif SizeB==d% (isfield(Jcomp,'b_d') && Jcomp.b_d==1)
            a=spalloc(Npatch*d,d,Npatch*d);
            J.b=put_diags(a,aux);
        else
            disp('Error in the b size')
        end
    end
     
    if isfield(Jcomp,'H') && Jcomp.H==1
        disp(' Paramterization of the interaction kernel in the brightness transform  ')
        disp(' implies that derivation w.r.t. generic kernel is not possible ')
        disp(' J.H will be leaved empty. J.beta is computed instead ')
        J.H = [];
        Jcomp.beta = 1;
    end
    
    if isfield(Jcomp,'H_compact') && Jcomp.H_compact==1
        disp(' Paramterization of the interaction kernel in the brightness transform  ')
        disp(' implies that derivation w.r.t. generic kernel is not possible ')
        disp(' J.H_matrix and J.H_vector will be leaved empty. J.beta is computed instead')
        Jcomp.beta = 1;
        J.H_matrix= [];
        J.H_vector= [];
    end
    
    if isfield(Jcomp,'Hs') && Jcomp.Hs==1
        disp(' Paramterization of the interaction kernel in the brightness transform is not Gaussian ')
        disp(' therefore derivation w.r.t. sigma is not possible ')
        disp(' J.Hs will be leaved empty. J.beta is computed instead')
        Jcomp.beta = 1;
        J.Hs= [];
    end
    
    if isfield(Jcomp,'Hc') && Jcomp.Hc==1
        disp(' Paramterization of the interaction kernel in the brightness transform is not Gaussian ')
        disp(' therefore derivation w.r.t. the Gaussian amplitudes is not possible ')
        disp(' J.Hc will be leaved empty. J.beta is computed instead')
        Jcomp.beta = 1;
        J.Hs= [];
    end
    xim1=x;
    
    if isfield(Jcomp,'beta') && Jcomp.beta==1
        med = all_1*e;
        positif = (1/d)*(sy(:)./denom(:)).*(param.scale(param.channel)*med(:)).*e(:);   %%%%%%%%%%% added sign
        negatif = (1/d)*(sy(:)./denom(:)).*(K(:).*(e(:)./denom(:)).*med(:));            %%%%%%%%%%% added sign          
        
        % (kk/d)*(m(:)).*(e(:)./D(:)) - (1/d)*K.*(e(:)./(D(:).^2)).*m(:);
        %KK = param.scale*(b(:) + (beta/d)*med(:) + 1);
        %J.beta = (param.scale/d)*(med(:)).*(e(:)./denom(:)) - (1/d)*KK.*(e(:)./(denom(:).^2)).*med(:);
        
        J.beta = positif-negatif;
        clear med
    end
    
    if isfield(Jcomp,'scale') && Jcomp.scale==1
        J.scale = x(:)/param.scale(param.channel);
    end
    % Derivatives wrt signal-------------------------------------
    J.lx=0;
    d=length(xi);
    if (isfield(Jcomp,'sx') && Jcomp.sx==1)
        J.sx = J.ny*L;
    end
    if (isfield(Jcomp,'lx') && Jcomp.lx==1)
        J.lx = L;
    end
    % Deriv wrt parameters (not computed in resp_DN) ---------------------
    % computJ.L Jacobian of the linear response with regard to the linear matrix
    if (isfield(Jcomp,'L') && Jcomp.L==1)
        if Npatch==1
            J.L = J.ny*block_diagonal(xi',d);
        else
            disp(' Full derivative w.r.t. non-parametric kernel cannot be computed for multiple images (huge matrices)')
        end
    end
    if (isfield(Jcomp,'L_compact') && Jcomp.L_compact==1)
        J.L_matrix = J.ny;
        J.L_vector = xi;
    end
    
end