% defaultParams - Inicializa os valores por defecto da estructura params.
function params= defaultParams(sze)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Tamaño da imaxe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params.sze=sze;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parametros por defecto da gaussiana para achar o mapa
%densidade de atencion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params.maxhw = max(0,floor(min(sze(1)/2,sze(2)/2) - 1));
params.sig = max(sze)*0.04;     %poderiamos axustalo aos pixels de 1º visual
                                % en vez dun porcentaxe das dimensions da imaxe                               
%params.mindist=max(0,floor(min(sze(1)/10,sze(2)/10) - 1));
                          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% valores por defecto dos parametros de control do banco
% de filtros monoxenicos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Bfo=1;  %distribución das frecuencias centrales dos filtros en octavas
Bab=2.3548;  %Anchura do filtro en octavas  --> params.banco.sigmaOnf=0.5;
params.banco.nscale=3;
params.banco.minWaveLength=8;
params.banco.mult=exp(Bfo*log(2));
params.banco.sigmaOnf=(Bab*sqrt(log(2)/2))/2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parametros para controlar a compensacion de ruido
% e os puntos onde a fase se aliña ao longo das escalas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params.k               = 1;
params.cutOff          = 0.25;    
params.g               = 10;       
