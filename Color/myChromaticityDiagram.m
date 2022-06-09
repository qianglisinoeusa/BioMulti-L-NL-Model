function myChromaticityDiagram(img, useCurvefit, useStats)
%% 表示用のオブジェクトの作成
% Add the UI components
hs = addcomponents;
% コンポーネントが揃ってから可視化
% Make figure visible after adding components
hs.fig.Visible = 'on';
%% 画像データ表示、色空間表示
%Display image and 3-D color gamut
imshow(img,'Parent',hs.ax1)
hs.ax1.Title.String='Input Image';
hs.ax1.Title.FontName='Yu Gothic UI Light';
%RGB色空間で3D表示
%Use RGB color gamut as a default
colorcloud(img,'rgb','Parent',hs.ui);
hs.ui.Title = '3D Color Gamut';
hs.ui.FontName='Yu Gothic UI Light';
%% RGB > xyz変換
%Convert RGB to CIE 1931 XYZ
XYZ = rgb2xyz(img);
sz = size(XYZ);
XYZ = reshape(XYZ, [sz(1)*sz(2), 3]);
sX = XYZ(:,1) ./ (XYZ(:,1) + XYZ(:,2) + XYZ(:,3));
sY = XYZ(:,2) ./ (XYZ(:,1) + XYZ(:,2) + XYZ(:,3));
sZ = XYZ(:,3) ./ (XYZ(:,1) + XYZ(:,2) + XYZ(:,3));
%% XY色度で表現できる範囲を可視化
% Load spectral locus xy values at 1-nm intervals and plot it
load('locus.mat')
plotLineWidth = 2;
plot(locus(:,1),locus(:,2),'k','LineWidth',plotLineWidth,'Parent',hs.ax2);
grid(hs.ax2, 'on')
hold(hs.ax2, 'on')
axis([0.0 0.85 0.0 0.85])
xlabel('x')
ylabel('y')
% non-spectralな領域をプロット
% plot the non-spectral locus
plot(hs.ax2, [locus(1,1) locus(end,1)], [locus(1,2) locus(end,2)],'k','LineWidth',plotLineWidth)
%% xy色度図内の色分布をテーブルとして作成
% Create table for color reproduction on chromaticity diagram
xy4rgb = createChromaticityTable;
%% 入力画像のxyに該当する色情報を抽出
% Map xy value of input image to color reproduction table data
xy = [sX sY];
if license('test','curve_fitting_toolbox') && useCurvefit
    x = xy4rgb(:,1);
    y = xy4rgb(:,2);
    % モデルをデータに近似します
    % Fit surface to color reproduction table data
    fittedmodel_r = fit( [x, y], xy4rgb(:,3), 'linearinterp');
    fittedmodel_g = fit( [x, y], xy4rgb(:,4), 'linearinterp');
    fittedmodel_b = fit( [x, y], xy4rgb(:,5), 'linearinterp');
    R = fittedmodel_r(sX, sY);
    G = fittedmodel_g(sX, sY);
    B = fittedmodel_b(sX, sY);
    clr = [R G B];
else
    %% 用意したテーブル値(xy4rgb)に対し、入力画像のxyを割り当て
    % Find nearest XY coordinates from xy value of input image
    sz = size(img);
    ref = xy4rgb(:,1:2);
    idx = idxsearch(ref, xy, useStats);
    %% 算出したIndexに該当する色成分を抽出
    % Extract RGB value that is associated with xy data of input image
    clr = xy4rgb(idx,:);
    clr(:,1:2) = [];
end
%% Scatterでxy座標上に該当する色をプロット
% Plot RGB data associated with xy data of input image
scatter(xy(:,1), xy(:,2), [], clr,'.','Parent',hs.ax2)
hs.ax2.Title.String='Chromaticity Diagram';
hs.ax2.Title.FontName='Yu Gothic UI Light';
%% Supporting Functions
    %結果表示用オブジェクトの生成
    % Add the UI components
    function hs = addcomponents
        % 利用中のディスプレイ解像度算出とFigureの大きさ決定
        % Difine the size of figure based on screen size
        sz = get(0,'ScreenSize');
        hs.fig = figure('Visible','off','Position',[180 sz(4)-500 200+700 300],...
            'Tag','fig','SizeChangedFcn',@resizeui);
    
        figwidth = hs.fig.Position(3);
        figheight = hs.fig.Position(4);
        bheight = 50 * (figheight/300); 
        bwidth = 100 * (figwidth/900);
        %bbottomedge = figheight - bheight - 50;
        bbottomedge = -20;
        bleftedge = figwidth/3;
        fontsize = 8 * (figheight/300);
        hs.ax1 = axes('Parent',hs.fig,'Position', [0, 0.05, 0.3, 0.9],'Tag','ax1');
        hs.ui = uipanel('Parent',hs.fig,'Position', [0.35, 0, 0.3, 1],'Tag','ui');
        hs.ax2 = axes('Parent',hs.fig,'Position', [0.7, 0.1, 0.25, 0.8],'Tag','ax2');
    
        hs.pop = uicontrol(hs.fig,'Style', 'popup','Tag','popup',...
            'String', {'rgb','lab','hsv','ycbcr'},...
            'Position', [bleftedge bbottomedge bwidth bheight], 'Callback', @setspace, 'FontSize', fontsize);  
    end
    %% 色空間選択用パネル
    % Create UI panel for setting color space for 3-D color gamut display
    function setspace(hObject,event)
        hs.ui = uipanel('Parent',hs.fig,'Position', [0.35, 0, 0.3, 1],'Tag','ui');
        uistack(hs.pop, 'top');
        color = hs.pop.String{hs.pop.Value};
        colorcloud(img,color,'Parent',hs.ui);
        hs.ui.Title = '3D Color Gamut';
        hs.ui.FontName='Yu Gothic UI Light';    
    end
    %% Figureサイズ変更時のレイアウト管理
    % Managing the Layout in Resizable UIs
    function resizeui(hObject,event) 
        % Figureの高さ、幅を算出
        % Get the height and width of figure
        figwidth = hs.fig.Position(3);
        figheight = hs.fig.Position(4);
       
        % UIパネルの設置位置決定
        % Define the size and position of UI panel
        bheight = 50 * (figheight/300); 
        bwidth = 100 * (figwidth/900);    
        %bbottomedge = figheight - bheight - 50;
        bbottomedge = -20;
        bleftedge = figwidth/3;
        fontsize = 8 * (figheight/300);
        hs.pop.Position = [bleftedge bbottomedge bwidth bheight];
        hs.pop.FontSize = fontsize;
    end
    %% xyの各点の最近傍点をref内で検索
    % Find nearest XY coordinates from xy value of input image
    function idx = idxsearch(ref, xy, useStats)
        %Statistics and Machine Learning Toolboxのライセンスがある場合
        if license('test','statistics_toolbox') && useStats
            %Use knnsearch
            idx = knnsearch(ref,xy);
        %無い場合はユークリッド距離をベースとした近傍検索
        %If Statistics and Machine Learning Toolbox is not available,
        %calculate euclidean distance to find nearest XY value
        else
            s = size(xy);
            idx = zeros(s(1),1);
            for i = 1:s(1)
                dist = sqrt((ref(:,1)-xy(i,1)).^2 + (ref(:,2)-xy(i,2)).^2);
                [~, idx(i)] = min(dist);
            end
        end
        
        % x = mat2cell(xy(:,1),ones(s(1),1), 1);
        % y = mat2cell(xy(:,2),ones(s(1),1), 1);
        
        % A = cellfun(@(d1,d2) sqrt((ref(:,1)-d1).^2 + (ref(:,2)-d2).^2), x, y, 'UniformOutput',false);
    end
end
%   Copyright 2018 The MathWorks, Inc.