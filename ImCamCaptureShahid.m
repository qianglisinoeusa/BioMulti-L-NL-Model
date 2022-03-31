function varargout = ImCamCaptureShahid(varargin)
% IMCAMCAPTURESHAHID MATLAB code for ImCamCaptureShahid.fig
%   imaqhwinfo
%   VidObj= videoinput('winvideo',1, 'RGB24_320x240');
%   imaqhwinfo(VidObj)
%  See also: GUIDE, GUIDATA, GUIHANDLES
% Edit the above text to modify the response to help ImCamCaptureShahid
% Last Modified by GUIDE v2.5 16-May-2013 21:03:08
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ImCamCaptureShahid_OpeningFcn, ...
                   'gui_OutputFcn',  @ImCamCaptureShahid_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end
if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
% --- Executes just before ImCamCaptureShahid is made visible.
function ImCamCaptureShahid_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ImCamCaptureShahid (see VARARGIN)
set(handles.PRW,'Visible','off')
% Choose default command line output for ImCamCaptureShahid
handles.output = hObject;
% Update handles structure
guidata(hObject, handles);
% UIWAIT makes ImCamCaptureShahid wait for user response (see UIRESUME)
% uiwait(handles.figure1);
% --- Outputs from this function are returned to the command line.
function varargout = ImCamCaptureShahid_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Get default command line output from handles structure
varargout{1} = handles.output;
% --- Executes on button press in Start.
function Start_Callback(hObject, eventdata, handles)
% hObject    handle to Start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global IA DeviceID Format
IAHI=imaqhwinfo;
IA=(IAHI.InstalledAdaptors);
D=menu('Select Video Input Device:',IA);
if isempty(IA)||D==0
   msgbox({'You dont have any VideoInput Installed Adaptors!',...
           'OR',...
           'Please! try again and select Adaptor properly.'})
    return
end
IA=char(IA);
IA=IA(D,:);
IA(IA==' ')=[];
x=imaqhwinfo(IA);
try
DeviceID=menu('Select Device ID',x.DeviceIDs);
F=x.DeviceInfo(DeviceID).SupportedFormats;
nF=menu('Select FORMAT',F);
Format=F{nF};
catch e
   warndlg({'Try Another Device or ID ';...
            'You Donot Have Installed This Device(VideoInputDevice)'})
    return
end
% --- Executes on button press in Capture.
function Capture_Callback(hObject, eventdata, handles)
global S CAM;
if(CAM==1)
    CAM=0;
    S=getsnapshot(handles.VidObj);
    closepreview
%     clear VidObj
%     delete  VidObj
     imshow(S,'parent',handles.PRW);
else
   msgbox('Plz! Start Cam First by PUSHBUTTON') 
end
% --- Executes on button press in prwbutton.
function prwbutton_Callback(hObject, eventdata, handles)
global IA DeviceID Format  CAM
try
VidObj= videoinput(IA, DeviceID, Format);
handles.VidObj=VidObj;CAM=1;
vidRes = get(handles.VidObj, 'VideoResolution');
nBands = get(handles.VidObj, 'NumberOfBands');
set(handles.PRW,'Visible','off')
axes(handles.PRW)
hImage = image( zeros(vidRes(1), vidRes(2), nBands) );
preview(handles.VidObj, hImage)
catch E
    msgbox({'Configure The Cam Correctly!',' ',E.message},'CAM INFO')
end
guidata(hObject, handles);
% --- Executes on button press in SAVE.
function SAVE_Callback(hObject, eventdata, handles)
% hObject    handle to SAVE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global CAM
[F,~,NotGet]=imputfile;
S=getimage(handles.PRW);
if(~NotGet&&~isempty(S)&& ~CAM)
    imwrite(S,F)
    msgbox(strcat('Image is saved at :',F))
else 
    msgbox('Image is not saved: First CAPTURE IT')
end
% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    clear handles.VidObj
    delete (instrfind) 
% Hint: delete(hObject) closes the figure
delete(hObject);