function varargout = gui(varargin)
% GUI MATLAB code for gui.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help gui

% Last Modified by GUIDE v2.5 10-Feb-2020 19:06:58

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @gui_OpeningFcn, ...
                   'gui_OutputFcn',  @gui_OutputFcn, ...
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


% --- Executes just before gui is made visible.
function gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to gui (see VARARGIN)

% Choose default command line output for gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% next step button
global im3 im4
im4 = im3;
axes(handles.axes2);
imshow(im4);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% transform

global im4 im5
im5 = im4 ;
    
   
if (get(handles.radiobutton1, 'Value')== get(handles.radiobutton1, 'max'))    
    global im5
    img = im5;
    gray_img = rgb2gray(img);  
    Iy = imfilter(double(gray_img), fspecial('sobel') , 'replicate');
    Ix = imfilter(double(gray_img), fspecial('sobel')' , 'replicate');       
    gradient = sqrt(Ix.^2+Iy.^2);
    L = watershed(gradient);                                      
    Io = imopen(gray_img,strel('disk', 20) );
    Ie = imerode(gray_img, strel('disk', 20));
    Iobr = imreconstruct(Ie, gray_img);    
    Ioc = imclose(Io, strel('disk', 20)); 
    Iobrd = imdilate(Iobr, strel('disk', 20));
    Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr)); 
    Iobrcbr = imcomplement(Iobrcbr); 
    fgm = imregionalmax(Iobrcbr);
    fgm2 = imclose(fgm,strel(ones(5,5))); 
    fgm3 = imerode(fgm2,strel(ones(5,5)));
    fgm4 = bwareaopen(fgm3, 20);     
    bw = im2bw(Iobrcbr, graythresh(Iobrcbr));
    D = bwdist(bw); 
    DL = watershed(D);
    bgm = DL == 0;  
    gradient2 = imimposemin(gradient, bgm | fgm4);
    L = watershed(gradient2);
    bgm2 = L ==0;    
    final = gray_img; 
    final(imdilate(L == 0, ones(3, 3)) | bgm | fgm4); 
    result =final;
    axes(handles.axes3);
    Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
    im5 = Lrgb;
    imshow(im5);
                  
end



if (get(handles.radiobutton2, 'Value')== get(handles.radiobutton2, 'max'))
    global im5
    img = im5;
    I = rgb2gray(img);
    level1=multithresh(I,3);
    seg_I1 = imquantize(I,level1);
    L = label2rgb(seg_I1);
    im5 = L;
    axes(handles.axes3);
    imshow(im5);
end


if (get(handles.radiobutton3, 'Value')== get(handles.radiobutton3, 'max'))
    global im5
    A = im5;
    A = imresize(A,0.25);
    Agray = rgb2gray(A);
    imageSize = size(A);
    numRows = imageSize(1);
    numCols = imageSize(2);

    wavelengthMin = 4/sqrt(2);
    wavelengthMax = hypot(numRows,numCols);
    n = floor(log2(wavelengthMax/wavelengthMin));
    wavelength = 2.^(0:(n-2)) * wavelengthMin;

    deltaTheta = 45;
    orientation = 0:deltaTheta:(180-deltaTheta);

    g = gabor(wavelength,orientation);
    gabormag = imgaborfilt(Agray,g);
    for i = 1:length(g)
        sigma = 0.5*g(i).Wavelength;
        K = 3;
        gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),K*sigma); 
    end
    X = 1:numCols;
    Y = 1:numRows;
    [X,Y] = meshgrid(X,Y);
    featureSet = cat(3,gabormag,X);
    featureSet = cat(3,featureSet,Y);
    numPoints = numRows*numCols;
    X = reshape(featureSet,numRows*numCols,[]);
    X = bsxfun(@minus, X, mean(X));
    X = bsxfun(@rdivide,X,std(X));
    coeff = pca(X);
    feature2DImage = reshape(X*coeff(:,1),numRows,numCols);
    L = kmeans(X,2,'Replicates',5);
    L = reshape(L,[numRows numCols]);
    Aseg = zeros(size(A),'like',A);
    BW = L == 2;
    BW = repmat(BW,[1 1 3]);
    Aseg(BW) = A(BW);
    axes(handles.axes3);
    imshow(Aseg);
   
end




if (get(handles.radiobutton4, 'Value')== get(handles.radiobutton4, 'max'))
    axes(handles.axes2);
    global im5
    I = im5;
    if(exist('reg_maxdist','var')==0), reg_maxdist=0.2; end
    if(exist('y','var')==0),imshow(I,[]); [y,x]=getpts; y=round(y(1)); x=round(x(1)); end

    J = zeros(size(I));
    Isizes = size(I);

    reg_mean = I(x,y); 
    reg_size = 1; 
    neg_free = 10000; neg_pos=0;
    neg_list = zeros(neg_free,3); 

    pixdist=0;
    neigb=[-1 0; 1 0; 0 -1;0 1];
    while(pixdist<reg_maxdist&&reg_size<numel(I))
        for j=1:4
            xn = x +neigb(j,1); yn = y +neigb(j,2);
            ins=(xn>=1)&&(yn>=1)&&(xn<=Isizes(1))&&(yn<=Isizes(2));
            if(ins&&(J(xn,yn)==0)) 
                    neg_pos = neg_pos+1;
                    neg_list(neg_pos,:) = [xn yn I(xn,yn)]; J(xn,yn)=1;
            end
        end
        if(neg_pos+10>neg_free), neg_free=neg_free+10000; neg_list((neg_pos+1):neg_free,:)=0; end
        dist = abs(neg_list(1:neg_pos,3)-reg_mean);
        [pixdist, index] = min(dist);
        J(x,y)=2; reg_size=reg_size+1;
        reg_mean= (reg_mean*reg_size + neg_list(index,3))/(reg_size+1);
        x = neg_list(index,1); y = neg_list(index,2);
        neg_list(index,:)=neg_list(neg_pos,:); neg_pos=neg_pos-1;
    end
    J=J>1;

    axes(handles.axes3);
    im5 = I+J;
    imshow(im5);
    
    
end


if (get(handles.radiobutton5, 'Value')== get(handles.radiobutton5, 'max'))
    global im5
    I = im5;
    I = im2double(I);
    I = rgb2gray(I);
    I = I*256;
    I_out = splitmerge(I, 2, @predicate);
    axes(handles.axes3);
    imshow(I_out);
     
end



% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% load image
global im im2 im3
[path,user_cance] =imgetfile();
if user_cance
    msgbox(sprintf('You did not choose image'),'Error','Error');
    return
end
im=imread(path);
im=im2double(im);%convert to double
im2=im;%for backup process
im3=im;
axes(handles.axes1);
imshow(im3);




% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% reset
global im2 im3
im3 = im2;
axes(handles.axes1);
imshow(im3);




% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% crop
global im3 
axes(handles.axes1);
img = im3;
rect = wait(imrect);
I2 = imcrop(img,rect);
im3 = I2;
imshow(I2);





% --- Executes on button press in Right.
function Right_Callback(hObject, eventdata, handles)
% hObject    handle to Right (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%rotate -90
global im3 
axes(handles.axes1);
img = im3;
I2 = imrotate(img,-90);
im3 = I2;
imshow(I2)

% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%rotate +90
global im3 
axes(handles.axes1);
img = im3;
I2 = imrotate(img,+90);
im3 = I2;
imshow(I2)




% --- Executes on button press in pushbutton17.
function pushbutton17_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%save
axes(handles.axes3);
global im5
[filename,user_canceled] = imsave();


%%% split_merge segmentation

function  g=splitmerge(f,mindim,fun)
Q=2^nextpow2(max(size(f)));
[M,N]=size(f);
f=padarray(f,[Q-M,Q-N],'post');
S=qtdecomp(f,@split_test,mindim,fun);
Lmax=full(max(S(:)));
g=zeros(size(f));
MARKER=zeros(size(f));

for K=1:Lmax
    [vals,r,c]=qtgetblk(f,S,K);
    if ~isempty(vals)
        for I=1:length(r)
            xlow=r(I);
            ylow=c(I);
            xhigh=xlow+K-1;
            yhigh=ylow+K-1;
            region=f(xlow:xhigh,ylow:yhigh);
            flag=predicate2(region);
            if flag
                g(xlow:xhigh,ylow:yhigh)=1;
                MARKER(xlow,ylow)=1;
            end
        end
    end
end
g=bwlabel(imreconstruct(MARKER,g));
g=g(1:M,1:N);



function v=split_test(B,mindim,fun)
K=size(B,3);
v(1:K)=false;
for I=1:K
    quadregion=B(:,:,I);
    if size(quadregion,1)<=mindim
        v(I)=false;
        continue
    end
    flag=feval(fun,quadregion);
    if flag
        v(I)=true;
    end
end



function flag=predicate(region)
sd=std2(region);
flag=(sd>2);


function flag2=predicate2(region)
m=mean2(region);
flag2=(m<150);


function I_bw = ThresholdSplit(I)
level = graythresh(I);
disp(level);
I_bw = im2bw(I, level);


% --- Executes on button press in pushbutton19.
function pushbutton19_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im5 im4
sizex = size(im5);
i = ones(sizex);
axes(handles.axes3);
imshow(i);
