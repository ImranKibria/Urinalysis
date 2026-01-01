% PORTRAIT. 5x7in. A4 paper.  
%% WHITE BACKGROUND
clc; close all; clear; 

%% STEP 1: MacBeth COLOR CHECKER DESIGNING

DesignPaletteRGBs = [254 254 254; 254 248 188; 255 255 255; 255 255 255; 255 255 255; 230 218 175; 222,184,135; 207 159 150; 165 120 153; 110 083 125;   
            254 254 254; 254 251 223; 255 255 255; 255 255 255; 255 255 255; 251 219 217; 255 192 203; 246 182 201; 255 105 180; 238 079 131;    
            254 254 254; 252 212 174; 255 255 255; 255 255 255; 255 255 255; 249 169 135; 255 255 255; 242 132 146; 230 111 129; 238 079 131;  
            254 254 254; 223 230 125; 255 255 255; 255 255 255; 255 255 255; 187 216 107; 173 213 131; 119 190 152; 094 178 169; 000 149 149; 
            254 254 254; 245 140 080; 250 167 086; 255 255 255; 255 255 255; 254 196 109; 209 190 099; 137 149 085; 085 174 146; 000 127 130;  
            254 254 254; 250 175 077; 251 185 078; 255 255 255; 255 255 255; 255 255 255; 208 162 065; 162 156 084; 117 157 122; 070 129 109; 
            254 254 254; 005 113 127; 077 117 102; 255 255 255; 255 255 255; 123 136 105; 157 142 058; 176 162 052; 199 169 047; 210 172 043; 
            254 254 254; 250 189 149; 255 255 255; 255 255 255; 255 255 255; 247 160 138; 242 132 141; 202 089 116; 151 059 102; 120 041 090;  
            254 254 254; 254 251 223; 254 241 193; 255 255 255; 255 255 255; 254 224 145; 255 255 255; 252 189 132; 208 147 137; 172 127 132; 
            254 254 254; 111 204 221; 142 208 188; 255 255 255; 255 255 255; 153 208 149; 141 171 106; 165 130 068; 158 105 037; 137 089 041; ]; 


TotRows = 10; % Rows of Colors
TotCols = 10; % Columns of Colors

ColorPix = 300; % Pixels length for each color square
BoundaryWidth = 25; % Black boundary pixel width around each color
InterSpace = 25; % Pixel space between any 2 colors
ImgBoundary = 200; % Pixel Gap between palette boundary and colors
BoundaryWidthImg = 50; % Black boundary pixel thickness around pallete 
Pallete_W = TotRows*ColorPix + 2*ImgBoundary;
Pallete_H = TotCols*ColorPix + 2*ImgBoundary;

template = zeros(Pallete_W, Pallete_H, 3, 'uint8');
template(BoundaryWidthImg:Pallete_W-BoundaryWidthImg, BoundaryWidthImg:Pallete_H-BoundaryWidthImg, :) = 255 * ones(length(BoundaryWidthImg:Pallete_W-BoundaryWidthImg), length(BoundaryWidthImg:Pallete_H-BoundaryWidthImg), 3);

count = 0;
for RowNum = 1:TotRows
    for ColNum = 1:TotCols
        
        Standard_RGB = DesignPaletteRGBs(((RowNum-1)*TotCols)+ColNum, :);
        
        Low_Y = round(ImgBoundary + (((RowNum-1)*ColorPix) + InterSpace + 1));
        Up_Y = round(ImgBoundary + (RowNum * ColorPix) - InterSpace);
        Y = Low_Y : Up_Y;
        
        Low_X = round(ImgBoundary + (((ColNum-1) * ColorPix) + InterSpace + 1));
        Up_X =  round(ImgBoundary + (ColNum * ColorPix) - InterSpace);
        X = Low_X : Up_X;
        
        for i = 1:3
            template(Y,X,i) = Standard_RGB(i) * ones(length(Y), length(X), 1);         
        end 
        
        if(Standard_RGB(1) ~= 255 || Standard_RGB(2) ~= 255 || Standard_RGB(3) ~= 255)
            template(Y(1):Y(1)+BoundaryWidth, X, :) = zeros(length(Y(1):Y(1)+BoundaryWidth), length(X), 3); 
            template(Y(end)-BoundaryWidth:Y(end), X, :) = zeros(length(Y(end)-BoundaryWidth:Y(end)), length(X), 3); 
            template(Y, X(1):X(1)+BoundaryWidth, :) = zeros(length(Y), length(X(1):X(1)+BoundaryWidth), 3); 
            template(Y, X(end)-BoundaryWidth:X(end), :) = zeros(length(Y), length(X(end)-BoundaryWidth:X(end)), 3);             
            count = count + 1;
        end
    end
end

figure; imshow(template); title('STEP 1: Template Design');

%% Dip Stick Background & Integration with Standard Palette. 

W = round((TotRows*ColorPix)); % Width of Strip Box. 
H = round(ColorPix); % Height of Strip Box

BackgroundColor = zeros(W, H, 3); % First create a filled black box then fill the area with with except a certain boundary 
BackgroundColor(BoundaryWidth:W-BoundaryWidth, BoundaryWidth:H-BoundaryWidth, :) = 255 * ones(length(BoundaryWidth:W-BoundaryWidth), length(BoundaryWidth:H-BoundaryWidth), 3);

StartIdxC = round(ImgBoundary + (3.5*ColorPix));
StartIdxR = round(ImgBoundary ); 
template(StartIdxR+1 : StartIdxR + W, StartIdxC+1 : StartIdxC + H, :) = BackgroundColor;

figure; imshow(template)

%%
Print = 255 * ones(3850, 3850, 3, 'uint8');
Print(175+1:175+size(template, 1), 175+1:175+size(template, 2), :) = template;
figure; imshow(Print);
