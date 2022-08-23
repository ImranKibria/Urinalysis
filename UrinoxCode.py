import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
    
#  True Reference Colors
TrueRGB = np.array([[243, 243, 242], [254, 248, 188],                   [230, 218, 175],    [222, 184, 135],    [207, 159, 150], [165, 120, 153], [110, 83, 125],   [52, 52, 52], 
                    [243, 243, 242], [254, 251, 223],                                       [251, 219, 217],    [246, 182, 201], [242, 131, 166], [238, 79, 131],   [52, 52, 52],  
                    [243, 243, 242], [252, 212, 174],                   [249, 169, 135],                        [242, 132, 146], [230, 111, 129], [238, 79, 131],   [52, 52, 52],  
                    [243, 243, 242], [223, 230, 125],                   [187, 216, 107],    [173, 213, 131],    [119, 190, 152], [94, 178, 169],  [000, 149, 149],  [52, 52, 52],  
                    [243, 243, 242], [245, 140, 80],  [250, 167, 86],   [254, 196, 109],    [209, 190, 99],     [137, 149, 85],  [85, 174, 146],  [000, 127, 130],  [52, 52, 52],  
                    [243, 243, 242], [250, 175, 77],  [251, 185, 78],                       [208, 162, 65],     [162, 156, 84],  [117, 157, 122], [70, 129, 109],   [52, 52, 52],  
                    [243, 243, 242], [5, 113, 127],   [77, 117, 102],   [123, 136, 105],    [157, 142, 58],     [176, 162, 52],  [199, 169, 47],  [210, 172, 43],   [52, 52, 52],  
                    [243, 243, 242], [250, 189, 149],                   [247, 160, 138],    [242, 132, 141],    [202, 89, 116],  [151, 59, 102],  [120, 41, 90],    [52, 52, 52],  
                    [243, 243, 242], [254, 251, 223], [254, 241, 193],  [254, 224, 145],                        [252, 189, 132], [208, 147, 137], [172, 127, 132],  [52, 52, 52], 
                    [243, 243, 242], [111, 204, 221], [142, 208, 188],  [153, 208, 149],    [141, 171, 106],    [165, 130, 68],  [158, 105, 37],  [137, 89, 41],    [52, 52, 52]])

########################################################## CUSTOM BUILT FUNCTIONS ##########################################################

# # Custom Image Show Function
def imgshow (Picture, scale_percent = 10, wait_time = 2000):
    src = cv.cvtColor(Picture, cv.COLOR_RGB2BGR)
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dsize = (width, height)
    output = cv.resize(src, dsize)
    cv.imshow('OpImg', output)
    cv.waitKey(wait_time)
    cv.destroyAllWindows()

def imgwarp (ipImg, Contr):
    PtsSum = np.sum(Contr, axis=2)
    tl_Idx = np.argmin(PtsSum)
    br_Idx = np.argmax(PtsSum)

    PtsDiff = np.subtract(Contr[:, :, 1], Contr[:, :, 0])
    tr_Idx = np.argmin(PtsDiff)
    bl_Idx = np.argmax(PtsDiff)

    W = Contr[br_Idx, 0, 0] - Contr[tl_Idx, 0, 0]
    #H = Contr[br_Idx, 0, 1] - Contr[tl_Idx, 0, 1]
    H = W

    pts1 = np.float32([Contr[tl_Idx, :], Contr[tr_Idx, :], Contr[bl_Idx, :], Contr[br_Idx, :]])
    pts2 = np.float32([[0,0], [W,0], [0,H], [W,H]])

    Transform = cv.getPerspectiveTransform(pts1, pts2)
    WarpImg = cv.warpPerspective(ipImg, Transform, (W, H))
    return WarpImg

def urinalysis(CaptImg):
    TotalRows = 10       
    TotalColors = 81     
    StripColorCount = 10
    imgshow(CaptImg)
    
    ########################################################## PALETTE SEGMENTATION ##########################################################

    # GrayScale Conversion
    GrayImg = cv.cvtColor(CaptImg, cv.COLOR_RGB2GRAY)
    GrayImg = cv.medianBlur(GrayImg, 3)
    imgshow(GrayImg)

    # Binarize Image
    ret, BW_Img = cv.threshold(GrayImg, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    imgshow(BW_Img)
    
    # Contour Detection
    cnts, hierarchy = cv.findContours(BW_Img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    SrtCnts = sorted(cnts, key = cv.contourArea, reverse=True) # Sort Contours By Area
    PaletteContr = SrtCnts[0]

    # Image Warping 
    PaletteOnly = imgwarp(CaptImg, PaletteContr)

    
    # Removing Boundary Lines
    ImgBound_W = 0.02 # Hardcoded according to Palette Design
    ImgBound_H = 0.02 # Hardcoded according to Palette Design
    BoundCut_W = np.round(ImgBound_W * PaletteOnly.shape[0]).astype('int')
    BoundCut_H = np.round(ImgBound_H * PaletteOnly.shape[1]).astype('int')
    PaletteOnly = PaletteOnly[BoundCut_W:-BoundCut_W, BoundCut_H:-BoundCut_H, :]
    imgshow(PaletteOnly, scale_percent=20)

    # Pallete to Gray Scale
    PalleteGray = cv.cvtColor(PaletteOnly, cv.COLOR_RGB2GRAY)
    PalleteGray = cv.medianBlur(PalleteGray, 3)
    imgshow(PalleteGray, scale_percent=20)

    # Thresholding
    ret, PalleteBW = cv.threshold(PalleteGray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)  
    imgshow(PalleteBW, scale_percent=20)
    
    # Remove Test Strip From Image
    ImgContours, hierarchy = cv.findContours(PalleteBW, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ImgContours = sorted(ImgContours, key = cv.contourArea, reverse=True)

    [x, y, w, h] = cv.boundingRect(ImgContours[0])
    cv.rectangle(PalleteBW, (x,y), (x+w, y+h), (0,0,0), -1)
    imgshow(PalleteBW, scale_percent=20)

    # Contour Detection of Calibration Colors
    ExtCntrs, hierarchy = cv.findContours(PalleteBW, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(ExtCntrs) < TotalColors:
        exit('Palette Segmentation Failed!')

    # Retain Square Contours Only    
    ExtCntrsCpy = np.copy(ExtCntrs)
    NumCntrs = len(ExtCntrs)
    PopCount = 0
    SqrContrs = np.zeros((TotalColors, 4))
    for i in range(NumCntrs):
        [x, y, w, h] = cv.boundingRect(ExtCntrs[i])
        ar = w / float(h)
        if ar < 0.85 or ar > 1.15:
            ExtCntrsCpy.pop(i - PopCount)
            PopCount += 1
        else:
            SqrContrs[i  - PopCount, :] = [x, y, w, h]
    
    # Error Generation if Palette Segmentation Fails
    if len(ExtCntrsCpy) is not TotalColors:
        exit(('Palette Segmentation Failed! Colors detected = ', len(ExtCntrsCpy)))

    PaletteOnlyCpy = PaletteOnly.copy()
    cv.drawContours(PaletteOnlyCpy, ExtCntrsCpy, -1, (255, 0, 0), thickness=25)
    imgshow(PaletteOnlyCpy, scale_percent=20)

    # Sort Contours
    SrtSqrContrs1 = np.zeros_like(SqrContrs)
    SrtSqrContrs2 = np.zeros_like(SqrContrs)
    ArgSrt = np.argsort(SqrContrs[:,1])
    for i in range(TotalColors):
        SrtSqrContrs1[i, :] = SqrContrs[ArgSrt[i], :] # All elements in a row are together

    ColPerRow = np.array([0, 8, 15, 22, 30, 39, 47, 56, 64, 72, 81]) 

    for k in range(TotalRows):
        agrsrt = np.argsort(SrtSqrContrs1[ColPerRow[k]:ColPerRow[k+1], 0])
        ColorCount = ColPerRow[k+1] - ColPerRow[k]
        for j in range(ColorCount):
            SrtSqrContrs2[ColPerRow[k]+j, :] = SrtSqrContrs1[ColPerRow[k]+agrsrt[j], :]

    ########################################################## TRANSFORMATION DEVELOPMENT ##########################################################

    # Scan Squares
    ColorBoundRatio = 0.15 # Hardcoded according to Palette Design

    ScanRGB = np.zeros((TotalColors, 3))
    for i in range(TotalColors):
        X1 = round(SrtSqrContrs2[i, 0] + (SrtSqrContrs2[i, 2] * ColorBoundRatio))
        X2 = round(SrtSqrContrs2[i, 0] + SrtSqrContrs2[i, 2]  - (SrtSqrContrs2[i, 2] * ColorBoundRatio))
        Y1 = round(SrtSqrContrs2[i, 1] + (SrtSqrContrs2[i, 3] * ColorBoundRatio))
        Y2 = round(SrtSqrContrs2[i, 1] + SrtSqrContrs2[i, 3] - (SrtSqrContrs2[i, 3] * ColorBoundRatio))
        ColorMask = PaletteOnly[Y1:Y2, X1:X2, :]

        ScanRGB[i, 0] = np.mean(ColorMask[:,:,0])
        ScanRGB[i, 1] = np.mean(ColorMask[:,:,1])
        ScanRGB[i, 2] = np.mean(ColorMask[:,:,2])

    # Finding Transformations
    m = 3
    TransRGB = np.zeros([TotalColors, 3]).astype('int')
    T_cam = np.zeros([m + 1, m, TotalRows]) # Camera Calibration matrix for each row

    for i in range(TotalRows):

        TmpScanRGB = ScanRGB[ColPerRow[i]:ColPerRow[i+1], :]
        TmpTrueRGB = TrueRGB[ColPerRow[i]:ColPerRow[i+1], :]
        ColorCnt = ColPerRow[i+1] - ColPerRow[i]

        # Chromatic Adaptation using Multivariate Linear Regression (Normal Equation) 
        TmpRGB_BefCC = np.zeros([ColorCnt, m+1])

        for j in range(ColorCnt):
            TmpRGB_BefCC[j, :] = [1, TmpScanRGB[j,0], TmpScanRGB[j,1], TmpScanRGB[j,2]]

        T_cam[:,:,i] = np.dot(np.linalg.pinv(TmpRGB_BefCC), TmpTrueRGB)

        CC = np.dot(TmpRGB_BefCC, T_cam[:,:,i])
        TransRGB[ColPerRow[i]:ColPerRow[i+1], :] = CC

    plt.xlabel('Colors in Calibration Palette')
    plt.ylabel('rms Error from Original RGB Value')
    plt.stem(np.mean(abs(np.subtract(TrueRGB, ScanRGB)), axis=1), markerfmt='o')
    plt.stem(np.mean(abs(np.subtract(TrueRGB, TransRGB)), axis=1), markerfmt='*')
    plt.show()

    # CalibrationErr = np.mean(abs(np.subtract(TrueRGBCpy, TransRGBCpy)), axis=1)
    # if(max(CalibrationErr) > 30):
    #     exit('Calibration Error Greater than Limit. Try Again!')
    
    ########################################################## STRIP SEGMENTATION ##########################################################

    # Extract Strip Area
    [StripY, StripX, StripH, StripW] = cv.boundingRect(ImgContours[0])
    Strip = PaletteOnly[StripX : StripX + StripW, StripY: StripY + StripH]
    imgshow(Strip, scale_percent=35)
    CutRatioW = 0.01 # Hardcoded according to Palette Design
    CutRatioH = 0.10 # Hardcoded according to Palette Design

    # Remove Boundary Lines
    CutPix_W = np.round(CutRatioW * StripW).astype('int')
    CutPix_H = np.round(CutRatioH * StripH).astype('int')

    Strip = PaletteOnly[StripX + CutPix_W : StripX + StripW - CutPix_W, StripY + CutPix_H: StripY + StripH - CutPix_H]
    imgshow(Strip, scale_percent=35)
    
    # Create a Hypothetical Strip 
    Height = Strip.shape[0]
    Width = Strip.shape[1]

    Interspace = np.round(Height / 10).astype('int')
    Box_X = np.int(Width/2)
    Box_Y = np.int(Interspace/2)
    Dim = np.int(Width/5)

    IdealStrip = np.zeros_like(Strip)
    for i in range(StripColorCount):
        IdealStrip[Box_Y-Dim : Box_Y+Dim, Box_X-Dim : Box_X+Dim] = (255, 255, 255)
        Box_Y = Box_Y + Interspace

    # Extract Strip Colors 
    StripColor = cv.bitwise_and(Strip, IdealStrip)
    imgshow(StripColor, scale_percent=35)
    
    ########################################################## STRIP TRANSFORMATION ##########################################################

    # Scan Patch & Apply Transform
    CalibratedStrip = np.zeros_like(Strip)
    CalibratedResult = np.zeros([StripColorCount, 3], dtype=int)
    ScanPatchColor = np.zeros([StripColorCount, 3])

    Box_Y = np.int(Interspace/2)
    for i in range(StripColorCount):

        TempPatch = StripColor[Box_Y - Dim : Box_Y + Dim, Box_X - Dim : Box_X + Dim]
        
        Patch_R = np.mean(TempPatch[:,:,0])
        Patch_G = np.mean(TempPatch[:,:,1])
        Patch_B = np.mean(TempPatch[:,:,2])

        ScanPatchColor[i, :] = [Patch_R, Patch_G, Patch_B]
        InputPatchColor = [1, Patch_R, Patch_G, Patch_B]

        PatchColor_CC =  InputPatchColor @ T_cam[:,:,i]
        for x in range(3):
            if PatchColor_CC[x] > 255:
                PatchColor_CC[x] = 255
            elif PatchColor_CC[x] < 0:
                PatchColor_CC[x] = 0
        
        CalibratedResult[i, :] = np.round(PatchColor_CC)
        CalibratedStrip[Box_Y - Dim : Box_Y + Dim, Box_X - Dim : Box_X + Dim] = CalibratedResult[i, :]
        Box_Y = Box_Y + Interspace

    plt.subplot(1, 2, 1)
    plt.imshow(StripColor)
    plt.subplot(1, 2, 2)
    plt.imshow(CalibratedStrip)
    plt.show()
    
    ################################################## STEP 5: COLOR MATCHING ##################################################
    Report = np.zeros((StripColorCount))
    
    print('---------- Medical Report ----------')
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    for i in range(StripColorCount):
        ColorsPerRow = ColPerRow[i+1]-ColPerRow[i]
        RMSerr = np.zeros([ColorsPerRow])
        for j in range(ColorsPerRow):
            RMSerr[j] = rmse(CalibratedResult[i], TrueRGB[ColPerRow[i] + j])

        IdxMatch = np.argmin(RMSerr[1:ColorsPerRow-1]) # White & Black Patches should not be compared! 
        Report[i] = IdxMatch + 1 
        
        if i == 0:
            if IdxMatch == 0:   print('LEUKOCYTES:  NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('LEUKOCYTES:  15 TRACE.               Match Index:', IdxMatch)
            elif IdxMatch == 2: print('LEUKOCYTES:  POSITIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 3: print('LEUKOCYTES:  TRACE 75.               Match Index:', IdxMatch)
            elif IdxMatch == 4: print('LEUKOCYTES:  MODERATE 125.           Match Index:', IdxMatch)
            elif IdxMatch == 5: print('LEUKOCYTES:  LARGE 500.              Match Index:', IdxMatch)

        elif i == 1:
            if IdxMatch == 0:   print('NITRITE:     NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('NITRITE:     POSITIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 2: print('NITRITE:     POSITIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 3: print('NITRITE:     POSITIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 4: print('NITRITE:     POSITIVE.               Match Index:', IdxMatch)

        elif i == 2:
            if IdxMatch == 0:   print('UROILINOGEN: 3.2 NORMAL.             Match Index:', IdxMatch)
            elif IdxMatch == 1: print('UROILINOGEN: 16 NORMAL.              Match Index:', IdxMatch)
            elif IdxMatch == 2: print('UROILINOGEN: 32 +.                   Match Index:', IdxMatch)
            elif IdxMatch == 3: print('UROILINOGEN: 64 ++.                  Match Index:', IdxMatch)
            elif IdxMatch == 4: print('UROILINOGEN: 128 +++.                Match Index:', IdxMatch)

        elif i == 3:
            if IdxMatch == 0:   print('PROTEIN:     NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('PROTEIN:     TRACE.                  Match Index:', IdxMatch)
            elif IdxMatch == 2: print('PROTEIN:     0.3 +.                  Match Index:', IdxMatch)
            elif IdxMatch == 3: print('PROTEIN:     1.0 ++.                 Match Index:', IdxMatch)
            elif IdxMatch == 4: print('PROTEIN:     3.0 +++.                Match Index:', IdxMatch)
            elif IdxMatch == 5: print('PROTEIN:     >= 20 ++++.             Match Index:', IdxMatch)

        elif i == 4:
            if IdxMatch == 0:   print('PH:          5.0.                    Match Index:', IdxMatch)
            elif IdxMatch == 1: print('PH:          6.0.                    Match Index:', IdxMatch)
            elif IdxMatch == 2: print('PH:          6.5.                    Match Index:', IdxMatch)
            elif IdxMatch == 3: print('PH:          7.0.                    Match Index:', IdxMatch)
            elif IdxMatch == 4: print('PH:          7.5.                    Match Index:', IdxMatch)
            elif IdxMatch == 5: print('PH:          8.0.                    Match Index:', IdxMatch)
            elif IdxMatch == 6: print('PH:          8.5.                    Match Index:', IdxMatch)

        elif i == 5:
            if IdxMatch == 0:   print('BLOOD:       NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('BLOOD:       NON-HEMOLYZED TRACE.    Match Index:', IdxMatch)
            elif IdxMatch == 2: print('BLOOD:       10 TRACE.               Match Index:', IdxMatch)
            elif IdxMatch == 3: print('BLOOD:       25 SMALL.               Match Index:', IdxMatch)
            elif IdxMatch == 4: print('BLOOD:       80 MODERATE.            Match Index:', IdxMatch)
            elif IdxMatch == 5: print('BLOOD:       200 LARGE.              Match Index:', IdxMatch)

        elif i == 6:
            if IdxMatch == 0:   print('SP. GRAVITY: 1.000.                  Match Index:', IdxMatch)
            elif IdxMatch == 1: print('SP. GRAVITY: 1.005.                  Match Index:', IdxMatch)
            elif IdxMatch == 2: print('SP. GRAVITY: 1.010.                  Match Index:', IdxMatch)
            elif IdxMatch == 3: print('SP. GRAVITY: 1.015.                  Match Index:', IdxMatch)
            elif IdxMatch == 4: print('SP. GRAVITY: 1.020.                  Match Index:', IdxMatch)
            elif IdxMatch == 5: print('SP. GRAVITY: 1.025.                  Match Index:', IdxMatch)
            elif IdxMatch == 6: print('SP. GRAVITY: 1.030.                  Match Index:', IdxMatch)

        elif i == 7:
            if IdxMatch == 0:   print('KETONE:      NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('KETONE:      TRACE 0.5 .              Match Index:', IdxMatch)
            elif IdxMatch == 2: print('KETONE:      SMALL 1.5 .              Match Index:', IdxMatch)
            elif IdxMatch == 3: print('KETONE:      MODERATE 4.0 .           Match Index:', IdxMatch)
            elif IdxMatch == 4: print('KETONE:      LARGE 8.0.              Match Index:', IdxMatch)
            elif IdxMatch == 5: print('KETONE:      LARGE 16.0.             Match Index:', IdxMatch)

        elif i == 8:
            if IdxMatch == 0:   print('BILIRUBIN:   NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('BILIRUBIN:   NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 2: print('BILIRUBIN:   NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 3: print('BILIRUBIN:   SMALL 17.               Match Index:', IdxMatch)
            elif IdxMatch == 4: print('BILIRUBIN:   MODERATE 50.            Match Index:', IdxMatch)
            elif IdxMatch == 5: print('BILIRUBIN:   LARGE 100.              Match Index:', IdxMatch)

        elif i == 9:
            if IdxMatch == 0:   print('GLUCOSE:     NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('GLUCOSE:     NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 2: print('GLUCOSE:     5 TRACE.                Match Index:', IdxMatch)
            elif IdxMatch == 3: print('GLUCOSE:     15 +.                   Match Index:', IdxMatch)
            elif IdxMatch == 4: print('GLUCOSE:     30 ++.                  Match Index:', IdxMatch)
            elif IdxMatch == 5: print('GLUCOSE:     60 +++.                 Match Index:', IdxMatch)
            elif IdxMatch == 6: print('GLUCOSE:     110 ++++.               Match Index:', IdxMatch)

    return ScanPatchColor, CalibratedResult, Report
########################################### MAIN CODE STARTS FROM HERE ###########################################

x = 'Oppo.jpg'
CaptImg = cv.cvtColor(cv.imread(x), cv.COLOR_BGR2RGB)
CapturedRGBs, CorrectedRGBs, RefColorIdx = urinalysis(CaptImg)