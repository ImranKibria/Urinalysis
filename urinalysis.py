import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import xlwt 
from xlwt import Workbook

#  True Reference Colors
TrueRGB = np.array([[255, 255, 255], [254, 248, 188], [230, 218, 175], [222, 184, 135], [207, 159, 150], [165, 120, 153], [110, 83, 125], 
            [255, 255, 255], [254, 251, 223], [251, 219, 217], [255, 192, 203], [246, 182, 201], [255, 105, 180], [238, 79, 131], 
            [255, 255, 255], [252, 212, 174], [249, 169, 135], [242, 132, 146], [230, 111, 129], [238, 79, 131], 
            [255, 255, 255], [223, 230, 125], [187, 216, 107], [173, 213, 131], [119, 190, 152], [94, 178, 169], [000, 149, 149], 
            [255, 255, 255], [245, 140, 80], [250, 167, 86], [254, 196, 109], [209, 190, 99], [137, 149, 85], [85, 174, 146], [000, 127, 130], 
            [255, 255, 255], [250, 175, 77], [251, 185, 78], [208, 162, 65], [162, 156, 84], [117, 157, 122], [70, 129, 109], 
            [255, 255, 255], [5, 113, 127],  [77, 117, 102], [123, 136, 105], [157, 142, 58], [176, 162, 52], [199, 169, 47], [210, 172, 43], 
            [255, 255, 255], [250, 189, 149], [247, 160, 138], [242, 132, 141], [202, 89, 116], [151, 59, 102], [120, 41, 90], 
            [255, 255, 255], [254, 251, 223], [254, 241, 193], [254, 224, 145], [252, 189, 132], [208, 147, 137], [172, 127, 132],
            [255, 255, 255], [111, 204, 221], [142, 208, 188], [153, 208, 149], [141, 171, 106], [165, 130, 68], [158, 105, 37], [137, 89, 41]])

########################################################## CUSTOM BUILT FUNCTIONS ##########################################################

def imgwarp (ipImg, Contr):
    PtsSum = np.sum(Contr, axis=2)
    tl_Idx = np.argmin(PtsSum)
    br_Idx = np.argmax(PtsSum)

    PtsDiff = np.subtract(Contr[:, :, 1], Contr[:, :, 0])
    tr_Idx = np.argmin(PtsDiff)
    bl_Idx = np.argmax(PtsDiff)

    H = Contr[br_Idx, 0, 1] - Contr[tl_Idx, 0, 1]
    W = Contr[br_Idx, 0, 0] - Contr[tl_Idx, 0, 0]

    pts1 = np.float32([Contr[tl_Idx, :], Contr[tr_Idx, :], Contr[bl_Idx, :], Contr[br_Idx, :]])
    pts2 = np.float32([[0,0], [W,0], [0,H], [W,H]])

    Transform = cv.getPerspectiveTransform(pts1, pts2)
    WarpImg = cv.warpPerspective(ipImg, Transform, (W, H))
    return WarpImg

def urinalysis(CaptImg):
    TotalRows = 10       
    TotalColors = 72     
    StripColorCount = 10

    ########################################################## PALETTE SEGMENTATION ##########################################################

    # GrayScale Conversion
    GrayImg = cv.cvtColor(CaptImg, cv.COLOR_RGB2GRAY)
    GrayImg = cv.medianBlur(GrayImg, 3)
    # ImgDisp(GrayImg)

    # Binarize Image
    ret, BW_Img = cv.threshold(GrayImg, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # ImgDisp(BW_Img)

    # Contour Detection
    cnts, hierarchy = cv.findContours(BW_Img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    SrtCnts = sorted(cnts, key = cv.contourArea, reverse=True)
    PaletteContr = SrtCnts[0]

    # Image Warping & Removing Boundary Lines
    PaletteOnly = imgwarp(CaptImg, PaletteContr)
    
    ImgBound_W = 0.02 # Hardcoded according to Palette Design
    ImgBound_H = 0.02 # Hardcoded according to Palette Design
    BoundCut_W = np.round(ImgBound_W * PaletteOnly.shape[0]).astype('int')
    BoundCut_H = np.round(ImgBound_H * PaletteOnly.shape[1]).astype('int')
    PaletteOnly = PaletteOnly[BoundCut_W:-BoundCut_W, BoundCut_H:-BoundCut_H, :]
    # ImgDisp(PaletteOnly)

    # Pallete to Gray Scale
    PalleteGray = cv.cvtColor(PaletteOnly, cv.COLOR_RGB2GRAY)
    # ImgDisp(PalleteGray)

    # Thresholding
    ret, PalleteBW = cv.threshold(PalleteGray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)  
    PalleteGray = cv.medianBlur(PalleteGray, 3)
    # ImgDisp(PalleteBW)

    # Remove Test Strip 
    ImgContours, hierarchy = cv.findContours(PalleteBW, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ImgContours = sorted(ImgContours, key = cv.contourArea, reverse=True)

    [x, y, w, h] = cv.boundingRect(ImgContours[0])
    cv.rectangle(PalleteBW, (x,y), (x+w, y+h), (0,0,0), -1)
    # ImgDisp(PalleteBW)

    # Contour Detection
    ExtCntrs, hierarchy = cv.findContours(PalleteBW, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(ExtCntrs) < TotalColors:
        exit('Palette Segmentation Failed!')

    # Retain Squares Only    
    ExtCntrsCpy = ExtCntrs.copy()
    NumCntrs = len(ExtCntrs)
    PopCount = 0
    SqrContrs = np.zeros((TotalColors, 4))
    for i in range(NumCntrs):
        [x, y, w, h] = cv.boundingRect(ExtCntrs[i])
        ar = w / float(h)
        if ar < 0.85 or ar > 1.15:
            ExtCntrsCpy.pop(i  - PopCount)
            PopCount += 1
        else:
            SqrContrs[i  - PopCount, :] = [x, y, w, h]

    print('Template Colors Detected = ', len(ExtCntrsCpy))
    # Error Generation if Palette Segmentation Fails
    if len(ExtCntrsCpy) is not TotalColors:
        exit('Palette Segmentation Failed!')

    PaletteOnlyCpy = PaletteOnly.copy()
    cv.drawContours(PaletteOnlyCpy, ExtCntrsCpy, -1, (255, 0, 0), thickness=2)
    ImgDisp(PaletteOnlyCpy)

    # Sort Contours
    SrtSqrContrs1 = np.zeros_like(SqrContrs)
    SrtSqrContrs2 = np.zeros_like(SqrContrs)
    ArgSrt = np.argsort(SqrContrs[:,1])
    for i in range(TotalColors):
        SrtSqrContrs1[i, :] = SqrContrs[ArgSrt[i], :] # All elements in a row are together

    ColPerRow = np.array([0, 7, 14, 20, 27, 35, 42, 50, 57, 64, 72]) # Hardcoded

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
        # ImgDisp(ColorMask, scale_percent=500, wait_time=250)

        ScanRGB[i, 0] = np.mean(ColorMask[:,:,0])
        ScanRGB[i, 1] = np.mean(ColorMask[:,:,1])
        ScanRGB[i, 2] = np.mean(ColorMask[:,:,2])

    # Finding Transformations
    m = 3
    TransRGB = np.zeros([TotalColors, 3]).astype('int')
    T_light = np.zeros([3, 3, TotalRows])
    T_cam = np.zeros([m, 3, TotalRows])

    for i in range(TotalRows):

        TmpScanRGB = ScanRGB[ColPerRow[i]:ColPerRow[i+1], :]
        TmpTrueRGB = TrueRGB[ColPerRow[i]:ColPerRow[i+1], :]

        # White Balancing
        ColorCnt = ColPerRow[i+1] - ColPerRow[i]
        RefTag = 0

        RW_gt = TmpTrueRGB[RefTag, 0] # White Patch Reference Coordinates
        GW_gt = TmpTrueRGB[RefTag, 1]
        BW_gt = TmpTrueRGB[RefTag, 2]

        RW_m = TmpScanRGB[RefTag,0] # White Patch Measured Coordinates
        GW_m = TmpScanRGB[RefTag,1]
        BW_m = TmpScanRGB[RefTag,2]

        T_light[:,:,i] = [[(RW_gt/RW_m), 0, 0], [0, (GW_gt/GW_m), 0], [0, 0, (BW_gt/BW_m)]] # 3x3 transform matrix from Research Paper

        TmpWB_RGB = np.dot(TmpScanRGB, T_light[:,:,i])

        # Chromatic Adaptation
        TmpRGB_BefCC = np.zeros([ColorCnt,m])

        for j in range(ColorCnt):
            if m == 3:
                TmpRGB_BefCC[j, :] = [TmpWB_RGB[j,0], TmpWB_RGB[j,1], TmpWB_RGB[j,2]]
            if m == 6:
                TmpRGB_BefCC[j, :] = [TmpWB_RGB[j,0], TmpWB_RGB[j,1], TmpWB_RGB[j,2], TmpWB_RGB[j,0]*TmpWB_RGB[j,1], TmpWB_RGB[j,1]*TmpWB_RGB[j,2], TmpWB_RGB[j,0]*TmpWB_RGB[j,2]]
            if m == 8:
                TmpRGB_BefCC[j, :] = [1, TmpWB_RGB[j,0], TmpWB_RGB[j,1], TmpWB_RGB[j,2], TmpWB_RGB[j,0]*TmpWB_RGB[j,1], TmpWB_RGB[j,1]*TmpWB_RGB[j,2], TmpWB_RGB[j,0]*TmpWB_RGB[j,2], TmpWB_RGB[j,0]*TmpWB_RGB[j,1]*TmpWB_RGB[j,2]]

        T_cam[:,:,i] = np.dot(np.linalg.pinv(TmpRGB_BefCC), TmpTrueRGB)

        CC = np.dot(TmpRGB_BefCC, T_cam[:,:,i])
        TransRGB[ColPerRow[i]:ColPerRow[i+1], :] = CC

    plt.xlabel('Colors in Calibration Palette')
    plt.ylabel('rms Error from Original RGB Value')
    plt.stem(np.mean(abs(np.subtract(TrueRGB, TransRGB)), axis=1), markerfmt='o')
    plt.stem(np.mean(abs(np.subtract(TrueRGB, ScanRGB)), axis=1), markerfmt='*')
    plt.show()

    # CalibrationErr = np.mean(abs(np.subtract(TrueRGBCpy, TransRGBCpy)), axis=1)
    # if(max(CalibrationErr) > 30):
    #     exit('Calibration Error Greater than Limit. Try Again!')
    
    ########################################################## STRIP SEGMENTATION ##########################################################

    # Crop Strip Area
    [StripY, StripX, StripH, StripW] = cv.boundingRect(ImgContours[0])
    CutRatioW = 0.01 # Hardcoded according to Palette Design
    CutRatioH = 0.10 # Hardcoded according to Palette Design

    # Remove Boundary Lines
    CutPix_W = np.round(CutRatioW * StripW).astype('int')
    CutPix_H = np.round(CutRatioH * StripH).astype('int')

    Strip = PaletteOnly[StripX + CutPix_W : StripX + StripW - CutPix_W, StripY + CutPix_H: StripY + StripH - CutPix_H]
    # ImgDisp(Strip)

    # Create a Hypothetical Strip 
    Height = Strip.shape[0]
    Width = Strip.shape[1]

    Interspace = np.round(Height / StripColorCount).astype('int')
    Box_X = np.int(Width/2)
    Box_Y = np.int(Interspace/2)
    Dim = np.int(Width/4)

    IdealStrip = np.zeros_like(Strip)
    for i in range(StripColorCount):
        IdealStrip[Box_Y-Dim : Box_Y+Dim, Box_X-Dim : Box_X+Dim] = (255, 255, 255)
        Box_Y = Box_Y + Interspace
    # ImgDisp(IdealStrip)

    # Extract Strip Colors 
    StripColor = cv.bitwise_and(Strip, IdealStrip)
    ImgDisp(StripColor)

    ########################################################## STRIP TRANSFORMATION ##########################################################

    # Scan Patch & Apply Transform
    CalibratedStrip = np.zeros_like(Strip)
    CalibratedResult = np.zeros([StripColorCount, 3], dtype=int)

    Box_Y = np.int(Interspace/2)
    for i in range(StripColorCount):

        TempPatch = StripColor[Box_Y - Dim : Box_Y + Dim, Box_X - Dim : Box_X + Dim]
        imgshow(TempPatch, scale_percent=500, wait_time=1000)
        Patch_R = np.mean(TempPatch[:,:,0])
        Patch_G = np.mean(TempPatch[:,:,1])
        Patch_B = np.mean(TempPatch[:,:,2])

        ScanPatchColor = [Patch_R, Patch_G, Patch_B]
        PatchColor_WB = ScanPatchColor @ T_light[:,:,i]

        if m == 3:
            PatchColor_WB = np.array([PatchColor_WB[0], PatchColor_WB[1], PatchColor_WB[2]])
        if m == 6:
            PatchColor_WB = np.array([PatchColor_WB[0], PatchColor_WB[1], PatchColor_WB[2], PatchColor_WB[0]*PatchColor_WB[1], PatchColor_WB[1]*PatchColor_WB[2], PatchColor_WB[0]*PatchColor_WB[2]])
        if m == 8:
            PatchColor_WB = np.array([1, PatchColor_WB[0], PatchColor_WB[1], PatchColor_WB[2], PatchColor_WB[0]*PatchColor_WB[1], PatchColor_WB[1]*PatchColor_WB[2], PatchColor_WB[0]*PatchColor_WB[2], PatchColor_WB[0]*PatchColor_WB[1]*PatchColor_WB[2]])

        PatchColor_CC =  PatchColor_WB @ T_cam[:,:,i]
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
    Report = np.zeros(StripColorCount)
    
    print('---------- Medical Report ----------')
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    for i in range(StripColorCount):
        ColorsPerRow = ColPerRow[i+1]-ColPerRow[i]
        RMSerr = np.zeros([ColorsPerRow])
        for j in range(ColorsPerRow):
            RMSerr[j] = rmse(CalibratedResult[i], TrueRGB[ColPerRow[i] + j])

        IdxMatch = np.argmin(RMSerr[1:ColorsPerRow]) # White Patches should not be compared!
        Report[i] = IdxMatch+2
        
        if i == 0:
            if IdxMatch == 0:   print('LEUKOCYTES:  NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('LEUKOCYTES:  15 TRACE.               Match Index:', IdxMatch)
            elif IdxMatch == 2: print('LEUKOCYTES:  POSITIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 3: print('LEUKOCYTES:  75 TRACE.               Match Index:', IdxMatch)
            elif IdxMatch == 4: print('LEUKOCYTES:  125 MODERATE.           Match Index:', IdxMatch)
            elif IdxMatch == 5: print('LEUKOCYTES:  500 LARGE.              Match Index:', IdxMatch)

        elif i == 1:
            if IdxMatch == 0:   print('NITRITE:     NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('NITRITE:     POSITIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 2: print('NITRITE:     POSITIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 3: print('NITRITE:     POSITIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 4: print('NITRITE:     POSITIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 5: print('NITRITE:     POSITIVE.               Match Index:', IdxMatch)

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
            elif IdxMatch == 1: print('PH:          5.5.                    Match Index:', IdxMatch)
            elif IdxMatch == 2: print('PH:          6.0.                    Match Index:', IdxMatch)
            elif IdxMatch == 3: print('PH:          6.5.                    Match Index:', IdxMatch)
            elif IdxMatch == 4: print('PH:          7.0.                    Match Index:', IdxMatch)
            elif IdxMatch == 5: print('PH:          7.5.                    Match Index:', IdxMatch)
            elif IdxMatch == 6: print('PH:          8.0.                    Match Index:', IdxMatch)
            elif IdxMatch == 7: print('PH:          8.5.                    Match Index:', IdxMatch)

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
            elif IdxMatch == 1: print('KETONE:      0.5 TRACE.              Match Index:', IdxMatch)
            elif IdxMatch == 2: print('KETONE:      1.5 SMALL.              Match Index:', IdxMatch)
            elif IdxMatch == 3: print('KETONE:      4.0 MODERATE.           Match Index:', IdxMatch)
            elif IdxMatch == 4: print('KETONE:      8.0 LARGE.              Match Index:', IdxMatch)
            elif IdxMatch == 5: print('KETONE:      16.0 LARGE.             Match Index:', IdxMatch)

        elif i == 8:
            if IdxMatch == 0:   print('BILIRUBIN:   NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('BILIRUBIN:   NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 2: print('BILIRUBIN:   NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 3: print('BILIRUBIN:   17 SMALL.               Match Index:', IdxMatch)
            elif IdxMatch == 4: print('BILIRUBIN:   50 MODERATE.            Match Index:', IdxMatch)
            elif IdxMatch == 5: print('BILIRUBIN:   100 LARGE.              Match Index:', IdxMatch)

        elif i == 9:
            if IdxMatch == 0:   print('GLUCOSE:     NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 1: print('GLUCOSE:     NEGATIVE.               Match Index:', IdxMatch)
            elif IdxMatch == 2: print('GLUCOSE:     5 TRACE.                Match Index:', IdxMatch)
            elif IdxMatch == 3: print('GLUCOSE:     15 +.                   Match Index:', IdxMatch)
            elif IdxMatch == 4: print('GLUCOSE:     30 ++.                  Match Index:', IdxMatch)
            elif IdxMatch == 5: print('GLUCOSE:     60 +++.                 Match Index:', IdxMatch)
            elif IdxMatch == 6: print('GLUCOSE:     110 ++++.               Match Index:', IdxMatch)

    return Report
########################################### MAIN CODE STARTS FROM HERE ###########################################

# wb = Workbook() 
# sheet1 = wb.add_sheet('Sheet 1') 
# style = xlwt.easyxf('font: bold 1') 

# sheet1.write(0, 0, 'NAME', style) 
# sheet1.write(0, 1, 'AGE', style) 
# sheet1.write(0, 2, 'WEIGHT', style) 
# sheet1.write(0, 3, 'DATE & TIME', style) 

# sheet1.write(0, 5, 'LEUKOCYTES', style) 
# sheet1.write(0, 6, 'NITRITE', style) 
# sheet1.write(0, 7, 'UROILINOGEN', style) 
# sheet1.write(0, 8, 'PROTEIN', style) 
# sheet1.write(0, 9, 'PH', style) 
# sheet1.write(0, 10, 'BLOOD', style) 
# sheet1.write(0, 11, 'SP. GRAVITY', style) 
# sheet1.write(0, 12, 'KETONE', style) 
# sheet1.write(0, 13, 'BILIRUBIN', style) 
# sheet1.write(0, 14, 'GLUCOSE', style) 

# PatientCount = np.int(input('Enter Total Number of Patients = '))

# while PatientCount > 0:
    
#     DataTrue = 0
#     while (DataTrue != '1'):

#         Name = input('Enter Patient Full Name:     ')
#         Age =  input('Enter Patient Age (Yrs):     ')
#         Weight = input('Enter Patient Weight (Kgs):  ')
        
#         now = datetime.now()
#         dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        
#         print ('----------Patient Record---------- \nTest Date & Time: ', dt_string, '\nPatient Name:   ', Name, '\nPatient Age:    ', Age, '\nPatient Weight: ', Weight)  
#         DataTrue = input ('Press "1" to Continue & "0" to Enter Data Again: ')
        
#         if DataTrue == '1':
#             sheet1.write(1, 0, Name) 
#             sheet1.write(1, 1, Age) 
#             sheet1.write(1, 2, Weight) 
            
            
#     x = input('Enter Image File Name Patient (file.ext) : ')
#     CaptImg = cv.cvtColor(cv.imread(x), cv.COLOR_BGR2RGB)
#     Report = urinalysis(CaptImg)
    
#     sheet1.write(1, 5, Report[0]) 
#     sheet1.write(1, 6, Report[1]) 
#     sheet1.write(1, 7, Report[2]) 
#     sheet1.write(1, 8, Report[3]) 
#     sheet1.write(1, 9, Report[4]) 
#     sheet1.write(1, 10, Report[5]) 
#     sheet1.write(1, 11, Report[6]) 
#     sheet1.write(1, 12, Report[7]) 
#     sheet1.write(1, 13, Report[8]) 
#     sheet1.write(1, 14, Report[9]) 

#     PatientCount -= 1

# wb.save('xlwt example.xls') 



