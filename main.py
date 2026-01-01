from tkinter import *
from tkinter import ttk
from tkinter.ttk import *
from tkinter import filedialog
from datetime import datetime
from PIL import ImageTk, Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import xlwt 
from xlwt import Workbook
import time
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
import xlwt 
from xlwt import Workbook

############################################### CUSTOM FUNCTIONS START HERE ###############################################

def SaveButton():
    global Count, wb, sheet1
    Count += 1

    sheet1.write(Count, 0, name_entry.get()) 
    sheet1.write(Count, 1, age_entry.get()) 
    sheet1.write(Count, 2, gender_entry.get()) 
    
    sheet1.write(Count, 4, image_path_entry.get()) 

    for i, entry in enumerate(test_entries):
        sheet1.write(Count, 6+i, entry.get())  
            
    wb.save('Urinox.xls')
    
def NewButton():
    for i, entry in enumerate(test_entries):
        entry.delete(0, END) 
    
    date_entry.delete(0, END)
    image_path_entry.delete(0, END)
    
    name_entry.delete(0, END)
    age_entry.delete(0, END)
    gender_entry.delete(0, END)
    
    fig.clear()

    # Recreate a fresh axes
    global ax
    ax = fig.add_subplot(111)
    ax.axis("off")

    # Force redraw
    canvas.draw()
    canvas.flush_events()
    
    progress['value'] = 0

def BrowseButton():
    filename = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif"),
            ("All files", "*.*")
        ]
    )

    if not filename:
        return

    # Update entry field
    image_path_entry.delete(0, END)
    image_path_entry.insert(0, filename)

    # Load and display image
    img = cv.imread(filename)
    if img is None:
        return

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    ax.clear()
    ax.imshow(img)
    ax.axis("off")

    canvas.draw_idle()

def ProcessButton():
    filename = image_path_entry.get()
    PatientImg = (cv.cvtColor(cv.imread(filename),cv.COLOR_BGR2RGB))
    
    def handle_report(report):
        DispResult(report)
        
    urinalysis(PatientImg, callback=handle_report)
        
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

def imgwarp(ipImg, Contr):
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
   
def urinalysis(CaptImg, callback):
    TotalRows = 10       
    TotalColors = 72     
    StripColorCount = 10
    
    stage_results = {'CapturedImage' : [CaptImg]}

    # STEP 1: PALETTE SEGMENTATION 
    def chart_segmentation():
        CaptImg = stage_results['CapturedImage'][0]
        
        # GrayScale Conversion
        GrayImg = cv.cvtColor(CaptImg, cv.COLOR_RGB2GRAY)
        GrayImg = cv.medianBlur(GrayImg, 3)

        # Binarize Image
        ret, BW_Img = cv.threshold(GrayImg, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

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

        # Pallete to Gray Scale
        PalleteGray = cv.cvtColor(PaletteOnly, cv.COLOR_RGB2GRAY)

        # Thresholding
        ret, PalleteBW = cv.threshold(PalleteGray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)  
        PalleteGray = cv.medianBlur(PalleteGray, 3)

        # Remove Test Strip 
        ImgContours, hierarchy = cv.findContours(PalleteBW, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ImgContours = sorted(ImgContours, key = cv.contourArea, reverse=True)

        [x, y, w, h] = cv.boundingRect(ImgContours[0])
        cv.rectangle(PalleteBW, (x,y), (x+w, y+h), (0,0,0), -1)

        # Contour Detection
        ExtCntrs, hierarchy = cv.findContours(PalleteBW, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(ExtCntrs) < TotalColors:
            exit('Palette Segmentation Failed!')

        # Retain Squares Only    
        ExtCntrsCpy = ExtCntrs[:]
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

        # print('Template Colors Detected = ', len(ExtCntrsCpy))

        if len(ExtCntrsCpy) is not TotalColors:
            exit('Palette Segmentation Failed!')

        PaletteOnlyCpy = PaletteOnly.copy()
        cv.drawContours(PaletteOnlyCpy, ExtCntrsCpy, -1, (255, 0, 0), thickness=2)
            
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

        stage_results['ChartSegmentation'] = [SrtSqrContrs2, PaletteOnly, ColPerRow, ImgContours, PaletteOnly]
        ##################################### IMAGE DISPLAY #####################################
        
        # Show intermediate result
        ax.clear()
        ax.imshow(PaletteOnlyCpy)
        ax.axis("off")

        canvas.draw_idle()

        # Update progress bar
        progress['value'] = 25
        root.update_idletasks()

        root.after(500, color_calibration)

    # STEP 2: TRANSFORMATION DEVELOPMENT 
    def color_calibration():

        SrtSqrContrs2 = stage_results['ChartSegmentation'][0]
        PaletteOnly = stage_results['ChartSegmentation'][1]
        ColPerRow = stage_results['ChartSegmentation'][2]
        
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

        # CalibrationErr = np.mean(abs(np.subtract(TrueRGBCpy, TransRGBCpy)), axis=1)
        # if(max(CalibrationErr) > 30):
        #     exit('Calibration Error Greater than Limit. Try Again!')
        
        stage_results['ColorCalibration'] = [T_light, T_cam, m]
    
        ##################################### IMAGE DISPLAY #####################################

        ax.clear()

        y1 = np.mean(np.abs(TrueRGB - TransRGB), axis=1)
        y2 = np.mean(np.abs(TrueRGB - ScanRGB), axis=1)

        ax.stem(y1, markerfmt='o', linefmt='C0-', basefmt=' ')
        ax.stem(y2, markerfmt='*', linefmt='C1-', basefmt=' ')

        ax.set_title("Mean Absolute RGB Error")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Error")
        ax.grid(True)

        canvas.draw_idle()

        progress['value'] = 50
        root.update_idletasks()

        root.after(500, strip_segmentation)
     
    # STEP 3: STRIP SEGMENTATION 
    def strip_segmentation():

        ImgContours = stage_results['ChartSegmentation'][3]
        PaletteOnly = stage_results['ChartSegmentation'][4]
        
        # Crop Strip Area
        [StripY, StripX, StripH, StripW] = cv.boundingRect(ImgContours[0])
        CutRatioW = 0.01 # Hardcoded according to Palette Design
        CutRatioH = 0.10 # Hardcoded according to Palette Design

        # Remove Boundary Lines
        CutPix_W = np.round(CutRatioW * StripW).astype('int')
        CutPix_H = np.round(CutRatioH * StripH).astype('int')

        Strip = PaletteOnly[StripX + CutPix_W : StripX + StripW - CutPix_W, StripY + CutPix_H: StripY + StripH - CutPix_H]

        # Create a Hypothetical Strip 
        Height = Strip.shape[0]
        Width = Strip.shape[1]

        Interspace = np.round(Height / StripColorCount).astype('int')
        Box_X = np.int32(Width/2)
        Box_Y = np.int32(Interspace/2)
        Dim = np.int32(Width/4)

        IdealStrip = np.zeros_like(Strip)
        for i in range(StripColorCount):
            IdealStrip[Box_Y-Dim : Box_Y+Dim, Box_X-Dim : Box_X+Dim] = (255, 255, 255)
            Box_Y = Box_Y + Interspace
        
        # Extract Strip Colors 
        StripColor = cv.bitwise_and(Strip, IdealStrip)
        
        stage_results['StripSegmentation'] = [Strip, Interspace, StripColor, Dim, Box_X]
        ##################################### IMAGE DISPLAY #####################################

        ax.clear()

        ax.imshow(StripColor)
        ax.axis("off")  # hide axes

        canvas.draw_idle()

        progress['value'] = 75
        root.update_idletasks()

        root.after(500, strip_calibration)
    
    # STEP 4: STRIP TRANSFORMATION 
    def strip_calibration():

        Strip = stage_results['StripSegmentation'][0]
        Interspace = stage_results['StripSegmentation'][1] 
        StripColor = stage_results['StripSegmentation'][2]
        Dim = stage_results['StripSegmentation'][3]
        Box_X = stage_results['StripSegmentation'][4]
        T_light = stage_results['ColorCalibration'][0]
        T_cam = stage_results['ColorCalibration'][1]
        m = stage_results['ColorCalibration'][2]
        
        # Scan Patch & Apply Transform
        CalibratedStrip = np.zeros_like(Strip)
        CalibratedResult = np.zeros([StripColorCount, 3], dtype=int)

        Box_Y = np.int32(Interspace/2)
        for i in range(StripColorCount):

            TempPatch = StripColor[Box_Y - Dim : Box_Y + Dim, Box_X - Dim : Box_X + Dim]

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

        stage_results['StripCalibration'] = [CalibratedResult]
        ##################################### IMAGE DISPLAY #####################################

        ax.clear()  # single ax from original Figure
        ax.axis("off")

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(StripColor)
        ax1.axis("off")
        ax1.set_title("Original Strip")

        ax2.imshow(CalibratedStrip)
        ax2.axis("off")
        ax2.set_title("Calibrated Strip")

        canvas.draw_idle()

        progress['value'] = 100
        root.update_idletasks()

        root.after(500, color_matching)
    
    # STEP 5: COLOR MATCHING 
    def color_matching():
        Report = np.zeros(StripColorCount)
        ColPerRow = stage_results['ChartSegmentation'][2]
        CalibratedResult = stage_results['StripCalibration'][0]

        def rmse(predictions, targets):
            return np.sqrt(((predictions - targets) ** 2).mean())

        for i in range(StripColorCount):
            ColorsPerRow = ColPerRow[i+1]-ColPerRow[i]
            RMSerr = np.zeros([ColorsPerRow])
            for j in range(ColorsPerRow):
                RMSerr[j] = rmse(CalibratedResult[i], TrueRGB[ColPerRow[i] + j])

            IdxMatch = np.argmin(RMSerr[1:ColorsPerRow]) # White Patches should not be compared!
            Report[i] = IdxMatch
            
        stage_results['ColorMatching'] = [Report]
        callback(Report)  # call the function passed from ProcessButton

    chart_segmentation()
    
def DispResult(Report):
    # Update Date & Time
    now = datetime.now()
    d_string = now.strftime("%d/%m/%Y    %H:%M:%S")
    
    date_entry.delete(0, END)
    date_entry.insert(0, d_string)

    # Mapping of Report values to text
    mappings = [
        ["NEGATIVE", "15 TRACE", "POSITIVE", "75 TRACE", "125 MODERATE", "500 LARGE"],      # Leukocytes
        ["NEGATIVE", "POSITIVE", "POSITIVE", "POSITIVE", "POSITIVE", "POSITIVE"],           # Nitrite
        ["3.2 NORMAL", "16 NORMAL", "32 +", "64 ++", "128 +++"],                            # Urobilinogen
        ["NEGATIVE", "TRACE", "0.3 +", "1.0 ++", "3.0 +++", ">= 20 ++++"],                  # Protein
        ["5.0","5.5","6.0","6.5","7.0","7.5","8.0","8.5"],                                  # pH
        ["NEGATIVE","NON-HEMOLYZED TRACE","10 TRACE","25 SMALL","80 MODERATE","200 LARGE"],  # Blood
        ["1.000","1.005","1.010","1.015","1.020","1.025","1.030"],                          # Sp. Gravity
        ["NEGATIVE","0.5 TRACE","1.5 SMALL","4.0 MODERATE","8.0 LARGE","16.0 LARGE"],       # Ketone
        ["NEGATIVE","NEGATIVE","NEGATIVE","17 SMALL","50 MODERATE","100 LARGE"],            # Bilirubin
        ["NEGATIVE","NEGATIVE","5 TRACE","15 +","30 ++","60 +++","110 ++++"],               # Glucose
    ]

    for i, entry in enumerate(test_entries):
        entry.delete(0, END)  # Clear previous value
        if i < len(Report):
            value = np.int32(Report[i])
            if value < len(mappings[i]):
                entry.insert(0, mappings[i][value])
            else:
                entry.insert(0, str(value))  # fallback

############################################### CUSTOM FUNCTIONS END HERE ###############################################

wb = xlwt.Workbook()
sheet1 = wb.add_sheet('Results')

Count = 0

# ------------------------------------------------------------------
# ROOT WINDOW
# ------------------------------------------------------------------
root = Tk()
root.title("CARE [Urinox Project]")
root.minsize(900, 600)

# Root grid configuration
root.columnconfigure(0, weight=1)
root.rowconfigure(2, weight=1)

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
header = ttk.Frame(root)
header.grid(row=0, column=0, sticky="ew", pady=10)

ttk.Label(
    header,
    text="In-House Testing Application",
    font=("Times New Roman", 24),
    anchor="center"
).pack(fill="x", padx=20)

# ------------------------------------------------------------------
# PATIENT INFO
# ------------------------------------------------------------------
import tkinter.font as tkfont; tkfont.nametofont("TkDefaultFont").configure(size=20)

patient = ttk.LabelFrame(root, text="Patient Information")
patient.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

for i in range(8):
    patient.columnconfigure(i, weight=1)

ttk.Label(patient, text="Name").grid(row=0, column=0, sticky="e", padx=5, pady=5)
name_entry = ttk.Entry(patient)
name_entry.grid(row=0, column=1, sticky="ew", padx=5)

ttk.Label(patient, text="Age").grid(row=0, column=2, sticky="e", padx=5)
age_entry = ttk.Entry(patient)
age_entry.grid(row=0, column=3, sticky="ew", padx=5)

ttk.Label(patient, text="Gender").grid(row=0, column=4, sticky="e", padx=5)
gender_entry = Combobox(patient, values=["", "Male", "Female", "Other"])
gender_entry.grid(
    row=0, column=5, sticky="ew", padx=5
)

ttk.Label(patient, text="Date & Time").grid(row=0, column=6, sticky="e", padx=5)
date_entry = ttk.Entry(patient)
date_entry.grid(row=0, column=7, sticky="ew", padx=5)

# ------------------------------------------------------------------
# IMAGE FILE INPUT
# ------------------------------------------------------------------
file_frame = ttk.Frame(root)
file_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
file_frame.columnconfigure(1, weight=1)

ttk.Label(file_frame, text="Image File").grid(row=0, column=0, sticky="e", padx=5)
image_path_entry = ttk.Entry(file_frame)
image_path_entry.grid(row=0, column=1, sticky="ew", padx=5)
ttk.Button(file_frame, text="Browse", command=BrowseButton).grid(
    row=0, column=2, padx=5
)

# ------------------------------------------------------------------
# MAIN CONTENT AREA
# ------------------------------------------------------------------
content = ttk.Frame(root)
content.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
content.columnconfigure(0, weight=3)
content.columnconfigure(1, weight=1)
content.rowconfigure(0, weight=1)

# ------------------------------------------------------------------
# IMAGE / PLOT AREA
# ------------------------------------------------------------------
image_frame = ttk.LabelFrame(content, text="Image Preview")
image_frame.grid(row=0, column=0, sticky="nsew", padx=5)
image_frame.rowconfigure(0, weight=1)
image_frame.columnconfigure(0, weight=1)

fig = Figure(figsize=(5, 4))
ax = fig.add_subplot(111)
ax.axis("off")
canvas = FigureCanvasTkAgg(fig, master=image_frame)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

# ------------------------------------------------------------------
# TEST RESULTS
# ------------------------------------------------------------------
results = ttk.LabelFrame(content, text="Test Results")
results.grid(row=0, column=1, sticky="nsew", padx=5)

results.columnconfigure(1, weight=1)

tests = [
    "Leukocytes", "Nitrite", "Urobilinogen", "Protein", "pH",
    "Blood", "Sp. Gravity", "Ketone", "Bilirubin", "Glucose"
]

test_entries = []  

for i, test in enumerate(tests):
    ttk.Label(results, text=test).grid(row=i, column=0, sticky="e", padx=5, pady=3)
    entry = ttk.Entry(results)
    entry.grid(row=i, column=1, sticky="ew", padx=5, pady=3)
    test_entries.append(entry)  # <-- save each Entry

# ------------------------------------------------------------------
# BUTTONS
# ------------------------------------------------------------------
buttons = ttk.Frame(root)
buttons.grid(row=4, column=0, pady=10)

for text, cmd in [
    ("Process", ProcessButton),
    ("Save", SaveButton),
    ("New", NewButton)
]:
    ttk.Button(buttons, text=text, width=12, command=cmd).pack(
        side="left", padx=10
    )

# ------------------------------------------------------------------
# PROGRESS BAR
# ------------------------------------------------------------------
progress = ttk.Progressbar(root, mode="determinate")
progress.grid(row=5, column=0, sticky="ew", padx=10, pady=5)

# ------------------------------------------------------------------
root.mainloop()
