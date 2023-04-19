#!/usr/bin/env python3
import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import model
import dataloader
import platform
from tqdm import tqdm
import torch.nn as nn
import numpy as np
#import SimpleITK as sitk
import pydicom
from pydicom import Dataset,DataElement

# For parsing commandline arguments
parser = argparse.ArgumentParser()
#parser.add_argument("--ffmpeg_dir", type=str, default="", help='path to ffmpeg.exe')
parser.add_argument("--video", type=str, required=True, help='path of video to be converted')
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model')
parser.add_argument("--fps", type=float, default=30, help='specify fps of output video. Default: 30.')
parser.add_argument("--sf", type=int, required=True, help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--output", type=str, default="output.mp4", help='Specify output file name. Default: output.mp4')
args = parser.parse_args()

def check():
    """
    Checks the validity of commandline arguments.

    Parameters
    ----------
        None

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """


    error = ""
    if (args.sf < 2):
        error = "Error: --sf/slomo factor has to be atleast 2"
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    if (args.fps < 1):
        error = "Error: --fps has to be atleast 1"
    return error


def main():
    # Check if arguments are okay
#    error = check()
#    if error:
#        print(error)
#        exit(1)
    '''
    # Create extraction folder and extract frames
    IS_WINDOWS = 'Windows' == platform.system()
    extractionDir = "tmpSuperSloMo"
    if not IS_WINDOWS:
        # Assuming UNIX-like system where "." indicates hidden directories
        extractionDir = "." + extractionDir
    if os.path.isdir(extractionDir):
        rmtree(extractionDir)
    os.mkdir(extractionDir)
    if IS_WINDOWS:
        FILE_ATTRIBUTE_HIDDEN = 0x02
        # ctypes.windll only exists on Windows
        ctypes.windll.kernel32.SetFileAttributesW(extractionDir, FILE_ATTRIBUTE_HIDDEN)

    extractionPath = os.path.join(extractionDir, "input")
    outputPath     = os.path.join(extractionDir, "output")
    os.mkdir(extractionPath)
    os.mkdir(outputPath)
    error = extract_frames(args.video, extractionPath)
    if error:
        print(error)
        exit(1)
    '''
    extractionPath = args.video
    outputPath = os.path.join(args.video, "output")
    if os.path.isdir(outputPath):
        rmtree(outputPath)
    os.mkdir(outputPath)
    # Initialize transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.150, 0.150, 0.150]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    
    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])
#
    # Load data
    videoFrames = dataloader.Video(root=extractionPath, transform=transform)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)
#    print(type(videoFramesloader))

    # Initialize model
    flowComp = nn.DataParallel(model.UNet(6, 4))
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = nn.DataParallel(model.UNet(20, 5))
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False
    
    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)

    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    # Interpolate frames
    frameCounter = 1

    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):

            I0 = frame0.to(device)
            I1 = frame1.to(device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            # Save reference frames in output folder
            
            for batchIndex in range(args.batch_size):
                tmp=frame0[batchIndex].detach()*6000 - 2000
                #print(tmp.shape)
                tmp=np.swapaxes(tmp,1,2)
                tmp=np.swapaxes(tmp,0,2)
                
                tmp = tmp[:,:,1]
                
                #print(tmp.shape)

                tmp = tmp.numpy()
                tmp=tmp.astype('int16')
#                print(tmp)
#                print("tmp type: ", type(tmp))
                dicom_temp_image = pydicom.dcmread("temp.dcm")
#                print("SHAPES:- ",dicom_temp_image.pixel_array.shape)
#                print(tmp.shape)
#                print("BEFORE",dicom_temp_image.pixel_array[100,100])
                dicom_temp_image.PixelData = tmp.tobytes()
                
#                dicom_temp_image.BitsAllocated = 32
#                print("AFTER",dicom_temp_image.pixel_array[100,100])

#                dicom_temp_image.Length =
                stringName = str(frameCounter + args.sf * batchIndex)
                stringNameLength = len(stringName)
                if stringNameLength == 1 :
                    dicom_temp_image.SOPInstanceUID = "00000" + str(frameCounter + args.sf * batchIndex)
                    dicom_temp_image.save_as(os.path.join(outputPath,"00000" + str(frameCounter + args.sf * batchIndex) + ".dcm"))
                elif stringNameLength == 2 :
                    dicom_temp_image.SOPInstanceUID = "0000" + str(frameCounter + args.sf * batchIndex)
                    dicom_temp_image.save_as(os.path.join(outputPath,"0000" + str(frameCounter + args.sf * batchIndex) + ".dcm"))
                elif stringNameLength == 3 :
                    dicom_temp_image.SOPInstanceUID = "000" + str(frameCounter + args.sf * batchIndex)
                    dicom_temp_image.save_as(os.path.join(outputPath,"000" + str(frameCounter + args.sf * batchIndex) + ".dcm"))
            
#                writer = sitk.ImageFileWriter()
#                writer.SetImageIO("GDCMImageIO")
#                writer.SetFileName(str(frameCounter + args.sf * batchIndex))
#                writer.Execute(tmp)
#                ds = Dataset()
#                ds.add(DataElement(0x00100020, 'LO', tmp))
#                print(type(ds))
#                print(ds.pixel_array.shape)
#                dcm_image = sitk.GetImageFromArray(tmp)
#                print("dcm_image type:  ----- ---- ",type(dcm_image))
#                dcm_image.save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex)))

#                dcm_image.save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".jpg"))
#
            frameCounter += 1


            # Generate intermediate frames
            for intermediateIndex in range(1, args.sf):
                t = intermediateIndex / args.sf
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)
                
                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                    
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0
                    
                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)
                
                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save intermediate frame
                
                for batchIndex in range(args.batch_size):
                     tmp=frame0[batchIndex].detach()*6000 - 2000
                     #print(tmp.shape)
                     tmp = tmp[1,:,:]
#                     tmp=np.swapaxes(tmp,1,2)
#                     tmp=np.swapaxes(tmp,0,2)
#                     print(tmp.shape)
                     tmp = tmp.numpy()
#                     print("tmp type: ", type(tmp))
                     tmp=tmp.astype('int16')
                    #                print(tmp)
#                     print("tmp type: ", type(tmp))
                     dicom_temp_image = pydicom.dcmread("temp.dcm")
                    #                print("SHAPES:- ",dicom_temp_image.pixel_array.shape)
                    #                print(tmp.shape)
#                     print("BEFORE",dicom_temp_image.pixel_array[100,100])
                     dicom_temp_image.PixelData = tmp.tobytes()
                    #                dicom_temp_image.BitsAllocated = 32
#                     print("AFTER",dicom_temp_image.pixel_array[100,100])

                    #                dicom_temp_image.Length =
                     stringName = str(frameCounter + args.sf * batchIndex)
                     stringNameLength = len(stringName)
                     if stringNameLength == 1 :
                         dicom_temp_image.SOPInstanceUID = "00000" + str(frameCounter + args.sf * batchIndex)
                         dicom_temp_image.save_as(os.path.join(outputPath,"00000" + str(frameCounter + args.sf * batchIndex) + ".dcm"))
                     elif stringNameLength == 2 :
                        dicom_temp_image.SOPInstanceUID = "0000" + str(frameCounter + args.sf * batchIndex)
                        dicom_temp_image.save_as(os.path.join(outputPath,"0000" + str(frameCounter + args.sf * batchIndex) + ".dcm"))
                     elif stringNameLength == 3 :
                         dicom_temp_image.SOPInstanceUID = "000" + str(frameCounter + args.sf * batchIndex)
                         dicom_temp_image.save_as(os.path.join(outputPath,"000" + str(frameCounter + args.sf * batchIndex) + ".dcm"))
#                     dcm_image.save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex)))
#                     tmp.save_as(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".jpg"))

                frameCounter += 1
                
            
            # Set counter accounting for batching of frames
            frameCounter += args.sf * (args.batch_size - 1)

    # Generate video from interpolated frames
    '''
    create_video(outputPath)
    '''
    # Remove temporary files
    '''
    rmtree(extractionDir)
    '''
    exit(0)

main()
