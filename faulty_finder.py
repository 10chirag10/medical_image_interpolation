import os, pydicom, tqdm
root = './dataset_folder/train'
for folders in tqdm.tqdm(os.listdir(root)):
    images = os.path.join(root,folders)
    for image in os.listdir(images):
        try:
                d = pydicom.dcmread(os.path.join(images,image)).pixel_array
        except:
                print(folders , " --------> " , image)
