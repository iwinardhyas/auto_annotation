import os
import shutil
import difflib

# folder_parent = "/home/dsc-01/dsc_erwin/google_image_downloader/google-images-download/mamam/"
# path = "/home/dsc-01/dsc_bariqi/yolov4/darknet-master/data/obj" 
# destination = "/home/dsc-01/dsc_erwin/google_image_downloader/google-images-download/images_check_MP4_JPG"

folder_parent = "/home/iwin/Documents/tools/Yolo_mark/x64/Release/data/img/"

jpg = []
txt = []

# for folder in os.listdir(folder_parent):
# 	for image in os.listdir(folder_parent+folder):
# 		#print(folder_parent+folder+image)
# 		if image.endswith('.JPG') or image.endswith('.mp4'):
# 			image_first = os.path.splitext(image)[0]
# 			shutil.move(folder_parent+folder+'/'+image, destination)
# 			#shutil.move(path+'/'+image_first+'.txt',destination)
# 			print(image,image_first+'.txt')
			

for image in os.listdir(folder_parent):
	# for image in os.listdir(folder_parent+folder):
	if image.endswith('.png') or image.endswith('.jpg'):
		image_first = os.path.splitext(image)[0]
		jpg.append(image_first)
	if image.endswith('.txt'):
		txt_first = os.path.splitext(image)[0]
		txt.append(txt_first)

data = list(set(jpg) - set(txt)) 

for img in data:
	image_moved = folder_parent+img+'.png'
	print(image_moved)