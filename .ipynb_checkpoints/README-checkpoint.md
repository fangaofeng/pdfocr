# pdfocr
this project can ocr scanning image and out excel,only support table or bill format.
you can open billocr.ipynb, then input your image or direct.

function removeStamp can remove red stamp.

debug is true ,show debug image or tips.
path = './'         path is image direct.
filename="test.jpeg"  filename is image name.

ocrimgtoexcel can deal only one image.

you can modiy baiduaip APP_ID='', API_KEY='', SECRET_KEY=''.
Or you can use you ocr api or tesserocr.function ocrclient.image_to_stringforcv2(roiimg) ,input image then return string.
