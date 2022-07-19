from PIL import Image
import os

dl = os.listdir('./data/data2/')
ml = os.listdir('./data/mask/')
for data in dl:
    i = Image.open('./data/data2/' + data)
    i.save('./data/data2/' + data.split('.')[0] + '.jpg')
for data in ml:
    i = Image.open('./data/mask/' + data)
    i.save('./data/mask/' + data.split('.')[0] + '.jpg')