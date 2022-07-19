import numpy as np
from PIL import Image
import os

ml = os.listdir('./data/test_mask2/')
for data in ml:
    im = Image.open('./data/test_mask2/'+data)
    im_np = np.array(im)
    for x in range(im_np.shape[0]):
        for y in range(im_np.shape[1]):
            if im_np[x][y] == 255:
                im_np[x][y] = 1
            else:
                im_np[x][y] = 0
    img = Image.fromarray(im_np, 'L')
    img.save('./data/test_mask2/'+data)
    print(data)

# import numpy as np
# from PIL import Image
# import os

# ml = os.listdir('./data/mask3/')
# for data in ml:
#     im = Image.open('./data/mask3/'+data)
#     im_np = np.array(im)
#     print(im_np.shape)
#     im_np = np.expand_dims(im_np, axis=0)
#     im_np = np.expand_dims(im_np, axis=0)
#     print(im_np.shape)
#     im_np = im_np.reshape(1, 500, 60)
#     print(im_np.shape)
#     break