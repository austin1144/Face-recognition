import cv2, os
import numpy as np
import matplotlib as plt


def imshow(name, images):
    ''' Display images (a list with images all equal number of channels) all together '''
    image = np.concatenate(images, axis=1)
    image = cv2.resize(image, dsize=tuple([s // 2 for s in image.shape if s > 3])[::-1])
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
X= np.array([[ 0.,  1.,  2.], [ 3.,  4.,  5.],[ 6.,  7.,  8.]])
X2 = [1,2,3]
print len(X2)
for i in range(len(X2)):
    X[i] += X2[i]
# mean = np.mean(X,1) #0 is avg(row), 1 is avg(col)
# print mean
# for i in range(3):
#     X[i] -= mean[i]
#     # std = np.std(mean,0)
# print X
print X

#print 6 eigenface
# print "start"
# for i in range(6):
#     plt.imshow(eigenvectors[:,i].reshape(50,50))
#     plt.show()
#     print i
# print "stop"
