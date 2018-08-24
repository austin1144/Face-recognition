import cv2, os
import numpy as np
import matplotlib.pyplot as plt

def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
    # just making a copy of image passed, so that passed image is not changed
    img_copy = colored_img.copy()
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=15);
    # go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = gray[y:y + h, x:x + w]
        # crop_img = img_copy[y:y + h, x:x + w]  # for colorful image
    return crop_img

def detect_and_save_faces(name, roi_size):
    #detect face and save the interesting region,    # define where to look for images and where to save the detected faces
    # dir_images = "C:/Users/Austin24/Downloads/ex3/data/{}".format(name)
    # dir_faces = "C:/Users/Austin24/Downloads/ex3/data/{}/faces".format(name)
    dir_images = "C:/Users/Austin24/Google 雲端硬碟/MAI/2nd semester/Computer_Vision/ex3/data/{}".format(name)
    dir_faces = "C:/Users/Austin24/Google 雲端硬碟/MAI/2nd semester/Computer_Vision/ex3/data/{}/faces".format(name)
    if not os.path.isdir(dir_faces): os.makedirs(dir_faces)  
    # put all images in a list
    names_images = [name for name in os.listdir(dir_images) if not name.startswith(".") and name.endswith(".jpg")] # can vary a little bit depending on your operating system

    # detect for each image the face and store this in the face directory with the same file name as the original image
    # haar_face_cascade = cv2.CascadeClassifier('C:/Users/Austin24/Downloads/ex3/data/haarcascade_frontalface_alt.xml')
    haar_face_cascade = cv2.CascadeClassifier('C:/Users/Austin24/Google 雲端硬碟/MAI/2nd semester/Computer_Vision/ex3/data/haarcascade_frontalface_alt.xml')
    print names_images
    for i in names_images:
        colored_img = cv2.imread(dir_images+"/"+i)
        faces_detected_img = detect_faces(haar_face_cascade, colored_img)
        faces_detected_img = cv2.resize(faces_detected_img,roi_size,1,1)
        # imshow("test",faces_detected_img)
        cv2.imwrite(dir_faces+"/0_"+i ,faces_detected_img,roi_size)
        cv2.destroyAllWindows()


def do_pca_and_build_model(name, roi_size, numbers):

    # define where to look for the detected faces
    dir_faces = "C:/Users/Austin24/Google 雲端硬碟/MAI/2nd semester/Computer_Vision/ex3/data/{}/faces".format(name)

    # put all faces in a list
    names_faces = ["0_{}.jpg".format(n) for n in numbers]
    N = len(names_faces)

    # put all faces as data vectors in a N x P data matrix X with N the number of faces and P=roi_size[0]*roi_size[1] the number of pixels
    P = (roi_size[0]*roi_size[1])
    X = np.zeros((N,P))

    # create zero array to store image for 3 the thing is I need to store 6 images.
    i = 0
    for k in names_faces:
        img = cv2.imread(dir_faces+"/"+k)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arr = np.array(gray)
        # print arr
        X[i] = arr.ravel() # or X[i] = arr.flat()
        X[i] = np.matrix(X[i]) # create vector
        i += 1

    mean, eigenvalues, eigenvectors = pca(X, number_of_components=N) # calculate the eigenvectors of X

    return [mean, eigenvalues, eigenvectors]
    
def test_images(name, roi_size, numbers, models):

    # define where to look for the detected faces
    # dir_faces = "C:/Users/Austin24/Downloads/ex3/data/{}/faces".format(name)
    dir_faces = "C:/Users/Austin24/Google 雲端硬碟/MAI/2nd semester/Computer_Vision/ex3/data/{}/faces".format(name)
    
    # put all faces in a list
    names_faces = ["0_{}.jpg".format(n) for n in numbers]
    N = len(names_faces)
    # print N
    # put all faces as data vectors in a N x P data matrix X with N the number of faces and P=roi_size[0]*roi_size[1] the number of pixels
    P = (roi_size[0] * roi_size[1])
    X = np.zeros((N, P))
    i = 0

    for k in names_faces:
        img = cv2.imread(dir_faces + "/" + k)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arr = np.array(gray)
        X[i] = arr.ravel()
        # create vector
        X[i] = np.matrix(X[i])
        i += 1

    # store the results as [[results_model_arnold_reconstructed_X, results_model_arnold_MSE], [results_model_barack_reconstructed_X, results_model_barack_MSE]]
    results = []
    for model in models:
        projections, reconstructions = project_and_reconstruct(X, model)
        mse = np.mean((X - reconstructions) ** 2, axis=1)
        results.append([reconstructions, mse])

    return results

def pca(X, number_of_components):
    # consider why do I need reshape
    mean = np.mean(X,0) #0 is avg(row), 1 is avg(col)
    new_x= np.subtract(X, mean)
    R = np.cov(new_x, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    key = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[key], eigenvectors[:, key]
    # print eigenvalue
    # plt.hist(eigenvalues.ravel(), bins=256, range=(0.0, 2500), fc='k', ec='k')
    # plt.show()
    # plt.plot(eigenvalues, 'r--')
    # plt.show()
    eigenvectors = eigenvectors[:,:number_of_components]
    # eigenvalues = eigenvalues
    eigenvalues = eigenvalues[:number_of_components]

    return [mean, eigenvalues, eigenvectors]


def project_and_reconstruct(X, model):
    mean = model[0]
    value = model[1]
    vector = model[2]
    X_shift = X- mean
    #do projection
    projections = np.dot(X_shift, vector)
    reconstructions = np.dot(projections, vector.T) + mean  # 2*2500
    # for i in range(1):
    #     plt.imshow(reconstructions[i].reshape((50, 50)))
    #     plt.show()

    return [projections, reconstructions]
def classification(results_arnold,results_barack):
    for i in range(len(results_arnold[0][1])):
        if (results_arnold[0][1][i] < results_arnold[1][1][i]):
            print "Figure is Arnold"
        else:
            print "Figure is Barack"
    for i in range(len(results_barack[0][1])):
        if (results_barack[0][1][i] < results_barack[1][1][i]):
            print "Figure is Arnold"
        else:
            print "Figure is Bara"
def visual_face(model_arnold, model_barack,roi_size):
    print "start"
    x=[model_arnold,model_barack]
    name=["arnold_eigen","bara_eigen"]
    m=0
    for j in x:
        eigenvector=j[2]
        eigenvalue=j[1]
        mean = j[0]
        print name[m]
        # dir_faces = "C:/Users/Austin24/Downloads/ex3/data/{}".format(name[m])
        dir_faces = "C:/Users/Austin24/Google 雲端硬碟/MAI/2nd semester/Computer_Vision/ex3/data/{}".format(name[m])
        if not os.path.isdir(dir_faces): os.makedirs(dir_faces)
        projections = np.dot(eigenvalue, eigenvector.T)
        eigenface = projections+ mean
        # eigenface = np.dot(projections, eigenvector.T) + mean
        # eigenface = eigenvector.T + mean
        faces_detected_img = eigenface.reshape(roi_size)
        faces_detected_img = cv2.resize(faces_detected_img, roi_size, 1, 1)
        cv2.imwrite(dir_faces + "/2_" + name[m]+".jpg", faces_detected_img, roi_size)
        cv2.destroyAllWindows()
        m += 1
        # for i in range(6):
        #     plt.imshow(eigenvector[:, i].reshape(roi_size))
        #     plt.show()
def visual_recon(name,model,roi_size):
    m=0
    dir_faces = "C:/Users/Austin24/Google 雲端硬碟/MAI/2nd semester/Computer_Vision/ex3/data/recon/"
    # dir_faces = "C:/Users/Austin24/Downloads/ex3/data/recon/"

    if not os.path.isdir(dir_faces): os.makedirs(dir_faces)
    print "model name: ",name
    num_model =len(model)
    for i in range(num_model):
        r_model = model[i]
        data = len(r_model[0])
        # print data
        for j in range(data):
            eigenface = r_model[0][j]
            # plt.imshow(eigenface.reshape(roi_size))
            # plt.show()
            faces_detected_img = eigenface.reshape(roi_size)
            faces_detected_img = cv2.resize(faces_detected_img, roi_size, 1, 1)
            print m
            cv2.imwrite(dir_faces + name + str(m)+".jpg", faces_detected_img, roi_size)
            cv2.destroyAllWindows()
            m += 1


if __name__ == '__main__':

    roi_size = (50, 50) # reasonably quick computation time

    # Detect all faces in all the images in the folder of a person (in this case "arnold" and "barack") and save them in a subfolder "faces" accordingly
    # detect_and_save_faces("arnold", roi_size=roi_size)
    # detect_and_save_faces("barack", roi_size=roi_size)

    # visualize detected ROIs overlayed on the original images and copy paste these figures in a document
    ## TODO ## # please comment this line when submitting

    # Perform PCA on the previously saved ROIs and build a model=[mean, eigenvalues, eigenvectors] for the corresponding person's face making use of a training set
    model_arnold = do_pca_and_build_model("arnold", roi_size=roi_size, numbers=[1, 2, 3, 4, 5, 6])
    model_barack = do_pca_and_build_model("barack", roi_size=roi_size, numbers=[1, 2, 3, 4, 5, 6])

    # visualize these "models" in some way (of your choice) and copy paste these figures in a document
    ## TODO ## # please comment this line when submitting
    visual_face(model_arnold,model_barack,roi_size)

    # Test and reconstruct "unseen" images and check which model best describes it (wrt MSE)
    # results=[[results_model_arnold_reconstructed_X, results_model_arnold_MSE], [results_model_barack_reconstructed_X, results_model_barack_MSE]]
    # The correct model-person combination should give best reconstructed images and therefor the lowest MSEs
    results_arnold = test_images("arnold", roi_size=roi_size, numbers=[7, 8], models=[model_arnold, model_barack])
    results_barack = test_images("barack", roi_size=roi_size, numbers=[7, 8, 9, 10], models=[model_arnold, model_barack])

    # visualize the reconstructed images and copy paste these figures in a document
    ## TODO ## # please comment this line when submitting
    # visual_recon("arnold", results_arnold, roi_size)
    # visual_recon("barack", results_barack, roi_size)
    # print classification(results_arnold,results_barack)
