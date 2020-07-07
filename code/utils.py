# -*- coding: utf-8 -*-
"""
package of useful methods for training models
"""

# import libraries
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from torch import nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import random
from torchvision import transforms, models
import torch.nn.functional as F
from facenet_pytorch import MTCNN
import pandas as pd
from joblib import load

#check gpu is enabled
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set random seeds
ia.seed(2)
torch.manual_seed(0)

# image augmentation
# define image augmentation sequence for training data
class AugSeq:
    def __init__(self):
        self.aug = iaa.Sequential([
                                   iaa.Fliplr(0.5), # horizontal flips
                                   # gaussian blur 
                                   iaa.GaussianBlur(sigma=(0, 3)),
                                   # rotate images between -10 to +10 degrees
                                   iaa.Affine(rotate=(-15, 15)),
                                   iaa.GammaContrast((0.5,1))
                                   ], random_order=True) # apply augmenters in random order
  
    def __call__(self, img):
        img = np.array(img)
        return Image.fromarray(self.aug.augment_image(img))

# neural net early stopping
# taken from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
        
        
# pca feature extractor
class pca_extractor():
    def __init__(self, pca=None, pca_train_data=None, pca_n_comps=128):
        self.pca = pca
        self.pca_n_comps = pca_n_comps

        # train pca
        if self.pca is None:
            self.pca_fit(pca_train_data)

    def flat_array(self, images):
        if type(images) is list:
            images = [np.ravel(np.array(image)) for image in images]
        else:
            images = np.ravel(np.array(images)).reshape(1,-1)
        return images
  
    def pca_fit(self, images, pca_n_comps=128):
        images = self.flat_array(images)
        pca = PCA(n_components=self.pca_n_comps, svd_solver='arpack')
        self.pca = pca.fit(images)

    def __call__(self, images):
        images = self.flat_array(images)
        return self.pca.transform(images).astype('float32')
    
# surf 'bag of visual words' feature extractor
class surf_extractor():
    def __init__(self, hessian_threshold=70, set_extended=False, kmeans=None, kmeans_train_data=None, n_clusters=85):

        self.hessian_threshold = hessian_threshold
        self.set_extended = set_extended
        self.kmeans = kmeans
        self.n_clusters = n_clusters
        self.kp = None
        self.des = None
    
        # fit kmeans
        if self.kmeans is None:
            self.kmeans_fit(kmeans_train_data)
    def surf_kp_extract(self, images):
        # extract surf key points and descriptors
        surf = cv2.xfeatures2d.SURF_create(self.hessian_threshold)
    
        if self.set_extended:
            surf.setExtended(True)
    
        if type(images) is list:
            kp_des = [surf.detectAndCompute(np.array(image),None) for image in images]
            kp = [item[0] for item in kp_des]
            des = [item[1] for item in kp_des]
        else:
            kp, des = surf.detectAndCompute(np.array(images),None)
    
        self.kp = kp
        self.des = des

    def kmeans_fit(self, images):
        # extract surf key points and descriptors
        self.surf_kp_extract(images)
        des = np.concatenate(self.des, axis=0)
    
        # fit kmeans clustering algorithm
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0, init_size=3*self.n_clusters)
        #kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        kmeans.fit(des)
        
        self.kmeans = kmeans

    def __call__(self, images, surf_extraction=True):
        # extract surf key points and descriptors
        if surf_extraction:
            self.surf_kp_extract(images)
    
        if type(images) is list:
            bovw = [np.zeros(self.n_clusters, dtype=np.int) for item in self.des]
            # get visual words associated with keypoints for each image
            vw = [self.kmeans.predict(item) for item in self.des]
            # get bag of visual words for each image
            vw_counts = [np.unique(item, return_counts=True) for item in vw]
            for i, item in enumerate(vw_counts):
                vw = item[0]
                counts = item[1]
                bovw[i][vw] = counts
            bovw = np.stack(bovw)
        else:
            bovw = np.zeros(self.n_clusters, dtype=np.int)
            vw = self.kmeans.predict(self.des)
            vw, counts = np.unique(vw, return_counts=True)
            bovw[vw] = counts
            bovw = np.expand_dims(bovw, axis=0)
    
        self.bovw = bovw.astype('float32')
    
          #bovw_bin_edges = [np.histogram(item, bins=len(self.kmeans.cluster_centers_)) for item in vw]
          #self.data = [item[0] for item in bovw_bin_edges]

        return self.bovw
    
# show input images
def imshow(images, image_titles=None, labels=None, specific_label=None, dims=(4,8), figsize=(12,6)):
        #shuffle images and labels and get as many as specified
        images_s = []
        image_titles_s = []
        labels_s = []
        index_s = list(range(len(images)))
        random.shuffle(index_s)
        for i in index_s:
            images_s.append(images[i])
            if image_titles is not None:
                image_titles_s.append(image_titles[i])
            if labels is not None:
                labels_s.append(labels[i])
        images = images_s
        if image_titles is not None:
            image_titles = image_titles_s
        if labels is not None:
            labels = labels_s

        class_names = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
                       '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32',
                       '33','34','36','38','40','42','44','46','48','50','52','54','56','58','60','78']

        if specific_label is not None:
            label_idx = class_names.index(specific_label)
            label_filter = (np.array(labels) == label_idx)
            images = [image for image, idx in zip(images,list(label_filter)) if idx]
            if image_titles is not None:
                image_titles = [image_title for image_title, idx in zip(image_titles,list(label_filter)) if idx]

        n_rows, n_cols = dims
        num_images = n_rows*n_cols
    
        images = images[:num_images]
    
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        ax = ax.reshape(-1)
        plt.gray()
        for i in range(len(images)):
            ax[i].imshow(images[i])
            ax[i].axis('off')
            if image_titles is not None:
                ax[i].set_title(f"{image_titles[i]}")
        plt.tight_layout()

# visualise outputs of cnn
def visualise_cnn(model, images, labels, class_names, show_errors=False, specific_label=None,
                    dims=(3,6), figsize=(12,6)):

    #shuffle images and labels and get as many as specified
    images_s = []
    labels_s = []
    index_s = list(range(len(images)))
    random.shuffle(index_s)
    for i in index_s:
        images_s.append(images[i])
        labels_s.append(labels[i])
    images = images_s
    labels = labels_s

    #feature extraction
    transform = transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    feature_vectors = torch.stack([transform(image) for image in images])

    #predictions
    model.eval()
    outputs = model(feature_vectors.to(device).float())
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    labels = np.array(labels)

    if specific_label is not None:
        label_idx = class_names.index(specific_label)
        label_filter = (labels == label_idx)
        preds = preds[label_filter]
        labels = labels[label_filter]
        images = [image for image, idx in zip(images,list(label_filter)) if idx]

    if show_errors:
        error_filter = (preds != labels)
        preds = preds[error_filter]
        labels = labels[error_filter]
        images = [image for image, idx in zip(images,list(error_filter)) if idx]

    n_rows, n_cols = dims
    num_images = n_rows*n_cols

    preds = preds[:num_images]
    labels = labels[:num_images]
    images = images[:num_images]

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax = ax.reshape(-1)
    plt.gray()
    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].axis('off')
        ax[i].set_title(f'{class_names[preds[i]]} ({class_names[labels[i]]})',
                        color=("green" if preds[i]==labels[i] else "red"))
    plt.tight_layout()
    
    
# visualise svm outputs
def visualise_svm(model, feature_extractor, images, labels, class_names, show_errors=False, specific_label=None,
                    dims=(3,6), figsize=(12,6)):

    #shuffle images and labels and get as many as specified
    images_s = []
    labels_s = []
    index_s = list(range(len(images)))
    random.shuffle(index_s)
    for i in index_s:
        images_s.append(images[i])
        labels_s.append(labels[i])
    images = images_s
    labels = labels_s

    #feature extraction
    feature_vectors = feature_extractor(images)

    #predictions
    preds = model.predict(feature_vectors)
    labels = np.array(labels)

    if specific_label is not None:
        label_idx = class_names.index(specific_label)
        label_filter = (labels == label_idx)
        preds = preds[label_filter]
        labels = labels[label_filter]
        images = [image for image, idx in zip(images,list(label_filter)) if idx]

    if show_errors:
        error_filter = (preds != labels)
        preds = preds[error_filter]
        labels = labels[error_filter]
        images = [image for image, idx in zip(images,list(error_filter)) if idx]

    n_rows, n_cols = dims
    num_images = n_rows*n_cols
  
    preds = preds[:num_images]
    labels = labels[:num_images]
    images = images[:num_images]

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax = ax.reshape(-1)
    plt.gray()
    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].axis('off')
        ax[i].set_title(f'{class_names[preds[i]]} ({class_names[labels[i]]})', 
                        color=("green" if preds[i]==labels[i] else "red"))
    plt.tight_layout()
    
    
# visualise mlp model outputs
def visualise_mlp(model, feature_extractor, images, labels, class_names, show_errors=False, 
                  specific_label=None, dims=(3,6), figsize=(12,6)):

    #shuffle images and labels and get as many as specified
    images_s = []
    labels_s = []
    index_s = list(range(len(images)))
    random.shuffle(index_s)
    for i in index_s:
        images_s.append(images[i])
        labels_s.append(labels[i])
    images = images_s
    labels = labels_s

    #feature extraction
    feature_vectors = feature_extractor(images)
    feature_vectors = torch.from_numpy(feature_vectors).squeeze().float()

    #predictions
    outputs = model(feature_vectors.to(device))
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    labels = np.array(labels)

    if specific_label is not None:
        label_idx = class_names.index(specific_label)
        label_filter = (labels == label_idx)
        preds = preds[label_filter]
        labels = labels[label_filter]
        images = [image for image, idx in zip(images,list(label_filter)) if idx]

    if show_errors:
        error_filter = (preds != labels)
        preds = preds[error_filter]
        labels = labels[error_filter]
        images = [image for image, idx in zip(images,list(error_filter)) if idx]

    n_rows, n_cols = dims
    num_images = n_rows*n_cols

    preds = preds[:num_images]
    labels = labels[:num_images]
    images = images[:num_images]

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax = ax.reshape(-1)
    plt.gray()
    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].axis('off')
        ax[i].set_title(f'{class_names[preds[i]]} ({class_names[labels[i]]})', 
                        color=("green" if preds[i]==labels[i] else "red"))
    plt.tight_layout()
    
# create MLP architecture
class Net(nn.Module):
    def __init__(self, input_size, n_classes, dropout_rate=0.5):
        super(Net, self).__init__()
        #input_size = torch.FloatTensor([input_size])
        #n_classes = torch.FloatTensor([n_classes])
        hl_size = (input_size+n_classes)//2
        self.fc1 = nn.Linear(input_size, hl_size)
        self.fc2 = nn.Linear(hl_size, n_classes)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        # flatten image input
        # x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # add output layer
        x = self.fc2(x)
        return x
    
# recognise face function
class RecogniseFace():
    def __init__(self, I, featureType='pca', classifierType='cnn', creativeMode=0, figsize=(18, 10)):
        #I = Image.open(I)
        I = cv2.imread(I, cv2.IMREAD_COLOR)
        I = Image.fromarray(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
        I_copy = I.copy()
        self.I = I_copy
        self.featureType = featureType
        self.classifierType = classifierType
        self.creativeMode = creativeMode
        self.figsize = figsize
        self.class_names = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
                            '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32',
                            '33','34','36','38','40','42','44','46','48','50','52','54','56','58','60','78','n/a']

        # extract faces and bounding boxes
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(select_largest=False, post_process=False, device=device, keep_all=True)

        self.faces = [face.permute(1,2,0).int().numpy().astype('uint8') for face in mtcnn(self.I)]

        boxes, _ = mtcnn.detect(self.I)
        self.boxes = [box.astype('int').tolist() for box in boxes]

        # preprocess and classify faces and collect predicted labels
        self.classify(self.faces)
 
        # create P matrix (predicted label, x, y)
        ID = [self.class_names[pred] for pred in self.preds]
        x = [int(np.mean(box[::2])) for box in self.boxes]
        y = [int(np.mean(box[1::2])) for box in self.boxes]
        #self.P = np.column_stack([ID, x, y])
        self.P = pd.DataFrame({'ID':ID,'x':x,'y':y})

        # apply creative mode (if required), create image with bounding boxes and predicted labels
        if creativeMode == 1:
            aug = iaa.Cartoon()
            faces_cartoon = [aug.augment_image(face) for face in self.faces]
            self.cartoonify(self.I, faces_cartoon, self.boxes)

        self.draw_boxes(self.I, self.boxes, self.preds)

        # print P matrix, and plot image
        print(self.P, '\n')

        plt.figure(figsize=self.figsize)
        plt.imshow(self.I)
        #plt.axis('off')
        plt.show()

    def classify(self, faces):
        models_path = '../models'
        if self.classifierType == 'cnn':
            # set up model
            alexnet_ft = models.alexnet()
            num_ftrs = alexnet_ft.classifier[6].in_features
            alexnet_ft.classifier[6] = nn.Linear(num_ftrs, 48)
            alexnet_ft.load_state_dict(torch.load(models_path + '/alexnet_ft.pt',map_location=lambda storage, loc: storage))
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            alexnet_ft = alexnet_ft.to(device)
            alexnet_ft.eval()
            # preprocess images
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            inputs = torch.stack([transform(face) for face in faces]).to(device)
            # perform classification
            outputs = alexnet_ft(inputs)
            scores, preds = torch.max(outputs,1)
            preds = preds.tolist()
            scores = scores.tolist()
            # set preds with low confidence to n/a
            preds_final = []
            for pred, score in zip(preds,scores):
                if score < -1000:
                    preds_final.append(48)
                else:
                    preds_final.append(pred)
            self.preds = preds_final
            
        elif self.featureType == 'pca' and self.classifierType == 'svm':
            # instantiate model
            svc = load(models_path + '/pca_svm.joblib') 
            # preprocess images
            pca = load(models_path + '/pca.joblib') 
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),pca_extractor(pca=pca, pca_n_comps=pca.components_.shape[0])])
            inputs = np.stack([transform(Image.fromarray(face)) for face in faces]).squeeze()
            # check if single dimensional array (as needs reshaping)
            if inputs.ndim == 1:
                inputs = inputs.reshape(1, -1)
            # perform classification
            preds = svc.predict(inputs).tolist()
            scores = [svc.predict_proba(input_.reshape(1, -1)).max() for input_ in inputs]
            # set preds with low confidence to n/a
            preds_final = []
            for pred, score in zip(preds,scores):
                if score < 0:
                    preds_final.append(48)
                else:
                    preds_final.append(pred)
            self.preds = preds_final
            
        elif self.featureType == 'pca' and self.classifierType == 'mlp':
            # set up model
            pca = load(models_path + '/pca.joblib') 
            mlp = Net(input_size=pca.components_.shape[0], n_classes=48)
            mlp.load_state_dict(torch.load(models_path + '/pca_mlp.pt',map_location=lambda storage, loc: storage))
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            mlp = mlp.to(device)
            mlp.eval()
            # preprocess images
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            pca_extractor(pca=pca, pca_n_comps=pca.components_.shape[0]),
                                            transforms.ToTensor()])
            inputs = torch.stack([transform(Image.fromarray(face)) for face in faces]).squeeze().to(device)
            # check if single dimensional tensor (as needs reshaping)
            if len(list(inputs.size())) == 1:
                inputs = inputs.unsqueeze(0)
            # perform classification
            outputs = mlp(inputs)
            scores, preds = torch.max(outputs,1)
            preds = preds.tolist()
            scores = scores.tolist()
            # set preds with low confidence to n/a
            preds_final = []
            for pred, score in zip(preds,scores):
                if score < -1000:
                    preds_final.append(48)
                else:
                    preds_final.append(pred)
            self.preds = preds_final
  
        elif self.featureType == 'surf' and self.classifierType == 'svm':
            # instantiate model
            svc = load(models_path + '/surf_svm.joblib') 
            # preprocess images
            kmeans = load(models_path + '/kmeans.joblib') 
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            surf_extractor(kmeans=kmeans, hessian_threshold=31, n_clusters=kmeans.cluster_centers_.shape[0])])
            inputs = np.stack([transform(Image.fromarray(face)) for face in faces]).squeeze()
            # check if single dimensional array (as needs reshaping)
            if inputs.ndim == 1:
                inputs = inputs.reshape(1, -1)
            # perform classification
            preds = svc.predict(inputs).tolist()
            scores = [svc.predict_proba(input_.reshape(1, -1)).max() for input_ in inputs]
            # set preds with low confidence to n/a
            preds_final = []
            for pred, score in zip(preds,scores):
                if score < 0:
                    preds_final.append(48)
                else:
                    preds_final.append(pred)
            self.preds = preds_final
            
        elif self.featureType == 'surf' and self.classifierType == 'mlp':
            # set up model
            kmeans = load(models_path + '/kmeans.joblib') 
            mlp = Net(input_size=kmeans.cluster_centers_.shape[0], n_classes=48)
            mlp.load_state_dict(torch.load(models_path + '/surf_mlp.pt',map_location=lambda storage, loc: storage))
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            mlp = mlp.to(device)
            mlp.eval()
            # preprocess images
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            surf_extractor(kmeans=kmeans, hessian_threshold=31, n_clusters=kmeans.cluster_centers_.shape[0]),
                                            transforms.ToTensor()])
            inputs = torch.stack([transform(Image.fromarray(face)) for face in faces]).squeeze().to(device)
            # check if single dimensional tensor (as needs reshaping)
            if len(list(inputs.size())) == 1:
                inputs = inputs.unsqueeze(0)
            # perform classification
            outputs = mlp(inputs)
            scores, preds = torch.max(outputs,1)
            preds = preds.tolist()
            scores = scores.tolist()
            # set preds with low confidence to n/a
            preds_final = []
            for pred, score in zip(preds,scores):
                if score < -1000:
                    preds_final.append(48)
                else:
                    preds_final.append(pred)
            self.preds = preds_final

    def cartoonify(self, image, faces, boxes):
        for face, box in zip(faces,boxes):
            startx, starty, endx, endy = box
            face = Image.fromarray(face).resize((endx-startx,endy-starty))
            image.paste(face, (startx,starty))
        self.I = image

    def draw_boxes(self, image, boxes, preds):
        # get a font
        fontset = '../font/Arial.ttf'
        fnt = ImageFont.truetype(fontset, 70)
        # Draw faces
        draw = ImageDraw.Draw(image)
        for box, pred in zip(boxes,preds):
            startx, starty, endx, endy = box
            draw.rectangle(box, outline=(255, 255, 255), width=6)
            draw.text((startx+15,starty+15), str(self.class_names[pred]), font=fnt, fill=(255,255,255,128), width=7)
        self.I = image