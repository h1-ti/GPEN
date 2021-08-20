'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import numpy as np
from PIL import Image
import __init_paths
from retinaface.retinaface_detection import RetinaFaceDetection
from face_model.face_gan import FaceGAN
from align_faces import warp_and_crop_face, get_reference_facial_points
from skimage import transform as tf

class FaceEnhancement(object):
    def __init__(self, base_dir='./', size=512, model=None, channel_multiplier=2):
        self.facedetector = RetinaFaceDetection(base_dir)
        self.facegan = FaceGAN(base_dir, size, model, channel_multiplier)
        self.size = size
        self.threshold = 0.9

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)

        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
                (self.size, self.size), inner_padding_factor, outer_padding, default_square)

    def process(self, img):
        facebs, landms = self.facedetector.detect(img)
        
        orig_faces, enhanced_faces = [], []
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4]<self.threshold: continue
            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.size, self.size))
            
            # enhance the face
            ef = self.facegan.process(of)
            
            orig_faces.append(of)
            enhanced_faces.append(ef)
            
            tmp_mask = self.mask
            tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            if min(fh, fw)<100: # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)
            
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
            full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]

        full_mask = full_mask[:, :, np.newaxis]
        img = cv2.convertScaleAbs(img*(1-full_mask) + full_img*full_mask)

        return img, orig_faces, enhanced_faces
        

if __name__=='__main__':
    from face_enhancement import FaceEnhancement
    from tqdm import tqdm
    import os, os.path as osp
    import zipfile
    import glob
    import cv2
    import shutil

    from DFLIMG.DFLPNG import DFLPNG
    from DFLIMG.DFLJPG import DFLJPG

    model = {'name':'GPEN-BFR-512', 'size':512}

    faceset_dir = "/content/drive/MyDrive/faceset"
    indir = "temp_input"
    os.makedirs(indir, exist_ok=True)

    src_names = [osp.splitext(d)[0] for d in os.listdir(faceset_dir)
                    if osp.splitext(d)[1] == ".zip" and "superres" not in d]

    faceenhancer = FaceEnhancement(size=model['size'], model=model['name'], channel_multiplier=2)

    for id, src in enumarate(src_names):
        print("{} / {} [{}]".format(id+1, len(src_names), src))
        zip_path = osp.join(faceset_dir, src+".zip")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(indir)

        files = sorted(glob.glob(osp.join(indir, src, '*.*g')))

        outdir = src+"_superres"
        os.makedirs(outdir, exist_ok=True)

        for n, file in enumerate(tqdm(files[:], total=len(files))):
            filename = osp.basename(file)
            ext = osp.splitext(file)[1]
            if ext == ".jpg":
                dflimg = DFLJPG.load(osp.join(input_dir, filename))
            elif ext == ".png":
                dflimg = DFLPNG.load(osp.join(input_dir, filename))
            else:
                continue

            if not dflimg:
                continue

            im = cv2.imread(file, cv2.IMREAD_COLOR) # BGR
            input_x, input_y = im.shape[:2]
            if not isinstance(im, np.ndarray): print(filename, 'error'); continue
            im = cv2.resize(im, (0,0), fx=2, fy=2)

            img, orig_faces, enhanced_faces = faceenhancer.process(im)
            img = cv2.resize(img, (input_x, input_y))
            cv2.imwrite(osp.join(outdir, filename), img)

            if ext == ".jpg":
                DFLJPG.embed_dfldict (osp.join(outdir, filename), 
                                        {'face_type': dflimg.get_face_type(),
                                            'landmarks': dflimg.get_landmarks(),
                                            'ie_polys' : dflimg.get_ie_polys(),
                                            'source_filename': dflimg.get_source_filename(),
                                            'source_rect': dflimg.get_source_rect(),
                                            'source_landmarks': dflimg.get_source_landmarks(),
                                            'image_to_face_mat': dflimg.get_image_to_face_mat(),
                                            'fanseg_mask' : dflimg.dfl_dict.get ('fanseg_mask', None),
                                            'xseg_mask' : dflimg.dfl_dict.get('xseg_mask', None),
                                            'eyebrows_expand_mod' : None,
                                            'relighted' : None,
                                            "histgram" : None,
                                            "recognition" : dflimg.get_recognition(),
                                        })
            elif ext == ".png":
                dflimg = DFLPNG.embed_dfldict (osp.join(outdir, filename), 
                                        {'face_type': dflimg.get_face_type(),
                                            'landmarks': dflimg.get_landmarks(),
                                            'ie_polys' : dflimg.get_ie_polys(),
                                            'source_filename': dflimg.get_source_filename(),
                                            'source_rect': dflimg.get_source_rect(),
                                            'source_landmarks': dflimg.get_source_landmarks(),
                                            'image_to_face_mat': dflimg.get_image_to_face_mat(),
                                            'fanseg_mask' : dflimg.dfl_dict.get ('fanseg_mask', None),
                                            'xseg_mask' : dflimg.dfl_dict.get('xseg_mask', None),
                                            'eyebrows_expand_mod' : None,
                                            'relighted' : None,
                                            "histgram" : None,
                                            "recognition" : dflimg.get_recognition(),
                                        })


        shutil.make_archive(outdir, 'zip', root_dir=outdir)
        shutil.move(outdir+".zip", faceset_dir)
        shutil.rmtree(outdir)
        
