import os
import sys
sys.path.append('./tools')
from utils import exec_cmd
import random
import numpy as np
import glob
import yaml
import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel

from PIL import Image
from mmpretrain import get_model
import argparse

class Similarity:
    def __init__(self, sim_base_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.sim_base_name = sim_base_name
        if self.sim_base_name == 'clip':
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        elif self.sim_base_name == 'dinov2':
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.base = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        else:
            # Define your own similarity base function
            pass
        self.img_id_pair_to_sim_dict = defaultdict()
        self.cls_id_to_img_id_ls_dict = defaultdict()
        self.img_id_to_feats_dict = defaultdict()

    def extract_feats(self, img):
        if self.sim_base_name == 'clip':
            with torch.no_grad():
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                feats = self.base.get_image_features(**inputs)
        elif self.sim_base_name == 'dinov2':
            with torch.no_grad():
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                outputs = self.base(**inputs)
                feats = outputs.last_hidden_state
                feats = feats.mean(dim=1)
        return feats

    def cos_sim(self, feat0, feat1):
        cos = nn.CosineSimilarity(dim=0)
        sim = cos(feat0, feat1).item()
        return (sim + 1) / 2

    def sim_alpha(self, cls_ls): #, img_id_to_feats_dict, cls_id_to_img_id_ls_dict):
        '''
        Intra-Class Similarity
        '''
        print('\n================================')
        print('\nsim_alpha')
        print('\ncls_ls: ', cls_ls)

        sim_alpha_ls = []
        # =============================
        # Iterate over all classes >>>
        for cls in cls_ls:
            cls = str(cls)
            # print('\nself.cls_id_to_img_id_ls_dict: ', self.cls_id_to_img_id_ls_dict)
            img_id_ls = self.cls_id_to_img_id_ls_dict[cls]
            # print('\nimg_id_ls: ', img_id_ls)

            sim_alpha_ls_cls = []
            # ================================================
            # Iterate over all image pairs within a class >>>
            i = 0
            while i < len(img_id_ls):
                img_id_i = img_id_ls[i]
                j = i + 1
                while j < len(img_id_ls):
                    img_id_j = img_id_ls[j]
                    # cos = nn.CosineSimilarity(dim=0)
                    # sim = cos(self.img_id_to_feats_dict[img_id_i][0], self.img_id_to_feats_dict[img_id_j][0]).item()
                    # sim = (sim + 1) / 2
                    sim = self.cos_sim(self.img_id_to_feats_dict[img_id_i][0], self.img_id_to_feats_dict[img_id_j][0])

                    # Cache
                    if (img_id_i, img_id_j) in self.img_id_pair_to_sim_dict:
                        sim = self.img_id_pair_to_sim_dict[(img_id_i, img_id_j)]
                    else:
                        self.img_id_pair_to_sim_dict[(img_id_i, img_id_j)] = sim
                        self.img_id_pair_to_sim_dict[(img_id_j, img_id_i)] = sim
                    
                    sim_alpha_ls_cls.append(sim)
                    j += 1
                i += 1
            # Iterate over all image pairs within a class <<<
            # ================================================
            sim_alpha_ls.extend(sim_alpha_ls_cls)
        # Iterate over all classes <<<
        # =============================
        sim_alpha = np.mean(sim_alpha_ls)
        print('\nsim_alpha: ', sim_alpha)
        return sim_alpha


    def sim_beta(self, cls_ls): #, img_id_to_feats_dict, cls_id_to_img_id_ls_dict):
        '''
        Inter-Class Similarity
        '''
        print('\n================================')
        print('\nsim_beta')
        print('\ncls_ls: ', cls_ls)
        
        sim_beta_ls = []
        # =============================
        # Iterate over all classes >>>
        cls_i = 0
        while cls_i < len(cls_ls):
            img_id_ls_cls_i = self.cls_id_to_img_id_ls_dict[str(cls_ls[cls_i])]
            # ==================================================
            # Iterate over all classes different from cls_i >>>
            cls_j = cls_i + 1
            while cls_j < len(cls_ls):
                img_id_ls_cls_j = self.cls_id_to_img_id_ls_dict[str(cls_ls[cls_j])]
                # =====================================
                # Iterate over all images in cls_i >>>
                i = 0
                while i < len(img_id_ls_cls_i):
                    img_id_i = img_id_ls_cls_i[i]
                    # =====================================
                    # Iterate over all images in cls_j >>>
                    j = i + 1
                    while j < len(img_id_ls_cls_j):
                        img_id_j = img_id_ls_cls_j[j]
                        # cos = nn.CosineSimilarity(dim=0)
                        # sim = self.cos(self.img_id_to_feats_dict[img_id_i][0], self.img_id_to_feats_dict[img_id_j][0]).item()
                        # sim = (sim + 1) / 2
                        sim = self.cos_sim(self.img_id_to_feats_dict[img_id_i][0], self.img_id_to_feats_dict[img_id_j][0])

                        # Cache
                        if (img_id_i, img_id_j) in self.img_id_pair_to_sim_dict:
                            sim = self.img_id_pair_to_sim_dict[(img_id_i, img_id_j)]
                        else:
                            self.img_id_pair_to_sim_dict[(img_id_i, img_id_j)] = sim
                            self.img_id_pair_to_sim_dict[(img_id_j, img_id_i)] = sim

                        sim_beta_ls.append(sim)
                        j += 1
                    # Iterate over all images in cls_j <<<
                    # =====================================
                    i += 1
                    # print(f'\ncls_i: {cls_i}, cls_j: {cls_j}, i: {i}, j: {j}')

                # Iterate over all images in cls_i <<<
                # =====================================
                cls_j += 1
            # Iterate over all classes different from cls_i <<<
            # ==================================================
            cls_i += 1
        # Iterate over all classes <<<
        # =============================
        sim_beta = np.mean(sim_beta_ls)
        print('\nsim_beta: ', sim_beta)
        return sim_beta


    def sim_SS(self, cls_ls): #, img_id_to_feats_dict, cls_id_to_img_id_ls_dict):
        '''
        Nearest Inter-Class Similarity
        SimSS: Similarity-Based Silhouette Score
        '''
        print('\n================================')
        print('\nsim_SS')
        print('\ncls_ls: ', cls_ls)

        sim_SS_ls = []
        # =============================
        # Iterate over all classes >>>
        cls_a_p = 0
        while cls_a_p < len(cls_ls):
            img_id_ls_cls_a_p = self.cls_id_to_img_id_ls_dict[str(cls_ls[cls_a_p])]

            # =======================================
            # Iterate over all images in cls_a_p >>>
            i = 0
            while i < len(img_id_ls_cls_a_p):
                img_id_i = img_id_ls_cls_a_p[i]
                # print(f'\ni: {i}') 
                # ============================================================
                # a_p(i): Iterate over all images in cls_a_p other than i >>>
                a_p_ls, j = [], 0
                while j < len(img_id_ls_cls_a_p):
                    if i != j:
                        img_id_j = img_id_ls_cls_a_p[j]
                        a_p_ls.append(self.cos_sim(self.img_id_to_feats_dict[img_id_i][0], self.img_id_to_feats_dict[img_id_j][0]))
                    j += 1
                    # print(f'\ni: {i}, j: {j}')
                a_p = np.mean(a_p_ls)
                # a_p(i): Iterate over all images in cls_a_p other than i >>>
                # ============================================================
 
                # ====================================================
                # b_p(i): Iterate over all classes other than a_p >>>
                sim_c_p_ls = [] # a list of similarity scores for sample i and other classes
                cls_c_p = 0 # cls_c_p: class index of the class other than a_p
                while cls_c_p < len(cls_ls):
                    if cls_c_p != cls_a_p: # cls_c_p is a different class than cls_a_p
                        img_id_ls_cls_c_p = self.cls_id_to_img_id_ls_dict[str(cls_ls[cls_c_p])]
                        # =================================================
                        # Iterate over all images in the class cls_c_p >>>
                        sim_c_p_ls_cls_c_p = []
                        for img_id_c_p in img_id_ls_cls_c_p:
                            sim_c_p_ls_cls_c_p.append(self.cos_sim(self.img_id_to_feats_dict[img_id_i][0], self.img_id_to_feats_dict[img_id_c_p][0]))
                        # Iterate over all images in the class cls_c_p <<<
                        # =================================================
                        sim_c_p_ls.append(np.mean(sim_c_p_ls_cls_c_p))
                    cls_c_p += 1

                b_p = max(sim_c_p_ls)
                # b_p(i): Iterate over all classes other than a_p <<<
                # ====================================================
 
                # Silhouette Score
                sim_SS = (a_p - b_p) / max(a_p, b_p) 
                sim_SS_ls.append(sim_SS)

                i += 1
            # Iterate over all images in cls_a_p <<<
            # =======================================
            cls_a_p +=1
        # Iterate over all classes <<<
        # =============================
        return np.mean(sim_SS_ls)

 
    def custom_sim(self, cls_ls):
        '''
        Define Your Custom Similarity Metric
        '''
        pass
