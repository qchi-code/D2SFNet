import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import apply_dct_to_tensor
from .attention import MultiHeadAttention, MultiLayerPerceptron, AttentionFusion
from .encoder import Res50Encoder, Res50Encoder_f, Res101Encoder, Res101Encoder_f
from matplotlib import pyplot as plt


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.encoder_f = Res101Encoder_f(replace_stride_with_dilation=[True, True, False],
                                    pretrained_weights=pretrained_weights)  # or "resnet101"

        # self.encoder = Res50Encoder(replace_stride_with_dilation=[True, True, False],
        #                              pretrained_weights=pretrained_weights)  # or "resnet101"
        # self.encoder_f = Res50Encoder_f(replace_stride_with_dilation=[True, True, False],
        #                             pretrained_weights=pretrained_weights)  # or "resnet101"
        self.scaler = 20.0
        self.device = torch.device('cuda')
        self.criterion = nn.NLLLoss()
        self.fg_sampler = np.random.RandomState(1289)
        self.fg_num = 5  # number of foreground partitions
        self.Fusion = AttentionFusion(in_channels=512)

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, aux=None, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W
        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)  # torch.Size([2, 3, 257, 257])
        imgs_concat = imgs_concat[:, :, :256, :256]

        imgs_concat_f = apply_dct_to_tensor(imgs_concat[:, 0:1, :, :])  # 8=torch.Size([2, 64, 32, 32])  4=torch.Size([2, 16, 64, 64])
        img_fts_f = self.encoder_f(imgs_concat_f.to(self.device))  # torch.Size([2, 512, 32, 32])
        img_fts_f = F.interpolate(img_fts_f, size=(64, 64), mode='bilinear', align_corners=True)
        #################
        img_fts, tao = self.encoder(imgs_concat)  # torch.Size([2, 512, 64, 64])
        ################transformer
        # img_fts = self.Tran(img_fts, img_fts_f)
        ###############
        img_fts = self.Fusion(img_fts, img_fts_f)
        # result = torch.cat((img_fts, img_fts_f), dim=1)
        # img_fts = self.reduce1(result)
        ###################
        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])

        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])

        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        #################
        if aux != None :
            supp_aux = []
            supp_aux.append(aux[0])
            qry_aux = aux[1]
            imgs_aux = torch.cat([torch.cat(way, dim=0) for way in supp_aux]
                                    + [torch.cat(qry_aux, dim=0), ], dim=0)  # torch.Size([2, 3, 257, 257])
            imgs_aux = imgs_aux[:, :, :256, :256]  # torch.Size([2, 3, 256, 256])
            imgs_concat_f_aux = apply_dct_to_tensor(imgs_aux[:, 0:1, :, :])  # 8=torch.Size([2, 64, 32, 32])  4=torch.Size([2, 16, 64, 64])
            img_fts_f_aux = self.encoder_f(imgs_concat_f_aux.to(self.device))
            img_fts_f_aux = F.interpolate(img_fts_f_aux, size=(64, 64), mode='bilinear', align_corners=True)
            #################
            img_aux, ta_aux = self.encoder(imgs_aux)

            # result_aux = torch.cat((img_aux, img_fts_f_aux), dim=1)
            # img_fts_aux = self.reduce1(result_aux)
            img_fts_aux = self.Fusion(img_aux, img_fts_f_aux)
            supp_fts_aux = img_fts_aux[:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
                supp_bs, self.n_ways, self.n_shots, -1, *img_fts_aux.shape[-2:])

            qry_fts_aux = img_fts_aux[self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
                qry_bs, self.n_queries, -1, *img_fts_aux.shape[-2:])
        #################

        # Compute loss #
        # align_loss = torch.zeros(1).to(self.device)
        intra_domain_loss = torch.zeros(1).to(self.device)
        # inter_domain_loss = torch.zeros(1).to(self.device)
        outputs = []
        outputs_aux = []
        for epi in range(supp_bs):
            # calculate coarse query prototype
            supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]

            fg_prototypes = self.getPrototype(supp_fts_)  # the coarse foreground

            qry_pred = torch.stack(
                [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

            # Combine predictions of different feature maps #
            qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)

            preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)

            outputs.append(preds)

            if aux != None :
                supp_fts_aux = [[self.getFeatures(supp_fts_aux[[epi], way, shot], supp_mask[[epi], way, shot])
                              for shot in range(self.n_shots)] for way in range(self.n_ways)]

                fg_prototypes_aux = self.getPrototype(supp_fts_aux)  # the coarse foreground

                qry_pred_aux = torch.stack(
                    [self.getPred(qry_fts_aux[epi], fg_prototypes_aux[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

                # Combine predictions of different feature maps #
                qry_pred_up_aux = F.interpolate(qry_pred_aux, size=img_size, mode='bilinear', align_corners=True)

                preds_aux = torch.cat((1.0 - qry_pred_up_aux, qry_pred_up_aux), dim=1)

                outputs_aux.append(preds_aux)
            if train:
                fg_partition_prototypes = [[self.compute_multiple_prototypes(
                    self.fg_num, supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], self.fg_sampler)
                    for shot in range(self.n_shots)] for way in range(self.n_ways)]

                intra_domain_loss = self.intra_domain_shift(fg_partition_prototypes[0][0][0], qry_fts[0], qry_mask)
                # intra_domain_loss = torch.zeros(1).to(self.device)

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])
        if aux != None :
            output_aux = torch.stack(outputs_aux, dim=1)
            output_aux = output_aux.view(-1, *output_aux.shape[2:])
            output = torch.cat((output, output_aux), dim=0)

        return output, intra_domain_loss / supp_bs

    def Tran(self, q, s):
        # q and s : 1，1024，256
        T = self.SKConv(q, s)
        return T

    def intra_domain_shift(self, output_x, output_y, y_label):
        I = np.identity(int(output_x.size(1)))
        I = np.expand_dims(I, axis=0)
        I = torch.from_numpy(I).cuda()
        S = output_x.unsqueeze(-1)
        S_t = S.transpose(2, 1)
        S1 = S_t @ S  # torch.Size([9, 1, 1])
        if torch.any(S1 == 0):
             print("S1:", S1)
             return torch.zeros(1).to(self.device)
        try:
             S1_I = torch.linalg.inv(S1)
        except RuntimeError as e:
             print("S1:", S1)
             return torch.zeros(1).to(self.device)
        # S1_I = torch.linalg.inv(S1)
        S_star = S1_I @ S_t
        P = I - (S @ S_star) + 1e-5
        ######
        mask = y_label.float()
        mask = mask.unsqueeze(0)
        mask = F.interpolate(mask, size=(64, 64), mode='bilinear', align_corners=False)
        mask_flat = mask.view(-1)
        valid_indices = (mask_flat > 0).nonzero(as_tuple=True)[0]
        a_flat = output_y.view(1, 512, -1)
        a_valid = a_flat[:, :, valid_indices]
        R = a_valid.squeeze(0).transpose(0, 1).double()

        intra_domain_loss_list = []
        for i in range(R.size(0)):
            inner_products = torch.matmul(P, R[i])
            norms = torch.norm(inner_products, p=2, dim=1)
            min_norm = torch.min(norms)
            intra_domain_loss_list.append(min_norm)
        if len(intra_domain_loss_list) == 0:
            print("intra_domain_loss_list is empty")
            return torch.zeros(1).to(self.device)
        else:
            intra_domain_loss = max(intra_domain_loss_list)
        return intra_domain_loss

    def compute_multiple_prototypes(self, fg_num, sup_fts, sup_fg, sampler):
        """

        Parameters
        ----------
        fg_num: int
            Foreground partition numbers
        sup_fts: torch.Tensor
             [B, C, h, w], float32
        sup_fg: torch. Tensor
             [B, h, w], float32 (0,1)
        sampler: np.random.RandomState

        Returns
        -------
        fg_proto: torch.Tensor
            [B, k, C], where k is the number of foreground proxies

        """

        B, C, h, w = sup_fts.shape  # B=1, C=512
        fg_mask = F.interpolate(sup_fg.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear')
        fg_mask = fg_mask.squeeze(0).bool()  # [B, h, w] --> bool
        batch_fg_protos = []

        for b in range(B):
            fg_protos = []

            fg_mask_i = fg_mask[b]  # [h, w]

            # Check if zero
            with torch.no_grad():
                if fg_mask_i.sum() < fg_num:
                    fg_mask_i = fg_mask[b].clone()  # don't change original mask
                    fg_mask_i.view(-1)[:fg_num] = True

            # Iteratively select farthest points as centers of foreground local regions
            all_centers = []
            first = True
            pts = torch.stack(torch.where(fg_mask_i), dim=1)
            for _ in range(fg_num):
                if first:
                    i = sampler.choice(pts.shape[0])
                    first = False
                else:
                    dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                    # choose the farthest point
                    i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                pt = pts[i]  # center y, x
                all_centers.append(pt)

            # Assign fg labels for fg pixels
            dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
            fg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

            # Compute fg prototypes
            fg_feats = sup_fts[b].permute(1, 2, 0)[fg_mask_i]  # [N, C]
            for i in range(fg_num):
                proto = fg_feats[fg_labels == i].mean(0)  # [C]
                fg_protos.append(proto)

            fg_protos = torch.stack(fg_protos, dim=1)  # [C, k]
            batch_fg_protos.append(fg_protos)
        fg_proto = torch.stack(batch_fg_protos, dim=0).transpose(1, 2)  # [B, k, C]

        return fg_proto

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getFeatures_bg(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        mask = 1 - mask
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [self.getFeatures(qry_fts, pred_mask[way + 1])]
                fg_prototypes = self.getPrototype([qry_fts_])

                # Get predictions
                supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way],
                                         self.thresh_pred[way])  # N x Wa x H' x W'
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                          align_corners=True)

                # Combine predictions of different feature maps
                preds = supp_pred
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss