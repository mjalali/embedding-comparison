import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            **kwargs
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False, **kwargs):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


class AlignmentClipLoss(ClipLoss):
    def __init__(
            self,
            clip_loss_weight,
            custom_loss_weight,  # Add custom loss weight parameter
            ref_features=None,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.custom_loss_weight = custom_loss_weight
        self.ref_features = ref_features

        # Operator norm variables
        self.right_eigvec = None
        self.left_eigvec = None

    def forward(self, image_features, text_features, logit_scale, model, data, output_dict=False, compute_align_loss=True):
        # Initialize losses
        clip_loss = torch.tensor(0, dtype=image_features.dtype, device=image_features.device)
        custom_loss = torch.tensor(0, dtype=image_features.dtype, device=image_features.device)

        # Calculate CLIP Loss
        if True or self.clip_loss_weight > 0:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            # clip_loss = self.clip_loss_weight * clip_loss
            clip_loss = 0.5 * clip_loss

        # Calculate Custom Loss (Expectation of squared cosine similarity)
        # if compute_align_loss or self.custom_loss_weight > 0:
        if compute_align_loss:
            custom_loss = 1 * self.alignment_loss(model, data=data, num_pairs=512, device=image_features.device)
            # custom_loss = self.custom_loss_weight * custom_loss
            custom_loss = 0.5 * custom_loss
            logging.info(f'custom loss: {custom_loss}')

        # Return losses as a dictionary if requested
        if output_dict:
            return {
                "contrastive_loss": clip_loss,
                "custom_loss": custom_loss
            }

        return clip_loss, custom_loss

    def alignment_loss(self, model, data, norm='op', num_pairs=256, batch_size=32, device='cuda', dtype=torch.float):
        import numpy as np
        # ref_features = np.load('/home/student/Documents/evaluation/t2i/imagenette_text_label_on_image/by_true_class/true_class_dino.npz')['dino_features']
        if self.ref_features is None:
            self.ref_features = np.load('/home/student/Documents/evaluation/t2i/COCO/dataset_2017/train2017_dino_csv.npz')['dino_features']
        dataset = data['train'].dataloader.dataset
        align_loss = 0

        if norm == 'fro':
            random_pairs = np.random.randint(len(dataset), size=(num_pairs, 2))

            kxx = torch.zeros(num_pairs, device=device)
            for start in range(0, random_pairs.shape[0], batch_size):
                end = min(start + batch_size, num_pairs)
                index_i, index_j = random_pairs[start:end][:,0], random_pairs[start:end][:,1]
                samples_i = torch.zeros((batch_size, *dataset[0][0].shape))
                samples_j = torch.zeros((batch_size, *dataset[0][0].shape))
                for k in range(len(index_i)):
                    samples_i[k] = dataset[index_i[k]][0]
                    samples_j[k] = dataset[index_j[k]][0]
                kxx[start:end] = F.cosine_similarity(
                    model(samples_i.to(device=device, non_blocking=True))['image_features'],
                    model(samples_j.to(device=device, non_blocking=True))['image_features']
                )
            torch.cuda.empty_cache()

            kyy = F.cosine_similarity(torch.from_numpy(self.ref_features[random_pairs[:, 0]]).to(device=device),
                                torch.from_numpy(self.ref_features[random_pairs[:, 1]]).to(device=device))
            align_loss = torch.sum((kxx - kyy) ** 2) / num_pairs

        elif norm == 'op':  # Operator norm
            def power_method(V, b=None, num_iterations=20, tolerance=1e-6):
                V = V
                # Randomly initialize a vector
                if b is None:
                    b_k = torch.rand(V.shape[1], device=V.device)
                    num_iterations = 1000  # TODO need to iterate more on the first round
                else:
                    b_k = b
                prev_b = None
                for _ in range(num_iterations):
                    b_k1 = V @ b_k
                    b_k1 = b_k1 / torch.norm(b_k1)
                    # if torch.norm(b_k1 - b_k) < tolerance:
                    #     break
                    b_k = b_k1


                eigenvalue = torch.dot(b_k.T, V @ b_k) / torch.dot(b_k.T, b_k).float()

                return b_k

            random_pairs = np.random.randint(len(dataset), size=(num_pairs, 2))
            x = torch.stack([torch.tensor(self.ref_features[i[0]]).to(device=device, non_blocking=True) for i in random_pairs]).float()
            Vx = x / torch.norm(x, dim=1, keepdim=True)  # Normalize Vx

            output_features = []
            for start in range(0, len(random_pairs), batch_size):
                end = min(start + batch_size, len(random_pairs))
                input_batch = torch.cat([
                    dataset[random_pairs[i][1]][0].unsqueeze(dim=0).to(device=device, non_blocking=True)
                    for i in range(start, end)
                ])
                output_features.append(model(input_batch)['image_features'].float())

            Vy = torch.cat(output_features)
            Vy = Vy / torch.norm(Vy, dim=1, keepdim=True)
            del output_features
            torch.cuda.empty_cache()

            # y = torch.stack([
            #     model(dataset[i[1]][0].unsqueeze(dim=0).to(device=device, non_blocking=True))['image_features'].squeeze().float()
            #     for i in random_pairs
            # ])
            # Vy = y / torch.norm(y, dim=1, keepdim=True)

            matrix_first_row = torch.hstack([Vx.T @ Vx, Vx.T @ Vy])
            matrix_second_row = torch.hstack([-Vy.T @ Vx, -Vy.T @ Vy])
            V = torch.vstack([matrix_first_row, matrix_second_row])
            V = (1 / Vx.shape[0]) * V
            # torch.eye(V.shape[0], device=device) +

            with torch.no_grad():
                #  Power method to compute left and right eigenvectors
                self.right_eigvec = power_method(V, b=self.right_eigvec)
                self.left_eigvec = power_method(V.T, b=self.left_eigvec)

            # Compute alignment loss
            align_loss = torch.abs((self.left_eigvec.T @ V @ self.right_eigvec) / (self.left_eigvec.T @ self.right_eigvec))

        torch.cuda.empty_cache()
        return align_loss

#
# class AlignmentCoCaLoss(CoCaLoss):
#     def __init__(
#             self,
#             caption_loss_weight,
#             clip_loss_weight,
#             custom_loss_weight,  # Add custom loss weight parameter
#             ref_features,
#             pad_id=0,  # pad_token for open_clip custom tokenizer
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#             rank=0,
#             world_size=1,
#             use_horovod=False,
#     ):
#         super().__init__(
#             local_loss=local_loss,
#             gather_with_grad=gather_with_grad,
#             cache_labels=cache_labels,
#             rank=rank,
#             world_size=world_size,
#             use_horovod=use_horovod
#         )
#
#         self.clip_loss_weight = clip_loss_weight
#         self.caption_loss_weight = caption_loss_weight
#         self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
#         self.custom_loss_weight = custom_loss_weight
#         self.ref_features = ref_features
#
#     def forward(self, image_features, text_features, logits, labels, logit_scale, model, data, output_dict=False, compute_align_loss=True):
#         # Initialize losses
#         clip_loss = torch.tensor(0, dtype=image_features.dtype, device=image_features.device)
#         custom_loss = torch.tensor(0, dtype=image_features.dtype, device=image_features.device)
#
#         # Calculate CLIP Loss
#         if self.clip_loss_weight:
#             clip_loss = super().forward(image_features, text_features, logit_scale)
#             clip_loss = 0.45 * self.clip_loss_weight * clip_loss
#
#         caption_loss = self.caption_loss(
#             logits.permute(0, 2, 1),
#             labels,
#         )
#         caption_loss = 0.45 * caption_loss * self.caption_loss_weight
#
#         # Calculate Custom Loss (Expectation of squared cosine similarity)
#         # if compute_align_loss or self.custom_loss_weight > 0:
#         if compute_align_loss:
#             custom_loss = 20 * self.alignment_loss(model, data=data, num_pairs=128)
#             # custom_loss = self.custom_loss_weight * custom_loss
#             custom_loss = 0.1 * custom_loss
#             logging.info(f'custom loss: {custom_loss}')
#
#         # Return losses as a dictionary if requested
#         if output_dict:
#             return {
#                 "contrastive_loss": clip_loss,
#                 "caption_loss": caption_loss,
#                 "custom_loss": custom_loss
#             }
#
#         return clip_loss, caption_loss, custom_loss
#
#     def alignment_loss(self, model, data, num_pairs=512, batch_size=32):
#         import numpy as np
#         # ref_features = np.load('/home/student/Documents/evaluation/t2i/imagenette_text_label_on_image/by_true_class/true_class_dino.npz')['dino_features']
#         ref_features = np.load('/home/student/Documents/evaluation/t2i/COCO/dataset_2017/train2017_dino_csv.npz')['dino_features']
#         dataset = data['train'].dataloader.dataset
#         align_loss = 0
#
#         random_pairs = np.random.randint(len(dataset), size=(num_pairs, 2))
#
#         kxx = torch.zeros(num_pairs, device='cuda:1')
#         for start in range(0, random_pairs.shape[0], batch_size):
#             end = min(start + batch_size, num_pairs)
#             index_i, index_j = random_pairs[start:end][:,0], random_pairs[start:end][:,1]
#             samples_i = torch.zeros((batch_size, *dataset[0][0].shape))
#             samples_j = torch.zeros((batch_size, *dataset[0][0].shape))
#             for k in range(len(index_i)):
#                 samples_i[k] = dataset[index_i[k]][0]
#                 samples_j[k] = dataset[index_j[k]][0]
#             kxx[start:end] = F.cosine_similarity(
#                 model(samples_i.to(device='cuda:1', non_blocking=True))['image_features'],
#                 model(samples_j.to(device='cuda:1', non_blocking=True))['image_features']
#             )
#         torch.cuda.empty_cache()
#
#         kyy = F.cosine_similarity(torch.from_numpy(ref_features[random_pairs[:, 0]]).to(device='cuda:1'),
#                             torch.from_numpy(ref_features[random_pairs[:, 1]]).to(device='cuda:1'))
#         align_loss = torch.sum((kxx - kyy) ** 2) / num_pairs
#         torch.cuda.empty_cache()
#         return align_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss
