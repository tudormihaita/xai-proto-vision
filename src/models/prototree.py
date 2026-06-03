import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import PrototypeModel, build_backbone


class ProtoTree(PrototypeModel):
    """ProtoTree implementation for a soft binary prototype tree."""

    def __init__(
        self,
        backbone_name: str = "resnet34",
        num_classes: int = 200,
        depth: int = 6,
        lambda_cluster: float = 0.8,
    ) -> None:
        backbone, feature_dim = build_backbone(backbone_name)
        super().__init__(backbone, num_classes)

        if depth < 1:
            raise ValueError("ProtoTree depth must be >= 1")

        self.depth = depth
        self.num_nodes = 2 ** depth - 1
        self.num_leaves = 2 ** depth
        self.feature_dim = feature_dim
        self.num_prototypes = self.num_nodes
        
        # Vom folosi lambda_cluster pentru noul 'Balance Loss'
        self.lambda_cluster = lambda_cluster

        self.prototypes = nn.Parameter(torch.randn(self.num_nodes, feature_dim) * 0.01)
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves, num_classes) * 0.01)
        
        # TOP PRINCIPLE: Scale redus inițial pentru a permite curgerea gradienților
        self.node_scales = nn.Parameter(torch.ones(self.num_nodes) * 1.0)
        self.node_biases = nn.Parameter(torch.zeros(self.num_nodes))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        path_right, path_left = self._build_leaf_directions()
        self.register_buffer("path_right", path_right)
        self.register_buffer("path_left", path_left)

    def _build_leaf_directions(self) -> tuple[torch.Tensor, torch.Tensor]:
        path_right = torch.zeros(self.num_leaves, self.num_nodes, dtype=torch.float32)
        path_left = torch.zeros(self.num_leaves, self.num_nodes, dtype=torch.float32)
        
        for leaf in range(self.num_leaves):
            code = format(leaf, f"0{self.depth}b")
            node_id = 0
            for bit in code:
                if bit == "1":
                    path_right[leaf, node_id] = 1.0
                else:
                    path_left[leaf, node_id] = 1.0
                node_id = 2 * node_id + 1 + int(bit)
                
        return path_right, path_left

    def _compute_leaf_path_probs(
        self,
        p_right: torch.Tensor,
        p_left: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8
        log_right = torch.log(p_right.clamp(min=eps))
        log_left = torch.log(p_left.clamp(min=eps))

        path_right = self.path_right.to(p_right.device)
        path_left = self.path_left.to(p_left.device)
        
        log_path = torch.matmul(log_right, path_right.T) + torch.matmul(log_left, path_left.T)
        return torch.exp(log_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        self._features = features

        b, c, h, w = features.shape
        patches = features.flatten(2).permute(0, 2, 1)
        self._patches = patches

        proto_norm = F.normalize(self.prototypes, dim=1)
        patch_norm = F.normalize(patches, dim=2)
        patch_sims = torch.matmul(patch_norm, proto_norm.T)
        self._patch_sims = patch_sims.view(b, h, w, self.num_nodes).permute(0, 3, 1, 2)

        node_sims, _ = patch_sims.max(dim=1)
        self._node_sims = node_sims

        p_right = torch.sigmoid(node_sims * self.node_scales + self.node_biases)
        self._p_right = p_right
        p_left = 1.0 - p_right

        path_probs = self._compute_leaf_path_probs(p_right, p_left)
        self._path_probs = path_probs

        logits = path_probs @ self.leaf_logits
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cls_loss = F.cross_entropy(logits, labels)
        
        # TOP PRINCIPLE: Balance Loss (înlocuiește cluster_loss-ul distructiv)
        # Forțăm arborele să trimită aproximativ 50% din batch la stânga și 50% la dreapta la fiecare nod
        mean_routing = self._p_right.mean(dim=0)  # (num_nodes,)
        target_routing = torch.full_like(mean_routing, 0.5)
        balance_loss = F.mse_loss(mean_routing, target_routing)
        
        # Calculăm similaritatea doar ca metrică pentru log-uri (fără gradienți, deci nu distruge rutarea!)
        with torch.no_grad():
            b, hw, c = self._patches.shape
            patch_norm = F.normalize(self._patches, dim=2)
            proto_norm = F.normalize(self.prototypes, dim=1)
            sims = torch.matmul(patch_norm, proto_norm.T)
            sims_flat = sims.view(-1, self.num_nodes)
            min_sims, _ = sims_flat.max(dim=0)
            cluster_metric = -min_sims.mean()
        
        # Folosim factorul lambda_cluster (e.g. 0.8) pentru a pondera necesitatea de echilibru în arbore
        total_loss = cls_loss + self.lambda_cluster * balance_loss
        
        return {
            "total": total_loss,
            "cls": cls_loss,
            "cluster": cluster_metric,  # Păstrăm cheia ca să nu crape scriptul tău de logare
            "balance": balance_loss
        }

    def get_prototype_params(self):
        return [self.prototypes, self.leaf_logits, self.node_scales, self.node_biases]

    def push_prototypes(self, train_loader, device: str | torch.device) -> None:
        self.eval()
        device = torch.device(device)
        self.to(device)

        best_sims = torch.full((self.num_nodes,), -1.0, device=device)
        best_patches = torch.zeros_like(self.prototypes.data, device=device)

        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(device)
                features = self.backbone(images)
                b, c, h, w = features.shape
                patches = features.flatten(2).permute(0, 2, 1)

                patch_norm = F.normalize(patches, dim=2)
                proto_norm = F.normalize(self.prototypes, dim=1)
                sims = torch.matmul(patch_norm, proto_norm.T)
                sims_flat = sims.view(-1, self.num_nodes)

                patch_flat = patches.reshape(-1, c)
                values, indices = sims_flat.max(dim=0)

                update_mask = values > best_sims
                if update_mask.any():
                    best_sims[update_mask] = values[update_mask]
                    best_patches[update_mask] = patch_flat[indices[update_mask]]

        self.prototypes.data.copy_(best_patches)
    
    def post_push_init(self) -> None:
        self.init_leaf_logits_balanced(self.num_classes)
    
    def init_leaf_logits_balanced(self, num_classes: int) -> None:
        with torch.no_grad():
            self.leaf_logits.data.fill_(0.0)
            self.leaf_logits.data.add_(torch.randn_like(self.leaf_logits) * 0.001)

    def explain(self, x: torch.Tensor) -> dict:
        logits = self.forward(x)

        with torch.no_grad():
            p_right = self._p_right
            decisions = (p_right > 0.5).long()

            batch_size = decisions.size(0)
            path_node_ids = torch.zeros(batch_size, self.depth, dtype=torch.long, device=decisions.device)
            routing_probs = [torch.zeros(batch_size, device=decisions.device) for _ in range(self.depth)]
            node_similarities = torch.zeros(batch_size, self.depth, device=decisions.device)
            activation_maps = []
            leaf_reached = torch.zeros(batch_size, dtype=torch.long, device=decisions.device)

            for i in range(batch_size):
                node_id = 0
                bits = 0
                sample_decisions = decisions[i]
                for d in range(self.depth):
                    path_node_ids[i, d] = node_id
                    routing_probs[d][i] = p_right[i, node_id]
                    node_similarities[i, d] = self._node_sims[i, node_id]
                    activation_maps.append(self._patch_sims[i, node_id])
                    bit = sample_decisions[node_id].item()
                    bits = (bits << 1) | bit
                    node_id = 2 * node_id + 1 + bit
                leaf_reached[i] = bits

            h, w = self._patch_sims.shape[-2:]
            maps = torch.stack(activation_maps, dim=0).view(batch_size, self.depth, h, w)
            activation_maps = [maps[:, d, :, :] for d in range(self.depth)]

        return {
            "logits": logits,
            "routing_probs": routing_probs,
            "path_node_ids": path_node_ids,
            "leaf_reached": leaf_reached,
            "node_similarities": node_similarities,
            "activation_maps": activation_maps,
        }