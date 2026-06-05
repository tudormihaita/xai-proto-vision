import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import PrototypeModel, build_backbone

class ProtoTree(PrototypeModel):
    """ProtoTree implementation optimized with 1x1 Convolutions and Symmetry Breaking."""

    def __init__(
        self,
        backbone_name: str = "resnet34",
        num_classes: int = 200,
        depth: int = 6,
        lambda_cluster: float = 0.8,
        temperature: float = 1.0, 
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
        self.lambda_cluster = lambda_cluster
        self.temperature = temperature

        # Inițializare Xavier/He pentru prototipuri
        self.prototypes = nn.Parameter(torch.randn(self.num_nodes, feature_dim) / (feature_dim ** 0.5))
        
        # Logits inițiale pentru frunze (clasificatorul final)
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves, num_classes) * 0.01)
        
        # REPARAȚIE CRITICĂ 1: Rupem Simetria (The Symmetric Saddle Point)
        # Folosim un scale mediu (5.0) și adăugăm zgomot în bias (0.5) 
        # ca arborele să fie curajos și să o ia la stânga/dreapta de la prima epocă!
        self.node_scales = nn.Parameter(torch.ones(self.num_nodes) * 5.0)
        self.node_biases = nn.Parameter(torch.randn(self.num_nodes) * 0.5)

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

    def _compute_leaf_path_probs(self, p_right: torch.Tensor, p_left: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        log_right = torch.log(p_right.clamp(min=eps))
        log_left = torch.log(p_left.clamp(min=eps))

        path_right = self.path_right.to(p_right.device)
        path_left = self.path_left.to(p_left.device)
        
        log_path = torch.matmul(log_right, path_right.T) + torch.matmul(log_left, path_left.T)
        return torch.exp(log_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        b, c, h, w = features.shape

        # OPTIMIZARE VITEZĂ & RAM: Calculul similarităților prin Convoluție 1x1 
        features_norm = F.normalize(features, dim=1) # (B, C, H, W)
        proto_norm = F.normalize(self.prototypes, dim=1).view(self.num_nodes, c, 1, 1)
        
        self._patch_sims = F.conv2d(features_norm, proto_norm) # (B, num_nodes, H, W)

        # Extragem scorul cel mai mare cu un max-pool global
        self._node_sims = F.adaptive_max_pool2d(self._patch_sims, (1, 1)).view(b, self.num_nodes)

        # REPARAȚIE: torch.abs(self.node_scales) previne inversarea logicii
        scaled_sims = (self._node_sims * torch.abs(self.node_scales) + self.node_biases) / self.temperature
        p_right = torch.sigmoid(scaled_sims)
        self._p_right = p_right
        p_left = 1.0 - p_right

        path_probs = self._compute_leaf_path_probs(p_right, p_left)
        self._path_probs = path_probs

        logits = path_probs @ self.leaf_logits
        return logits

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        cls_loss = F.cross_entropy(logits, labels)
        
        # Balance loss ponderat exponențial în funcție de adâncimea nivelului
        mean_routing = self._p_right.mean(dim=0)
        target_routing = torch.full_like(mean_routing, 0.5)
        node_mse = (mean_routing - target_routing) ** 2
        
        weights = torch.zeros(self.num_nodes, device=node_mse.device)
        node_id = 0
        for d in range(self.depth):
            nodes_in_level = 2 ** d
            # Nodurile de sus sunt critice (w=1.0). Frunzele contează mai puțin.
            weights[node_id : node_id + nodes_in_level] = 0.5 ** d
            node_id += nodes_in_level
            
        balance_loss = torch.sum(node_mse * weights) / torch.sum(weights)
        
        with torch.no_grad():
            max_sims, _ = self._node_sims.max(dim=0) # Refolosim calculele din forward!
            cluster_metric = -max_sims.mean()
        
        total_loss = cls_loss + self.lambda_cluster * balance_loss
        
        return {
            "total": total_loss,
            "cls": cls_loss,
            "cluster": cluster_metric, 
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

                # Același truc rapid 1x1 Conv pentru a găsi patch-urile de înlocuit
                features_norm = F.normalize(features, dim=1)
                proto_norm = F.normalize(self.prototypes, dim=1).view(self.num_nodes, c, 1, 1)
                
                patch_sims = F.conv2d(features_norm, proto_norm)
                patch_sims_flat = patch_sims.view(b, self.num_nodes, -1) 
                
                max_sims_per_img, max_indices_per_img = patch_sims_flat.max(dim=2) 
                batch_max_sims, batch_max_indices = max_sims_per_img.max(dim=0) 

                update_mask = batch_max_sims > best_sims
                if update_mask.any():
                    features_flat = features.view(b, c, -1) 
                    
                    for node_idx in torch.where(update_mask)[0]:
                        best_sims[node_idx] = batch_max_sims[node_idx]
                        b_idx = batch_max_indices[node_idx]
                        spatial_idx = max_indices_per_img[b_idx, node_idx]
                        best_patches[node_idx] = features_flat[b_idx, :, spatial_idx]

        self.prototypes.data.copy_(best_patches)
    
    def post_push_init(self) -> None:
        # REPARAȚIE CRITICĂ 2: Oprim Amnezia!
        # Rețeaua are nevoie de stratul ei liniar (leaf_logits) învățat timp de 80 de epoci.
        # NU mai suprascriem cu zero.
        pass
    
    def init_leaf_logits_balanced(self, num_classes: int) -> None:
        # Păstrăm funcția doar pentru compatibilitate cu interfața de bază, dar nu o mai apelăm la push.
        pass

    def explain(self, x: torch.Tensor) -> dict:
        logits = self.forward(x)

        with torch.no_grad():
            p_right = self._p_right
            decisions = (p_right > 0.5).long()
            batch_size = decisions.size(0)
            
            path_node_ids = torch.zeros(batch_size, self.depth, dtype=torch.long, device=decisions.device)
            routing_probs = [torch.zeros(batch_size, device=decisions.device) for _ in range(self.depth)]
            node_similarities = torch.zeros(batch_size, self.depth, device=decisions.device)
            leaf_reached = torch.zeros(batch_size, dtype=torch.long, device=decisions.device)
            
            activation_maps_lists = [[] for _ in range(self.depth)]

            for i in range(batch_size):
                node_id = 0
                bits = 0
                sample_decisions = decisions[i]
                for d in range(self.depth):
                    path_node_ids[i, d] = node_id
                    routing_probs[d][i] = p_right[i, node_id]
                    node_similarities[i, d] = self._node_sims[i, node_id]
                    activation_maps_lists[d].append(self._patch_sims[i, node_id])
                    
                    bit = sample_decisions[node_id].item()
                    bits = (bits << 1) | bit
                    node_id = 2 * node_id + 1 + bit
                leaf_reached[i] = bits

            activation_maps = [torch.stack(depth_maps, dim=0) for depth_maps in activation_maps_lists]

        return {
            "logits": logits,
            "routing_probs": routing_probs,
            "path_node_ids": path_node_ids,
            "leaf_reached": leaf_reached,
            "node_similarities": node_similarities,
            "activation_maps": activation_maps,
        }