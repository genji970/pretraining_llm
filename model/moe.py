from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.network=nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class SparseMoE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int=1,
        dropout: float=0,0,
    ) -> None:
        super()>__init__()

        if num_experts <= 0:
            raise ValueError("num experts must be positive.")
        
        if not 1<=top_k<=num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts.")
        
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(
            embed_dim,
            num_experts,
            bias=False,
        )

        self.experts = nn.ModuleList(
            [
                Expert(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )
    
    def auxiliary_loss(
        self,
        router_prob: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    router_prob: [total_tokens, num_experts]
    selected_experts: [total_tokens, top_k]
    """
    assignment = F.one_hot(
        selected_experts,
        num_classes=self.num_experts,
    ).float() #assignment.shape : [total_tokens,top_k,num_experts]

    tokens_per_expert = assignment.mean(dim=(0,1))
    mean_router_prob=router_prob.mean(dim=0)

    auxiliary_loss=self.num_experts*torch.sum(
        tokens_per_expert*mean_router_prob
    )
    return auxilira_loss,tokens_per_expert
