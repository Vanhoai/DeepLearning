import torch

from .module import HSModule


class HSLoss(HSModule): ...


class HSMSELoss(HSLoss):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore
        self._cache["pred"] = pred
        self._cache["target"] = target
        return torch.mean((pred - target) ** 2)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        pred = self._cache["pred"]
        target = self._cache["target"]
        N = pred.shape[0]
        return 2 * (pred - target) / N


class HSCrossEntropyLoss(HSLoss):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore
        exp_pred = torch.exp(pred - torch.max(pred, dim=1, keepdim=True).values)
        softmax = exp_pred / torch.sum(exp_pred, dim=1, keepdim=True)

        # Convert target to one-hot
        N, C = pred.shape
        target_onehot = torch.zeros(N, C)
        target_onehot[torch.arange(N), target.long()] = 1

        # Cross entropy
        loss = -torch.mean(torch.sum(target_onehot * torch.log(softmax + 1e-8), dim=1))

        # Cache
        self._cache["softmax"] = softmax
        self._cache["target"] = target_onehot

        return loss

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        softmax = self._cache["softmax"]
        target = self._cache["target"]

        N = softmax.shape[0]
        return (softmax - target) / N


class HSBCELoss(HSLoss):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore
        self._cache["pred"] = pred
        self._cache["target"] = target
        return -torch.mean(
            target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8)
        )

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        pred = self._cache["pred"]
        target = self._cache["target"]
        N = pred.shape[0]
        return (-(target / (pred + 1e-8)) + (1 - target) / (1 - pred + 1e-8)) / N
