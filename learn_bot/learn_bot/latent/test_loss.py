import torch
from torch import nn

weighted_cross_entropy_loss_fn = nn.CrossEntropyLoss(reduction='none', weight=torch.Tensor([10., 1., 1.]))
unweighted_cross_entropy_loss_fn = nn.CrossEntropyLoss(reduction='none')

prediction = torch.Tensor([[1., 1., 1.], [1., 1., 1.]])
ground_truth = torch.Tensor([[1., 0., 0.], [0., 1., 0.]])

weighted_loss = weighted_cross_entropy_loss_fn(prediction, ground_truth)
unweighted_loss = unweighted_cross_entropy_loss_fn(prediction, ground_truth)

print(weighted_loss)
