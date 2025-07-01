import os
import torch
import math
from functools import reduce
import operator
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def batch_interp(queries, xs, ys):
    """
    Vectorized interpolation for batchsize x t times.

    Args:
        queries: Tensor of shape (batchsize, t, n_queries) - query points for each batch and time
        xs: Tensor of shape (batchsize, t, n_points) - x coordinates for each batch and time
        ys: Tensor of shape (batchsize, t, n_points) - y coordinates for each batch and time

    Returns:
        Tensor of shape (batchsize, t, n_queries) - interpolated values
    """
    queries = queries.transpose(1, 2)
    xs = xs.transpose(1, 2)
    ys = ys.transpose(1, 2)

    batchsize, t, n_points = xs.shape
    _, _, n_queries = queries.shape

    # Sort each batch of x and corresponding y
    sorted_xs, indices = torch.sort(xs, dim=2)
    sorted_ys = torch.gather(ys, 2, indices)

    # Find where query values would fit in x (shape: (batchsize, t, n_queries))
    idx_right = torch.searchsorted(sorted_xs, queries)
    idx_left = idx_right - 1

    # Clamp indices to avoid out-of-bounds
    idx_left = torch.clamp(idx_left, 0, n_points - 1)
    idx_right = torch.clamp(idx_right, 0, n_points - 1)

    # Create index tensors for gather
    batch_indices = torch.arange(batchsize, device=xs.device)[:, None, None].expand(batchsize, t, n_queries)
    t_indices = torch.arange(t, device=xs.device)[None, :, None].expand(batchsize, t, n_queries)

    # Gather left/right x and y values
    x_left = sorted_xs[batch_indices, t_indices, idx_left]
    x_right = sorted_xs[batch_indices, t_indices, idx_right]
    y_left = sorted_ys[batch_indices, t_indices, idx_left]
    y_right = sorted_ys[batch_indices, t_indices, idx_right]

    # Avoid division by zero (if x_left == x_right)
    alpha = torch.zeros_like(queries)
    mask = x_left != x_right
    alpha[mask] = (queries[mask] - x_left[mask]) / (x_right[mask] - x_left[mask])

    # Linear interpolation
    return (y_left + alpha * (y_right - y_left)).transpose(1,2)
def gradients(u, x, order=1):
    x = x
    u = u
    if order == 1:
        return torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]
    else:
        return gradients(
            gradients(u, x), x, order=order - 1)
def compute_ntk(model, x):
    model.zero_grad()
    y = model(x)
    grads = []
    for i in range(y.shape[0]):
        grad = torch.autograd.grad(y[i], model.parameters(), retain_graph=True)
        grads.append(torch.cat([g.view(-1) for g in grad]))
    grads = torch.stack(grads)
    ntk = torch.mm(grads, grads.t())
    return ntk

# 自适应调整损失权重
def adaptive_loss_weighting(ntk, mse_loss, physics_loss):
    eigenvalues = torch.linalg.eigvalsh(ntk)
    lambda_mse = 1.0 / eigenvalues[0]
    lambda_physics = 1.0 / eigenvalues[1]
    total_loss = lambda_mse * mse_loss + lambda_physics * physics_loss
    return total_loss
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        if p.is_complex():
            size = p.size() + (2,)
        else:
            size = p.size()
        if size:
            c += reduce(operator.mul, size, 1)
    return c
def get_n_params(model):
    pp = 0  # pp 代表总参数数量
    for p in list(model.parameters()):
        nn = 1
        # 遍历参数张量的每个维度，计算总大小
        for s in list(p.size()):
            nn = nn * s
        pp += nn  # 累加每个参数张量的总元素数量
    return pp
def count_params_with_grad(model):
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:  # Check if gradient is required
            size = param.size()
            total_params += reduce(operator.mul, size, 1) if size else 0 # Handle potential empty size
    return total_params
class LpLoss(object):
    def __init__(self, d=1, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
                                                          self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p)


        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def central_diff_1d(x, h, fix_x_bnd=False):
    dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1)) / (2.0 * h)

    if fix_x_bnd:
        dx[..., 0] = (x[..., 1] - x[..., 0]) / h
        dx[..., -1] = (x[..., -1] - x[..., -2]) / h

    return dx


# x: (*, s1, s2)
# y: (*, s1, s2)
def central_diff_2d(x, h, fix_x_bnd=False, fix_y_bnd=False):
    if isinstance(h, float):
        h = [h, h]

    dx = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2)) / (2.0 * h[0])
    dy = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1)) / (2.0 * h[1])

    if fix_x_bnd:
        dx[..., 0, :] = (x[..., 1, :] - x[..., 0, :]) / h[0]
        dx[..., -1, :] = (x[..., -1, :] - x[..., -2, :]) / h[0]

    if fix_y_bnd:
        dy[..., :, 0] = (x[..., :, 1] - x[..., :, 0]) / h[1]
        dy[..., :, -1] = (x[..., :, -1] - x[..., :, -2]) / h[1]

    return dx, dy


# x: (*, s1, s2, s3)
# y: (*, s1, s2, s3)
def central_diff_3d(x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
    if isinstance(h, float):
        h = [h, h, h]

    dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3)) / (2.0 * h[0])
    dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2)) / (2.0 * h[1])
    dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1)) / (2.0 * h[2])

    if fix_x_bnd:
        dx[..., 0, :, :] = (x[..., 1, :, :] - x[..., 0, :, :]) / h[0]
        dx[..., -1, :, :] = (x[..., -1, :, :] - x[..., -2, :, :]) / h[0]

    if fix_y_bnd:
        dy[..., :, 0, :] = (x[..., :, 1, :] - x[..., :, 0, :]) / h[1]
        dy[..., :, -1, :] = (x[..., :, -1, :] - x[..., :, -2, :]) / h[1]

    if fix_z_bnd:
        dz[..., :, :, 0] = (x[..., :, :, 1] - x[..., :, :, 0]) / h[2]
        dz[..., :, :, -1] = (x[..., :, :, -1] - x[..., :, :, -2]) / h[2]

    return dx, dy, dz



class H1Loss(object):
    def __init__(self, d=1, L=2 * math.pi, reduce_dims=0, reductions='sum', fix_x_bnd=False, fix_y_bnd=False,
                 fix_z_bnd=False):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L] * self.d
        else:
            self.L = L

    def compute_terms(self, x, y, h):
        dict_x = {}
        dict_y = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            x_x = central_diff_1d(x, h[0], fix_x_bnd=self.fix_x_bnd)
            y_x = central_diff_1d(y, h[0], fix_x_bnd=self.fix_x_bnd)

            dict_x[1] = x_x
            dict_y[1] = y_x

        elif self.d == 2:
            dict_x[0] = torch.flatten(x, start_dim=-2)
            dict_y[0] = torch.flatten(y, start_dim=-2)

            x_x, x_y = central_diff_2d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)
            y_x, y_y = central_diff_2d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-2)
            dict_x[2] = torch.flatten(x_y, start_dim=-2)

            dict_y[1] = torch.flatten(y_x, start_dim=-2)
            dict_y[2] = torch.flatten(y_y, start_dim=-2)

        else:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd,
                                            fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd,
                                            fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)

        return dict_x, dict_y

    def uniform_h(self, x):
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j)

        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def abs(self, x, y, h=None):
        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        dict_x, dict_y = self.compute_terms(x, y, h)

        const = math.prod(h)
        diff = const * torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False) ** 2

        for j in range(1, self.d + 1):
            diff += const * torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False) ** 2

        diff = diff ** 0.5

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def rel(self, x, y, h=None):
        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        dict_x, dict_y = self.compute_terms(x, y, h)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False) ** 2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False) ** 2

        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False) ** 2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False) ** 2

        diff = (diff ** 0.5) / (ynorm ** 0.5)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def __call__(self, y_pred, y, h=None, **kwargs):
        return self.rel(y_pred, y, h=h)