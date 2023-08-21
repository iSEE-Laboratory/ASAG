import torch
import torch.nn as nn
import torch.nn.functional as F

def build_activation(type):
    if type == 'relu':
        return nn.ReLU()
    elif type == 'gelu':
        return nn.GELU()
    elif type == 'silu':
        return nn.SiLU()
    else:
        raise NotImplementedError


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activations = nn.ModuleList(build_activation(activation) for _ in range(num_layers - 1))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AdaptiveMixing(nn.Module):
    def __init__(self, in_dim, in_points, n_groups, query_dim=None,
                 out_dim=None, out_points=None, sampling_rate=None, activation='relu'):
        super(AdaptiveMixing, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points//sampling_rate
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim//n_groups
        self.eff_out_dim = out_dim//n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points

        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Sequential(
            nn.Linear(self.query_dim, self.n_groups*self.total_parameters),
        )

        self.out_proj = nn.Linear(
            self.eff_out_dim*self.out_points*self.n_groups, self.query_dim, bias=True
        )

        self.act1 = build_activation(activation)
        self.act2 = build_activation(activation)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator[-1].weight)

    def forward(self, x, query):


        B, N, g, P, C = x.size()
        # batch, num_query, group, point, channel
        G = self.n_groups
        assert g == G

        # query: B, N, C
        # x: B, N, G, Px, Cx
        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*N, G, -1)

        out = x.reshape(B*N, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)

        M = M.reshape(B*N, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*N, G, self.out_points, self.in_points)

        '''adaptive channel mixing
        the process also can be done with torch.bmm
        but for clarity, we use torch.matmul
        '''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act1(out)

        '''adaptive spatial mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act2(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, N, -1)
        out = self.out_proj(out)

        out = query + out

        return out


def sampling_each_level(sample_points: torch.Tensor,
                        value: torch.Tensor,
                        weight=None,
                        n_points=1):
    B1, n_queries, _t, n_groups_points, _ = sample_points.shape
    assert _t == 1
    B2, C_feat, H_feat, W_feat = value.shape
    assert B1 == B2
    B = B1

    n_groups = n_groups_points//n_points
    n_channels = C_feat//n_groups

    sample_points = sample_points \
        .view(B, n_queries, n_groups, n_points, 2) \
        .permute(0, 2, 1, 3, 4).flatten(0, 1)
    sample_points = sample_points*2.0-1.0

    # `sampling_points` now has the shape [B*n_groups, n_queries, n_points, 2]

    value = value.view(B*n_groups, n_channels, H_feat, W_feat)
    out = F.grid_sample(
        value, sample_points,
        mode='bilinear', padding_mode='zeros', align_corners=False,
    )

    # `out`` now has the shape [B*n_groups, C, n_queries, n_points]

    if weight is not None:
        weight = weight.view(B, n_queries, n_groups, n_points) \
            .permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1)
        # `weight`` has the shape [B*n_groups, 1, n_queries, n_points]
        out *= weight

    return out \
        .view(B, n_groups, n_channels, n_queries, n_points) \
        .permute(0, 3, 1, 4, 2)

    # `out`` has shape [B, n_queries, n_groups, n_points, n_channels]


def translate_to_linear_weight(ref: torch.Tensor, num_total, tau=2.0):

    grid = torch.arange(num_total, device=ref.device, dtype=ref.dtype).view(
        *[len(ref.shape)*[1, ]+[-1, ]])

    ref = ref.unsqueeze(-1).clone()
    l2 = (ref-grid).pow(2.0).div(tau).abs().neg()
    weight = torch.softmax(l2, dim=-1)

    return weight


def sampling_3d(
    sample_points: torch.Tensor,
    multi_lvl_values,
    featmap_strides,
    n_points: int = 1,
    num_levels: int = None,
    tau=2.0,
):
    B, n_queries, _t, n_groups_points, _ = sample_points.shape
    assert _t == 1
    B, C_feat, _, _ = multi_lvl_values[0].shape

    n_groups = n_groups_points//n_points
    n_channels = C_feat//n_groups

    if num_levels is None:
        num_levels = len(featmap_strides)

    sample_points_xy = sample_points[..., 0:2] * featmap_strides[0]

    sample_points_z = sample_points[..., 2].clone()
    sample_points_lvl_weight = translate_to_linear_weight(
        sample_points_z, num_levels,
        tau=tau)

    sample_points_lvl_weight_list = sample_points_lvl_weight.unbind(-1)

    out = sample_points.new_zeros(
        B, n_queries, n_groups, n_points, n_channels)

    for i in range(num_levels):
        value = multi_lvl_values[i]
        lvl_weights = sample_points_lvl_weight_list[i]

        stride = featmap_strides[i]

        mapping_size = value.new_tensor(
            [value.size(3), value.size(2)]).view(1, 1, 1, 1, -1) * stride
        normalized_xy = sample_points_xy/mapping_size

        out += sampling_each_level(normalized_xy, value,
                                   weight=lvl_weights, n_points=n_points)

    return out, None



def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])


    lt = torch.max(bboxes1[..., :, None, :2],
                    bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = torch.min(bboxes1[..., :, None, 2:],
                    bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = (rb - lt).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]

    if mode in ['iou', 'giou']:
        union = area1[..., None] + area2[..., None, :] - overlap
    else:
        union = area1[..., None]
    if mode == 'giou':
        enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def position_embedding(token_xyzr, num_feats, temperature=10000):
    assert token_xyzr.size(-1) == 4
    term = token_xyzr.new_tensor([1000, 1000, 1, 1]).view(1, 1, -1)
    token_xyzr = token_xyzr / term
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=token_xyzr.device)
    dim_t = (temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_feats)).view(1, 1, 1, -1)
    pos_x = token_xyzr[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
        dim=4).flatten(2)
    return pos_x


def decode_box(xyzr):
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -1,
                              xyzr[..., 3:4] * 1], dim=-1)
    wh = scale * ratio
    xy = xyzr[..., 0:2]
    roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
    return roi


def make_sample_points(offset, num_group, xyzr):
    '''
        offset_yx: [B, L, num_group*3], normalized by stride

        return: [B, L, num_group, 3]
        '''
    B, L, _ = offset.shape

    offset = offset.view(B, L, 1, num_group, 3)

    roi_cc = xyzr[..., :2]
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -1,
                               xyzr[..., 3:4] * 1], dim=-1)
    roi_wh = scale * ratio

    roi_lvl = xyzr[..., 2:3].view(B, L, 1, 1, 1)

    offset_yx = offset[..., :2] * roi_wh.view(B, L, 1, 1, 2)
    sample_yx = roi_cc.contiguous().view(B, L, 1, 1, 2) + offset_yx

    sample_lvl = roi_lvl + offset[..., 2:3]

    return torch.cat([sample_yx, sample_lvl], dim=-1)
