import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import average_precision_score


def init_module_weights(m, std: float = 0.02):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TemporalBatchMixin:
    def _forward(self, x):
        raise NotImplementedError

    def forward(self, x):
        assert x.ndim in [4, 5], "Only 4D or 5D tensors supported"
        if x.ndim == 5:
            b = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            out = self._forward(x)
            out = rearrange(out, "(b t) c h w -> b c t h w", b=b)
            return out
        return self._forward(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet5(TemporalBatchMixin, nn.Module):
    def __init__(self, in_d, h_d, out_d, s1=1, s2=1, s3=1, avg_pool=False):
        super().__init__()
        self.avg_pool = avg_pool
        self.conv1 = nn.Conv2d(in_d, h_d, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(h_d, h_d, stride=s1)
        self.layer2 = ResidualBlock(h_d, h_d * 2, stride=s2)
        self.layer3 = ResidualBlock(h_d * 2, out_d, stride=s3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avg_pool else nn.Identity()

    def _forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        if self.avg_pool:
            out = out.flatten(1)
        return out


class ResUNet(TemporalBatchMixin, nn.Module):
    def __init__(self, in_d, h_d, out_d, is_rnn=False):
        super().__init__()
        self.is_rnn = is_rnn
        self.conv1 = nn.Conv2d(in_d, h_d, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)

        self.enc1 = ResidualBlock(h_d, h_d, stride=1)
        self.enc2 = ResidualBlock(h_d, 2 * h_d, stride=2)
        self.enc3 = ResidualBlock(2 * h_d, 4 * h_d, stride=2)
        self.bott = ResidualBlock(4 * h_d, 8 * h_d, stride=2)

        self.up3 = nn.ConvTranspose2d(8 * h_d, 4 * h_d, 2, 2)
        self.dec3 = ResidualBlock(8 * h_d, 4 * h_d, stride=1)
        self.up2 = nn.ConvTranspose2d(4 * h_d, 2 * h_d, 2, 2)
        self.dec2 = ResidualBlock(4 * h_d, 2 * h_d, stride=1)
        self.up1 = nn.ConvTranspose2d(2 * h_d, 1 * h_d, 2, 2)
        self.dec1 = ResidualBlock(2 * h_d, 1 * h_d, stride=1)
        self.head = nn.Conv2d(h_d, out_d, 1)

    @staticmethod
    def _match_size(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def _forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        s1 = self.enc1(x0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bott(s3)

        d3 = self._match_size(self.up3(b), s3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self._match_size(self.up2(d3), s2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self._match_size(self.up1(d2), s1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        return self.head(d1)


class StateOnlyPredictor(nn.Module):
    def __init__(self, predictor, context_length=2):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = predictor.is_rnn
        self.context_length = context_length

    def forward(self, x, a=None):
        prev_state = x[:, :, :-1]
        next_state = x[:, :, 1:]
        combined = torch.cat((prev_state, next_state), dim=1)
        return self.predictor(combined)


class Projector(nn.Module):
    def __init__(self, mlp_spec: str):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.extend([nn.Linear(f[i], f[i + 1]), nn.BatchNorm1d(f[i + 1]), nn.ReLU(True)])
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]

    def forward(self, x):
        return self.net(x)


class ImageDecoder(TemporalBatchMixin, nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, 3, 1, 1),
        )
        self.apply(init_module_weights)

    def _forward(self, x):
        return self.net(x)


class conv3d2(nn.Sequential):
    def __init__(self, in_d, h_d, out_d, tk, ts, sk, ss, pad):
        super().__init__(
            nn.Conv3d(in_d, h_d, (tk, sk, sk), (1, 1, 1), pad),
            nn.ReLU(),
            nn.Conv3d(h_d, out_d, (tk, sk, sk), (ts, ss, ss), pad),
        )
        self.apply(init_module_weights)
        if pad == "valid":
            self.t_shift = 2 * tk - 1
        elif pad == "same":
            self.t_shift = 2 * (tk - 1)


class SomaDetHead(nn.Module):
    def __init__(self, in_d, h_d, map_size=8):
        super().__init__()
        self.map_size = map_size
        self.head = nn.Sequential(conv3d2(in_d, h_d, 1, 1, 1, 3, 1, "same"))
        self.apply(init_module_weights)

    def forward(self, x):
        x = [
            F.adaptive_avg_pool2d(x[:, :, t], (self.map_size, self.map_size))
            for t in range(x.shape[2])
        ]
        x = torch.stack(x, 2)
        x = self.head(x).squeeze(1)
        return torch.sigmoid(x)

    @torch.no_grad()
    def score(self, preds, targets):
        scores = []
        for T in range(len(preds) - 1):
            x = preds[T]
            x = [
                F.adaptive_avg_pool2d(x[:, :, t], (self.map_size, self.map_size))
                for t in range(x.shape[2])
            ]
            x = torch.stack(x, 2)
            x = self.head(x).squeeze(1)
            y = targets[:, T:]
            x = x[:, T:]
            try:
                ap = average_precision_score(
                    y.flatten().detach().long().cpu().numpy(),
                    x.flatten().detach().cpu().numpy(),
                    average="weighted",
                )
            except ValueError:
                ap = 0.0
            scores.append(ap)
        return scores


class HingeStdLoss(nn.Module):
    def __init__(self, std_margin: float = 1.0):
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        return torch.mean(F.relu(self.std_margin - std))


class CovarianceLoss(nn.Module):
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)
        return self.off_diagonal(cov).pow(2).mean()


class VCLoss(nn.Module):
    def __init__(self, std_coeff, cov_coeff, proj=None):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.proj = nn.Identity() if proj is None else proj
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, x, actions=None):
        x = x.transpose(0, 1).flatten(1).transpose(0, 1)
        fx = self.proj(x)
        std_loss = self.std_loss_fn(fx)
        cov_loss = self.cov_loss_fn(fx)
        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        total_unweighted = std_loss + cov_loss
        loss_dict = {"std_loss": std_loss.item(), "cov_loss": cov_loss.item()}
        return loss, total_unweighted, loss_dict


class SquareLossSeq(nn.Module):
    def __init__(self, proj=None):
        super().__init__()
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, state, predi):
        state = self.proj(state.transpose(0, 1).flatten(1).transpose(0, 1))
        predi = self.proj(predi.transpose(0, 1).flatten(1).transpose(0, 1))
        return F.mse_loss(state, predi)


class JEPA(nn.Module):
    def __init__(self, encoder, aencoder, predictor, regularizer, predcost):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = aencoder
        self.predictor = predictor
        self.regularizer = regularizer
        self.predcost = predcost
        self.single_unroll = getattr(self.predictor, "is_rnn", False)

    def unroll(
        self,
        observations,
        actions=None,
        nsteps=1,
        unroll_mode="parallel",
        ctxt_window_time=1,
        compute_loss=True,
        return_all_steps=False,
    ):
        state = self.encoder(observations)
        context_length = getattr(self.predictor, "context_length", 0)

        if compute_loss:
            rloss, rloss_unweight, rloss_dict = self.regularizer(state, actions)
            ploss = 0.0
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        actions_encoded = self.action_encoder(actions) if actions is not None else None
        all_steps = [] if return_all_steps else None

        if unroll_mode == "parallel":
            predicted_states = state
            for _ in range(nsteps):
                predicted_states = self.predictor(predicted_states, actions_encoded)[:, :, :-1]
                if return_all_steps:
                    all_steps.append(predicted_states)
                predicted_states = torch.cat(
                    (state[:, :, :context_length], predicted_states), dim=2
                )
                if compute_loss:
                    ploss += self.predcost(state, predicted_states) / nsteps

        elif unroll_mode == "autoregressive":
            if actions is not None and nsteps > actions.size(2):
                raise ValueError(f"nsteps ({nsteps}) > action length ({actions.size(2)})")
            effective_ctxt = 1 if self.single_unroll else ctxt_window_time
            predicted_states = state[:, :, :effective_ctxt]
            for i in range(nsteps):
                context_states = predicted_states[:, :, -effective_ctxt:]
                context_actions = (
                    actions_encoded[:, :, max(0, i + 1 - effective_ctxt) : i + 1]
                    if actions_encoded is not None
                    else None
                )
                pred_step = self.predictor(context_states, context_actions)[:, :, -1:]
                predicted_states = torch.cat([predicted_states, pred_step], dim=2)
                if return_all_steps:
                    all_steps.append(predicted_states.clone())
                if compute_loss:
                    ploss += self.predcost(pred_step, state[:, :, i + 1 : i + 2]) / nsteps
        else:
            raise ValueError(f"Unknown unroll_mode: {unroll_mode}")

        if compute_loss:
            loss = rloss + ploss
            losses = (loss, rloss, rloss_unweight, rloss_dict, ploss)
        else:
            losses = None

        return (all_steps if return_all_steps else predicted_states), losses


class JEPAProbe(nn.Module):
    def __init__(self, jepa, head, hcost):
        super().__init__()
        self.jepa = jepa
        self.head = head
        self.hcost = hcost

    def forward(self, observations, targets):
        with torch.no_grad():
            state = self.jepa.encoder(observations)
        output = self.head(state.detach())
        return self.hcost(output, targets)
