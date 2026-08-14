import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quasirandom import SobolEngine


def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)


class CFLossFunc(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference.
    Args:
        alpha_for_loss: the weight for amplitude in CF loss, from 0-1
        beta_for_loss: the weight for phase in CF loss, from 0-1
    """

    def __init__(self, alpha_for_loss=0.5, beta_for_loss=0.5):
        super(CFLossFunc, self).__init__()
        self.alpha = alpha_for_loss
        self.beta = beta_for_loss
        self._sobol_engines = {}
        self._importance_generators = {}

    def sample_frequencies(self, count, dimension, device, args):
        """Sample frequencies for audited MC/QMC/importance variants."""
        mode = getattr(args, "frequency_sampler", "mc").lower()
        seed = int(getattr(args, "frequency_seed", 0)) + int(getattr(args, "rank", 0))
        if mode == "mc":
            return torch.randn(count, dimension, device=device), None
        if mode == "qmc":
            key = (dimension, seed)
            engine = self._sobol_engines.get(key)
            if engine is None:
                engine = SobolEngine(dimension, scramble=True, seed=seed)
                self._sobol_engines[key] = engine
            uniform = engine.draw(count).clamp_(1e-6, 1.0 - 1e-6)
            frequencies = (2.0**0.5) * torch.erfinv(2.0 * uniform - 1.0)
            return frequencies.to(device), None
        if mode == "importance":
            mean_shift = float(getattr(args, "importance_mean_shift", 0.5))
            generator = self._importance_generators.get(seed)
            if generator is None:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                self._importance_generators[seed] = generator
            mean = torch.zeros(dimension)
            mean[0] = mean_shift
            # Draw from q=N(mean,I).  The previous implementation drew from
            # N(0,I) but applied the p/q correction for a shifted proposal,
            # so the declared importance-sampling variant was not the stated
            # estimator.
            frequencies = torch.randn(count, dimension, generator=generator) + mean
            log_weight = -(frequencies @ mean) + 0.5 * mean.square().sum()
            return frequencies.to(device), log_weight.exp().to(device)
        raise ValueError(f"Unsupported frequency_sampler: {mode}")

    def forward(self, feat_tg, feat, t=None, args=None):
        """
        Calculate CF loss between target and synthetic features.
        Args:
            feat_tg: target features from real data [B1 x D]
            feat: synthetic features [B2 x D]
            args: additional arguments containing num_freqs
        """
        # Generate random frequencies
        weights = None
        if t is None:
            t, weights = self.sample_frequencies(
                args.num_freqs, feat.size(1), feat.device, args
            )
        t_x_real = calculate_real(torch.matmul(t, feat.t()))
        t_x_imag = calculate_imag(torch.matmul(t, feat.t()))
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        t_target_real = calculate_real(torch.matmul(t, feat_tg.t()))
        t_target_imag = calculate_imag(torch.matmul(t, feat_tg.t()))
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        # Calculate amplitude difference and phase difference
        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (
            torch.mul(t_target_norm, t_x_norm)
            - torch.mul(t_x_real, t_target_real)
            - torch.mul(t_x_imag, t_target_imag)
        )

        loss_pha = loss_pha.clamp(min=1e-12)  # Ensure numerical stability

        # Combine losses
        per_frequency = torch.sqrt(self.alpha * loss_amp + self.beta * loss_pha)
        # For importance sampling this is the exact p/q weighted estimator.
        # No clipping or self-normalization is applied; those are separate
        # biased-stabilized ablations and must not be called unbiased.
        if weights is not None:
            loss = torch.mean(weights * per_frequency)
        else:
            loss = torch.mean(per_frequency)
        return loss


def match_loss(img_real, img_syn, model,sampling_net, args=None):
    """Matching losses (feature or gradient)"""
    with torch.no_grad():
        _, feat_tg = model(img_real, return_features=True)
    _, feat = model(img_syn, return_features=True)
    feat = F.normalize(feat, dim=1)
    feat_tg = F.normalize(feat_tg, dim=1)
    if sampling_net is not None:
        t = sampling_net(args.device)
    else:
        t = None
    loss = 300 * args.cf_loss_func(feat_tg, feat, t, args)
    return loss


def mutil_layer_match_loss(img_real, img_syn, model,sampling_net, args=None):

    # Ensure layer_index is a list
    assert isinstance(
        args.layer_index, list
    ), "args.layer_index must be a list of layer indices"

    # Initialize loss as a tensor on the correct device
    loss = torch.tensor(0.0).to(img_real.device)

    # Extract features for both real and synthetic images
    with torch.no_grad():
        feat_tg_list = model.get_feature_mutil(img_real)  # Real image features
    feat_list = model.get_feature_mutil(img_syn)  # Synthetic image features

    for layer_index in args.layer_index:
        assert (
            0 <= layer_index <= 6
        ), f"layer_index {layer_index} must be between 0 and 6"
        if args.dis_metrics == "MMD":
            # If the metric is MMD, calculate the MMD loss for the selected layer
            feat = feat_list[layer_index]
            feat_tg = feat_tg_list[layer_index]
            loss += torch.sum((feat.mean(0) - feat_tg.mean(0)) ** 2)
        else:
            # Otherwise, calculate the feature matching loss for the selected layer
            feat = feat_list[layer_index]
            feat_tg = feat_tg_list[layer_index]
            feat = F.normalize(feat, dim=1)  # Normalize the feature
            feat_tg = F.normalize(feat_tg, dim=1)  # Normalize the target feature
            t = None  # Adjust this based on your CFLossFunc usage
            loss += 300 * args.cf_loss_func(feat_tg, feat, t, args)

    return loss


def cailb_loss(img_syn, label_syn, trained_model):
    logits = trained_model(img_syn, return_features=False)
    loss = F.cross_entropy(logits, label_syn)
    return loss


def pixel_mean_match_loss(img_real, img_syn, model, sampling_net, args=None):
    """A transparent pixel-space control objective.

    It matches the per-class mean image in the same normalized tensor space
    used by the condenser.  This is intentionally a control objective, not a
    claim that it is the original NCFM loss.
    """
    del model, sampling_net
    real_mean = img_real.mean(dim=0)
    syn_mean = img_syn.mean(dim=0)
    return 300.0 * F.mse_loss(syn_mean, real_mean)
