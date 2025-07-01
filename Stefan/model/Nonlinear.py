import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class DynamicActivation(nn.Module):
    def __init__(self, activation_type=1):
        super(DynamicActivation, self).__init__()
        self.activation_type = activation_type
        self.activations = {
            1: nn.ELU(),
            2: nn.Hardshrink(),
            3: nn.Hardsigmoid(),
            4: nn.Hardtanh(),
            5: nn.Hardswish(),
            6: nn.LeakyReLU(),
            7: nn.LogSigmoid(),
            8: nn.PReLU(),
            9: nn.ReLU(),
            10: nn.ReLU6(),
            11: nn.RReLU(),
            12: nn.SELU(),
            13: nn.CELU(),
            14: nn.GELU(),
            15: nn.Sigmoid(),
            16: nn.SiLU(),
            17: nn.Mish(),
            18: nn.Softplus(),
            19: nn.Softshrink(),
            20: nn.Softsign(),
            21: nn.Tanh(),
            22: nn.Tanhshrink(),
            23: nn.Threshold(value=0,threshold=0.5),
            24: nn.Identity(),
            25: nn.Softmin(),
            26: nn.Softmax(),
            27: nn.LogSoftmax(),
            28: nn.Identity(),
            29: BentIdentity(),
            30: Absolute(),
            31: Bipolar(),
            32: BipolarSigmoid(),
            33: Sinusoid(),
            34: Cosine(),
            35: Arcsinh(),
            36: Arccosh(),
            37: Arctanh(),
            38: LeCunTanh(),
            39: TanhExp(),
            40: Gaussian(),
            41: GCU(),
            42: ASU(),
            43: SOU(),
            44: NCU(),
            45: DSU(),
            46: SSU(),
            47: SReLU(),
            48: BReLU(),
            49: PELU(),
            50: Phish(),
            51: RBF(),
            52: SO_RBF(),
            53: ISRU(),
            54: CLL(),
            55: ISRLU(),
            56: SONL(),
            57: PLU(),
            58: APL(),
            59: InverseCubic(),
            60: SoftExponential(),
            61: ParametricLinear(),
            62: PiecewiseLinearUnit(),
            63: SquaredReLU(),
            64: ModReLU(),
            65: CosReLU(),
            66: SinReLU(),
            67: Probit(),
            68: Smish(),
            69: Multiquadratic(),
            70: InvMultiquadratic(),
            71: Psmish(),
            72: Eswish(),
            73: ELisH(),
            74: HardELisH(),
            75: Serf(),
            76: FReLU(),
            77: OReLU(),
            78: MOReLU(),
            79: CReLU(),
            80: HardELiSH(),
            81: Siren(),
            82: ShiftedSoftPlus(),
            83: Logit(),
            84: m_QReLU(),
            85: CoLU(),
            86: PAU(),
            87: DELU(),
            88: PDELU(),
            89: COSLU(),
            90: ELiSH(),
            91: Hermite(),
            92: AHAF(),
            93: SERLU(),
            94: ShiLU(),
            95: ReLUN(),
            96: SSFG(),
            97: QReLU(),
            98: ScaledSoftsign(),
            99: NormLinComb(),
            100: EvoNormB0(),
            101: EvoNormS0(),
            102: SmoothStep(),
            103: LinComb(),
            104: Nipuna(),
            105: StarReLU(),
            106:HermiteActivation(),
            107:act9_18(),
            108:comp(),
        }
        for key, activation in self.activations.items():
            if any(isinstance(param, nn.Parameter) for param in activation.parameters()):
                self.activations[key] = activation.cuda()

    def forward(self, x):
        if self.activation_type in self.activations:
            return self.activations[self.activation_type](x)
        else:
            raise ValueError("Invalid activation type")


class BentIdentity(nn.Module):
    def forward(self, x):
        return (torch.sqrt(x**2 + 1) - 1) / 2 + x

class Absolute(nn.Module):
    def forward(self, x):
        return torch.abs(x)

class Bipolar(nn.Module):
    def forward(self, x):
        x = torch.where(x >= 0, 1, -1)
        return x.float()

class comp(nn.Module):
    def __init__(self):
        super(comp, self).__init__()
    def forward(self, x):
        a1 = x**2+x
        a2 = x * torch.tanh(F.gelu(x))
        a3 = torch.cos(x)
        result = (a1+a2+a3)
        return result
class BipolarSigmoid(nn.Module):
    def forward(self, x):
        return (torch.sigmoid(x) - 0.5) * 2

class Sinusoid(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Cosine(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class Arcsinh(nn.Module):
    def forward(self, x):
        return torch.asinh(x)

class Arccosh(nn.Module):
    def forward(self, x):
        return torch.acosh(x)

class Arctanh(nn.Module):
    def forward(self, x):
        return torch.atanh(x)

class LeCunTanh(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh(2/3 * x)

class TanhExp(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

class Gaussian(nn.Module):
    def forward(self, x):
        return torch.exp(-x**2)

class GCU(nn.Module):
    def __init__(self):
        super(GCU, self).__init__()

    def forward(self, x):
        return x * torch.cos(x)

class ASU(nn.Module):
    def __init__(self):
        super(ASU, self).__init__()

    def forward(self, x):
        return x * torch.sin(x)

class SOU(nn.Module):
    def __init__(self):
        super(SOU, self).__init__()

    def forward(self, x):
        return x**2+x

class NCU(nn.Module):
    def __init__(self):
        super(NCU, self).__init__()

    def forward(self, x):
        return x-x**3

class DSU(nn.Module):
    def __init__(self):
        super(DSU, self).__init__()
        self.pi = torch.tensor(math.pi, dtype=torch.float32)

    def forward(self, x):
        # 计算 π * sinc(x - π)
        return self.pi * torch.sinc(x - self.pi)


class SSU(nn.Module):
    def __init__(self):
        super(SSU, self).__init__()
        self.pi = torch.tensor(math.pi)  # 定义 π 常量

    def forward(self, x):
        # 计算 sinc(x - π) 和 sinc(x + π)
        sinc_minus_pi = torch.sinc(x - self.pi)
        sinc_plus_pi = torch.sinc(x + self.pi)
        result = 0.5 * self.pi * (sinc_minus_pi - sinc_plus_pi)

        return result

class SReLU(nn.Module):
    def __init__(self, T_l=-0.4, T_r=0.4, a_l=-1.0, a_r=2.0, b_l=0.0, b_r=0.01):
        super(SReLU, self).__init__()
        self.T_l = T_l
        self.T_r = T_r
        self.a_l = a_l
        self.a_r = a_r
        self.b_l = b_l
        self.b_r = b_r

    def forward(self, x):
        return torch.where(x <= self.a_l, self.T_l * (x - self.a_l) + self.b_l,
                           torch.where(x < self.a_r, x, self.T_r * (x - self.a_r) + self.b_r))

class BReLU(nn.Module):
    def __init__(self):
        super(BReLU, self).__init__()

    def forward(self, x):
        return torch.abs(x)

class PELU(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(PELU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return torch.where(x > 0, (self.alpha/self.beta)*x, self.alpha * (torch.exp(x / self.beta) - 1))

class Phish(nn.Module):
    def __init__(self):
        super(Phish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.gelu(x))

class RBF(nn.Module):
    def __init__(self, gamma=1.0, input_shape=(50, 20, 64, 64)):
        super(RBF, self).__init__()
        self.centers = nn.Parameter(torch.randn(*input_shape, requires_grad=True))
        self.gamma = gamma

    def forward(self, x):
        diff = x - self.centers
        norm = x+diff
        return torch.exp(-self.gamma * norm**2)

class SO_RBF(nn.Module):
    def __init__(self):
        super(SO_RBF, self).__init__()

    def forward(self, x):
        abs_x = torch.abs(x)
        output = torch.zeros_like(x)
        mask1 = abs_x <= 1
        output[mask1] = 1 - 0.5 * x[mask1] ** 2
        mask2 = (abs_x > 1) & (abs_x < 2)
        output[mask2] = 0
        mask3 = abs_x >= 2
        output[mask3] = 0.5 * (2 - abs_x[mask3]) ** 2

        return output

class ISRU(nn.Module):
    def __init__(self, alpha=1.0):
        super(ISRU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return x / torch.sqrt(1 + self.alpha * x**2)

class CLL(nn.Module):
    def __init__(self):
        super(CLL, self).__init__()

    def forward(self, x):
        return 1 - torch.exp(-1 * torch.exp(x))

class ISRLU(nn.Module):
    def __init__(self, alpha=1.0):
        super(ISRLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x >= 0, x, x / torch.sqrt(1 + self.alpha * x**2))

class SONL(nn.Module):
    def __init__(self):
        super(SONL, self).__init__()

    def forward(self, x):
        device = x.device
        result = torch.where(x < -2,
                             torch.tensor(-1.0, device=device),  # x < -2 时返回 -1
                             torch.where(x < 0,
                                         x + 0.25 * x ** 2,  # -2 <= x < 0 时返回 x + 0.25 * x**2
                                         torch.where(x < 2,
                                                     x - 0.25 * x ** 2,  # 0 <= x < 2 时返回 x - 0.25 * x**2
                                                     torch.tensor(1.0, device=device)  # x >= 2 时返回 1
                                                     )
                                         )
                             )
        return result

class PLU(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(PLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return torch.where(x > 0, self.alpha * x, self.beta * (torch.exp(x) - 1))

class APL(nn.Module):
    def __init__(self, alpha=-0.2,b=0.4):
        super(APL, self).__init__()
        self.alpha = alpha
        self.b = b

    def forward(self, x):
        return F.relu(x) + self.alpha * F.relu(-x+self.b)


class InverseCubic(nn.Module):
    def forward(self, x):
        term = 0.5 * (torch.sqrt(9 * x ** 2 + 4) + 3 * x)
        return term ** (1 / 3) - term ** (-1 / 3)

class SoftExponential(nn.Module):
    def __init__(self, alpha=1.0):
        super(SoftExponential, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        if self.alpha == 0:
            return x
        elif self.alpha > 0:
            return (torch.exp(self.alpha * x) - 1) / self.alpha + self.alpha
        else:
            return -torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

class ParametricLinear(nn.Module):
    def __init__(self, slope=1.0, bias=0.0):
        super(ParametricLinear, self).__init__()
        self.slope = nn.Parameter(torch.tensor(slope))
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x):
        return self.slope * x + self.bias

class PiecewiseLinearUnit(nn.Module):
    def __init__(self, threshold=0.0):
        super(PiecewiseLinearUnit, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, x):
        return torch.where(x > self.threshold, x, self.threshold)

class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class ModReLU(nn.Module):
    def __init__(self, bias=0.0):
        super(ModReLU, self).__init__()
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.bias) * torch.sign(x)

class CosReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) + torch.cos(x)

class SinReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) + torch.sin(x)

class Probit(nn.Module):
    def forward(self, x):
        return torch.erf(x / torch.sqrt(torch.tensor(2.0)))

class Smish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))

class Multiquadratic(nn.Module):
    def forward(self, x):
        return torch.sqrt((x - 0.5)**2 + 9)

class InvMultiquadratic(nn.Module):
    def forward(self, x):
        return 1 / torch.sqrt((x - 0.5)**2 + 9)

class Psmish(nn.Module):
    def forward(self, x):
        return 0.5 * x * torch.tanh(torch.log(1 + torch.sigmoid(3 * x)))

class Eswish(nn.Module):
    def forward(self, x):
        return (3 * x) / (1 - torch.exp(-3 * x))

class ELisH(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.log(torch.abs(x) + 1))

class HardELisH(nn.Module):
    def forward(self, x):
        return x * torch.clamp(torch.log(torch.abs(x) + 1), min=-1, max=1)


class ELiSH(nn.Module):
    def __init__(self):
        super(ELiSH, self).__init__()

    def forward(self, x):
        sigmoid_x = torch.sigmoid(x)
        positive_part = x * sigmoid_x
        negative_part = (torch.exp(x) - 1) * sigmoid_x
        return torch.where(x >= 0, positive_part, negative_part)


class HardELiSH(nn.Module):
    def __init__(self):
        super(HardELiSH, self).__init__()

    def forward(self, x):
        # 计算max(0, min(1, 0.5*(x + 1)))
        clamp_value = torch.clamp(0.5 * (x + 1), 0, 1)

        # 根据x的符号选择不同的表达式
        positive_part = x * clamp_value
        negative_part = (torch.exp(x) - 1) * clamp_value

        # 使用torch.where来选择不同的表达式
        return torch.where(x >= 0, positive_part, negative_part)

class Serf(nn.Module):
    def forward(self, x):
        log_exp_x = torch.log(1 + torch.exp(x))
        erf_log_exp_x = torch.erf(log_exp_x)
        # 返回 x * erf(ln(1 + exp(x)))
        return x * erf_log_exp_x

class FReLU(nn.Module):
    def __init__(self, in_channels=20):
        super(FReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels,
                              bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        t = self.bn(self.conv(x))
        return torch.max(x, t)

class OReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) * torch.cos(x)

class MOReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) * torch.sin(x)
class CoLU(nn.Module):
    def __init__(self):
        super(CoLU, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        sum_x_exp_x = x + exp_x
        neg_sum_x_exp_x = -sum_x_exp_x
        exp_neg_sum_x_exp_x = torch.exp(neg_sum_x_exp_x)
        x_times_exp = x * exp_neg_sum_x_exp_x
        denominator = 1 - x_times_exp
        result = x / denominator

        return result

class QReLU(nn.Module):
    def __init__(self):
        super(QReLU, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, x, 0.01 * x * (x - 2))
class m_QReLU(nn.Module):
    def __init__(self):
        super(m_QReLU, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, x, 0.01 * x - x)
class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        pos = nn.functional.relu(x)
        neg = nn.functional.elu(-x)

        return pos+neg
class KAF_tainble_yongbuliao (nn.Module):
    def forward(self, x):
        return F.relu(x) * torch.exp(-x**2)

class Siren(nn.Module):
    def __init__(self, in_features=64, out_features=64, w0=1.0):
        super(Siren, self).__init__()
        self.in_features = in_features
        self.w0 = w0
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, input):
        return torch.sin(self.w0 * torch.matmul(input, self.weight.t()) + self.bias)

class ShiftedSoftPlus(nn.Module):
    def forward(self, x):
        return torch.log(0.5 + 0.5 * torch.exp(x))

class Logit(nn.Module):
    def forward(self, x):
        return (x / (1 - x))

class ARiA_sb(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MARCSinh_sb(nn.Module):
    def forward(self, x):
        return torch.sinh(x)

class PAU(nn.Module):
    def __init__(self,m=8):
        super(PAU, self).__init__()
        self.P_coeffs = nn.Parameter(torch.randn(m + 1))  # P(x)的系数，m+1个参数
        self.Q_coeffs = nn.Parameter(torch.randn(m + 1))  # Q(x)的系数，m+1个参数

    def forward(self, x):
        # 计算P(x)
        P_x = torch.zeros_like(x)
        for i in range(self.P_coeffs.size(0)):
            P_x += self.P_coeffs[i] * (x ** i)
        Q_x = torch.ones_like(x)
        for i in range(self.Q_coeffs.size(0)):
            Q_x += self.Q_coeffs[i] * (x ** i)
        Q_x+=1
        output = P_x / Q_x

        return output
class act9_18(nn.Module):
    def __init__(self):
        super(act9_18, self).__init__()

    def forward(self, x):
        return 0.5*x**2+x
class DELU(nn.Module):
    def __init__(self, n_init=0.5):
        super(DELU, self).__init__()
        self.n = nn.Parameter(torch.tensor(n_init, dtype=torch.float32))

    def forward(self, x):
        silu_output = F.silu(x)
        positive_output = (self.n + 0.5) * x + torch.abs(torch.exp(-x) - 1)
        output = torch.where(x < 0, silu_output, positive_output)
        return output

class PDELU(nn.Module):
    def __init__(self,t=0.8):
        super(PDELU, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.t = t

    def forward(self, x):
        positive_part = x
        negative_part = self.a * ((1 + (1 - self.t) * x) ** (1 / (1 - self.t)) - 1)
        return torch.where(x > 0, positive_part, negative_part)

class COSLU(nn.Module):
    def __init__(self, a=1.0, b=1.0):
        super(COSLU, self).__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, x):
        cos_term = torch.cos(self.b * x)
        sigmoid_term = torch.sigmoid(x)
        output = (x + self.a * cos_term) * sigmoid_term
        return output

class NFNsbdongxi(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class HermiteActivation(nn.Module):
    def __init__(self, num_pol=5, copy_fun='relu'):
        super(HermiteActivation, self).__init__()
        self.num_pol = num_pol
        self.copy_fun = copy_fun
        self.h = self.initialize_hermite_polynomials()
        self.k = self.get_initializations()

    def initialize_hermite_polynomials(self):
        h = []

        def h0(x): return torch.ones_like(x)
        h.append(h0)
        def h1(x): return x
        h.append(h1)
        def h2(x): return (torch.pow(x, 2) - 1) / torch.sqrt(torch.tensor(math.factorial(2), dtype=torch.float32))
        h.append(h2)
        def h3(x): return (torch.pow(x, 3) - 3 * x) / torch.sqrt(torch.tensor(math.factorial(3), dtype=torch.float32))
        h.append(h3)
        def h4(x): return (torch.pow(x, 4) - 6 * torch.pow(x, 2) + 3) / torch.sqrt(
            torch.tensor(math.factorial(4), dtype=torch.float32))
        h.append(h4)
        def h5(x): return (torch.pow(x, 5) - 10 * torch.pow(x, 3) + 15 * x) / torch.sqrt(
            torch.tensor(math.factorial(5), dtype=torch.float32))
        h.append(h5)
        def h6(x): return (torch.pow(x, 6) - 15 * torch.pow(x, 4) + 45 * torch.pow(x, 2) - 15) / torch.sqrt(
            torch.tensor(math.factorial(6), dtype=torch.float32))
        h.append(h6)
        def h7(x): return (torch.pow(x, 7) - 21 * torch.pow(x, 5) + 105 * torch.pow(x, 3) - 105 * x) / torch.sqrt(
            torch.tensor(math.factorial(7), dtype=torch.float32))
        h.append(h7)
        def h8(x): return (torch.pow(x, 8) - 28 * torch.pow(x, 6) + 210 * torch.pow(x, 4) - 420 * torch.pow(x,
                                                                                                            2) + 105) / torch.sqrt(
            torch.tensor(math.factorial(8), dtype=torch.float32))
        h.append(h8)
        def h9(x): return (torch.pow(x, 9) - 36 * torch.pow(x, 7) + 378 * torch.pow(x, 5) - 1260 * torch.pow(x,
                                                                                                             3) + 945 * x) / torch.sqrt(
            torch.tensor(math.factorial(9), dtype=torch.float32))
        h.append(h9)
        def h10(x): return (torch.pow(x, 10) - 45 * torch.pow(x, 8) + 630 * torch.pow(x, 6) - 3150 * torch.pow(x,
                                                                                                               4) + 4725 * torch.pow(
            x, 2) - 945) / torch.sqrt(torch.tensor(math.factorial(10), dtype=torch.float32))
        h.append(h10)
        return h

    def get_initializations(self):
        k = []
        if self.copy_fun == 'relu':
            for n in range(self.num_pol):
                if n == 0:
                    k.append(1.0 / math.sqrt(2 * math.pi))
                elif n == 1:
                    k.append(1.0 / 2)
                elif n == 2:
                    k.append(1.0 / math.sqrt(4 * math.pi))
                elif n > 2 and n % 2 == 0:
                    c = 1.0 * math.factorial(n - 3) ** 2 / math.sqrt(2 * math.pi * math.factorial(n))
                    k.append(c)
                elif n >= 2 and n % 2 != 0:
                    k.append(0.0)
        return k

    def forward(self, x):
        evals = torch.zeros_like(x)
        for i in range(self.num_pol):
            evals += self.k[i] * self.h[i](x)
        return evals

class AHAF(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0):
        super(AHAF, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(beta, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(gamma, requires_grad=True))

    def forward(self, x):
        return self.alpha * torch.tanh(self.beta * x) + self.gamma

class SERLU(nn.Module):
    def __init__(self, alpha=1.6732632423543772848170429916717,
                 lambd=1.0507009873554804934193349852946):
        super(SERLU, self).__init__()
        self.alpha = alpha
        self.lambd = lambd

    def forward(self, x):
        return self.lambd * torch.where(x > 0, x, self.alpha * x * torch.exp(x))

class ShiLU(nn.Module):
    def __init__(self):
        super(ShiLU, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.b = nn.Parameter(torch.tensor([0.5], requires_grad=True))
    def forward(self, x):
        return self.a * F.relu(x) + self.b

class ReLUN(nn.Module):
    def __init__(self):
        super(ReLUN, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.5], requires_grad=True))
    def forward(self, x):
        return torch.clamp(x, min=0, max=self.a.item())

class use(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class useless(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ScaledSoftsign(nn.Module):
    def __init__(self):
        super(ScaledSoftsign, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.b = nn.Parameter(torch.tensor([0.5], requires_grad=True))
    def forward(self, x):
        return (self.a*x) / (self.b + torch.abs(x))

class NormLinComb(nn.Module):
    def __init__(self):
        super(NormLinComb, self).__init__()
        self.weight_relu = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.weight_tanh = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.weight_elu = nn.Parameter(torch.tensor([0.5], requires_grad=True))

    def forward(self, x):
        # 计算每个激活函数的结果
        relu_output = F.relu(x)
        tanh_output = torch.tanh(x)
        elu_output = F.elu(x)
        mixed_output = self.weight_relu * relu_output + self.weight_tanh * tanh_output + self.weight_elu * elu_output
        total_weight = self.weight_relu + self.weight_tanh + self.weight_elu
        normalized_output = mixed_output / total_weight

        return normalized_output

class SSFG(nn.Module):
    def __init__(self):
        super(SSFG, self).__init__()
        self.weight_relu = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.weight_tanh = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.weight_elu = nn.Parameter(torch.tensor([0.5], requires_grad=True))

    def forward(self, x):
        # 计算每个激活函数的结果
        relu_output = F.gelu(x)
        tanh_output = F.relu(x)
        elu_output = F.elu(x)
        mixed_output = self.weight_relu * relu_output + self.weight_tanh * tanh_output + self.weight_elu * elu_output
        total_weight = self.weight_relu + self.weight_tanh + self.weight_elu
        normalized_output = mixed_output / total_weight

        return normalized_output

class EvoNormB0(nn.Module):
    def __init__(self,num_features=20, eps=1e-5):
        super(EvoNormB0, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        mean = x.mean((0, 2, 3), keepdim=True)
        var = x.var((0, 2, 3), keepdim=True)
        std = (var + self.eps).sqrt()
        y = (x - mean) / std
        return self.gamma * y + self.beta
#这个不太行
class EvoNormS0(nn.Module):
    def __init__(self,num_features=20, eps=1e-5):
        super(EvoNormS0, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.v = nn.Parameter(torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        mean = x.mean((2, 3), keepdim=True)
        var = x.var((2, 3), keepdim=True)
        std = (var + self.eps).sqrt()
        y = (x - mean) / std
        return self.gamma * y * torch.sigmoid(self.v * y) + self.beta


class SmoothStep(nn.Module):
    def __init__(self):
        super(SmoothStep, self).__init__()
    def forward(self, x):
        # 创建一个与输入张量x相同形状的零张量
        result = torch.zeros_like(x)

        # 对x中的每个元素应用SmoothStep函数
        result[(x <= -0.5)] = 0
        result[(x > -0.5) & (x <= 0.5)] = -2 * (x[(x > -0.5) & (x <= 0.5)] ** 3) + 1.5 * x[
            (x > -0.5) & (x <= 0.5)] + 0.5
        result[(x > 0.5)] = 1

        return result

class LinComb(nn.Module):
    def __init__(self):
        super(LinComb, self).__init__()
        self.weight_relu = nn.Parameter(torch.tensor([0.3], requires_grad=True))
        self.weight_tanh = nn.Parameter(torch.tensor([0.3], requires_grad=True))
        self.weight_elu = nn.Parameter(torch.tensor([0.3], requires_grad=True))

    def forward(self, x):
        # 计算每个激活函数的结果
        relu_output = F.relu(x)
        tanh_output = torch.tanh(x)
        elu_output = F.elu(x)

        # 使用可学习的权重进行加权求和
        mixed_output = self.weight_relu * relu_output + self.weight_tanh * tanh_output + self.weight_elu * elu_output

        return mixed_output

class Hermite(nn.Module):
    def __init__(self):
        super(Hermite, self).__init__()
        self.weight_relu = nn.Parameter(torch.tensor([0.3], requires_grad=True))
        self.weight_tanh = nn.Parameter(torch.tensor([0.3], requires_grad=True))
        self.weight_elu = nn.Parameter(torch.tensor([0.3], requires_grad=True))

    def forward(self, x):
        # 计算每个激活函数的结果
        relu_output = F.gelu(x)
        tanh_output = torch.tanh(x)
        elu_output = F.elu(x)

        # 使用可学习的权重进行加权求和
        mixed_output = self.weight_relu * relu_output + self.weight_tanh * tanh_output + self.weight_elu * elu_output

        return mixed_output


class Nipuna(nn.Module):
    def forward(self, x):
        g_x = x/(1+torch.exp(-x))
        return torch.max(g_x, x)

class StarReLU(nn.Module):
    def __init__(self):
        super(StarReLU, self).__init__()
        self.s = nn.Parameter(torch.tensor(0.8944, requires_grad=True))
        self.b = nn.Parameter(torch.tensor(-0.4472, requires_grad=True))
    def forward(self, x):
        return self.s *(F.relu(x))**2 + self.b


