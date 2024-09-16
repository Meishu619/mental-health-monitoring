import torch


class CCCLoss(torch.nn.Module):
    def forward(self, output, target):
        out_mean = torch.mean(output)
        target_mean = torch.mean(target)

        covariance = torch.mean((output - out_mean) * (target - target_mean))
        target_var = torch.mean((target - target_mean)**2)
        out_var = torch.mean((output - out_mean)**2)

        ccc = 2.0 * covariance / \
            (target_var + out_var + (target_mean - out_mean)**2 + 1e-10)
        loss_ccc = 1.0 - ccc

        return loss_ccc