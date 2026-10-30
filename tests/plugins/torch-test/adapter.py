import time

import torch

import sdk


class Adapter(sdk.BaseAdapter):
    def load(self, ctx):
        self.device = ctx.device
        self.resident = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=self.device)  # 64 MB "weights"

    def run(self, function, inputs, ctx):
        n = int(float(inputs["gb"]) * 1024 ** 3)
        buf = torch.empty(n, dtype=torch.uint8, device=self.device)
        buf.fill_(1)
        torch.cuda.synchronize()
        time.sleep(float(inputs.get("seconds", 0)))
        got = buf.numel() / 1024 ** 3
        del buf
        return {"allocated_gb": round(got, 3)}
