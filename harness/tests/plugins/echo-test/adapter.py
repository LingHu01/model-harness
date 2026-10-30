import os
import time

import sdk


class Adapter(sdk.BaseAdapter):
    def load(self, ctx):
        self.loaded_at = time.time()

    def run(self, function, inputs, ctx):
        time.sleep(float(inputs.get("sleep_s", 0)))
        ctx.report_stage("generating")
        for _ in range(int(inputs.get("ticks", 0))):
            ctx.tick()
            time.sleep(0.01)
        if inputs.get("crash"):
            os._exit(3)
        return {"text": inputs["text"], "pid": os.getpid()}
