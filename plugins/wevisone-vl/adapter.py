"""WeVisOne-VL adapter (2B and 4B variants share it). Wraps Tencent's `wevisone_vl` engine."""
from __future__ import annotations

import re

import numpy as np
import torch  # noqa: F401  (imported in the warm shell so resume skips it)
import transformers  # noqa: F401
from PIL import Image

import sdk

# function -> upstream task name
TASKS = {"detect": "detection", "refer": "referring", "point": "point", "count": "count_box",
         "segment": "segmentation", "matting": "matting", "ocr": "ocr"}

_TAG = re.compile(r"<ref>(.*?)</ref>\s*<(box|point|polygon)>(.*?)(?:</(?:box|point|polygon)>|(?=<ref>)|$)", re.S)
_BARE = re.compile(r"<(box|point)>(.*?)(?:</\1>|(?=<(?:box|point)>)|$)", re.S)
_BIN = re.compile(r"<(\d{1,4})>")


def parse_shapes(text: str, width: int, height: int) -> list[tuple[str, str, list[float]]]:
    """'<ref>cup</ref><box><12><40><380><990>,…</box>' -> [(label, 'box', [x0,y0,x1,y1] px), …]."""
    text = (text or "").split("<|im_end|>")[0]
    found = [(lab.strip(), kind, body) for lab, kind, body in _TAG.findall(text)]
    if not found:
        found = [("", kind, body) for kind, body in _BARE.findall(text)]
    out = []
    for label, kind, body in found:
        for seg in body.strip().rstrip(",").split(","):
            v = [int(x) for x in _BIN.findall(seg)]
            step = 2 if kind == "point" else 4
            for i in range(0, len(v) - step + 1, step):
                c = v[i:i + step]
                if kind == "point":
                    out.append((label, kind, [c[0] / 1000 * width, c[1] / 1000 * height]))
                elif c[2] > c[0] and c[3] > c[1]:
                    out.append((label, "box", [c[0] / 1000 * width, c[1] / 1000 * height,
                                               c[2] / 1000 * width, c[3] / 1000 * height]))
    return out


def _r(xs):
    return [round(float(x), 1) for x in xs]


class Adapter(sdk.BaseAdapter):
    def load(self, ctx: sdk.Context) -> None:
        from wevisone_vl import WeVisOneVL  # Tencent package (installed into this plugin's venv)
        # decoder_batch_size 1: the default of 16 decodes every object in an image together at full resolution.
        # 9 objects peaked at 24 GB and spilled past the card; one at a time took 7.7 GB and was 5x faster.
        self.engine = WeVisOneVL(ctx.weights_path, device=ctx.device, dtype="bfloat16", decoder_batch_size=1)
        self._stage_ctx = self._run_ctx = None
        self._batch = 1
        model = self.engine.model
        (getattr(model, "visual", None) or model.model.visual).register_forward_hook(self._vision_done)
        model.register_forward_pre_hook(self._step)

    def _vision_done(self, *_):
        """The image is encoded: preprocessing is over and token generation starts."""
        ctx, self._stage_ctx = self._stage_ctx, None
        if ctx:
            self._run_ctx = ctx
            ctx.report_stage("generating")

    def _step(self, *_):
        """Every generation step passes here: count its tokens and honour the time limit."""
        if self._run_ctx:
            self._run_ctx.tick(self._batch)

    def run(self, function: str, inputs: dict, ctx: sdk.Context) -> dict:
        return self._one(function, inputs, ctx, idx=0)

    def run_batch(self, function: str, inputs_list: list[dict], ctx: sdk.Context) -> list[dict]:
        return self.run_mixed([function] * len(inputs_list), inputs_list, ctx)

    def run_mixed(self, functions: list[str], inputs_list: list[dict], ctx: sdk.Context) -> list[dict]:
        """Batched generation (one padded generate call) with a prompt per request, so tasks can mix;
        masks are decoded per image afterwards, only for the responses that carry them."""
        from wevisone_vl import build_prompt
        self._stage_ctx, self._run_ctx, self._batch = ctx, None, len(inputs_list)
        missing = [f for f in functions if f not in TASKS]
        if missing:
            raise sdk.PluginError("unknown_function", missing[0])
        images = []
        for inp in inputs_list:
            with Image.open(inp["image"]) as im:
                images.append(im.convert("RGB"))
        prompts = [build_prompt(TASKS[fn], inp.get("query", ""), image_size=img.size)
                   for fn, inp, img in zip(functions, inputs_list, images)]
        results = self.engine.predict_batch(
            images, prompt=prompts, decode_matting=any(f in ("segment", "matting") for f in functions),
            max_new_tokens=max(int(i.get("max_new_tokens", 2048)) for i in inputs_list), temperature=0.0)
        return [self._format(fn, res, img, inp, ctx, idx)
                for idx, (fn, res, img, inp) in enumerate(zip(functions, results, images, inputs_list))]

    def _one(self, function, inputs, ctx, idx):
        task = TASKS.get(function)
        if task is None:
            raise sdk.PluginError("unknown_function", function)
        with Image.open(inputs["image"]) as im:
            image = im.convert("RGB")
        self._stage_ctx, self._run_ctx, self._batch = ctx, None, 1
        res = self.engine.predict(image, task=task, query=inputs.get("query", ""),
                                  decode_matting=function in ("segment", "matting"),
                                  max_new_tokens=int(inputs.get("max_new_tokens", 2048)), temperature=0.0)
        return self._format(function, res, image, inputs, ctx, idx)

    def _format(self, function, res, image, inputs, ctx, idx):
        w, h = image.size
        out: dict = {"response": res.response}
        shapes = parse_shapes(res.response, w, h)
        if function in ("detect", "refer"):
            out["instances"] = [{"box": _r(s), "label": lab} for lab, k, s in shapes if k == "box"]
        elif function == "point":
            out["points"] = [{"point": _r(s), "label": lab} for lab, k, s in shapes if k == "point"]
        elif function == "count":
            inst = [{"box": _r(s), "label": lab} for lab, k, s in shapes if k == "box"]
            nums = re.findall(r"\b(\d+)\b", re.sub(r"<[^>]*>", " ", res.response))
            out["instances"] = inst
            out["count"] = len(inst) if inst or not nums else int(nums[-1])
        elif function == "ocr":
            out["words"] = [{"text": lab, "box": _r(s)} for lab, k, s in shapes if k == "box"]
        else:  # segment / matting: boxes come normalised from the decoder, masks are float [H, W]
            thr = float(inputs.get("threshold", 0.5))
            inst = []
            for i, (box, mask) in enumerate(zip(res.boxes, res.masks)):
                m = np.asarray(mask, dtype=np.float32)
                if function == "segment":
                    png, key = Image.fromarray(((m >= thr) * 255).astype(np.uint8), "L"), "mask"
                else:
                    png, key = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), "L"), "alpha"
                path = ctx.out_path(f"{key}_{idx}_{i}.png")
                png.save(path)
                inst.append({"box": _r([box[0] * w, box[1] * h, box[2] * w, box[3] * h]), key: path})
            out["instances"] = inst
        return out
