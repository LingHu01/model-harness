"""Uniform error object (spec 5.3)."""
from __future__ import annotations

CODES = {
    "harness_not_running", "unknown_model", "unknown_function", "invalid_input", "queue_stalled",
    "runaway", "cancelled", "oom", "external_memory_holder", "worker_crashed", "model_too_large",
    "env_build_failed", "shutting_down", "busy", "load_failed", "worker_error",
}


class HarnessError(Exception):
    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra

    def to_dict(self, request_id: str | None = None) -> dict:
        err = {"code": self.code, "message": self.message}
        if request_id:
            err["request_id"] = request_id
        err.update({k: v for k, v in self.extra.items() if v is not None})
        return {"error": err}
