import os
import sys

def _unused_policy(obs):
    raise RuntimeError(
        "TrainingViewerHook must not call viewer.run(), viewer.tick(), "
        "or viewer._step_physics() during training."
    )

class TrainingViewerHook:
    def __init__(self, env, backend="auto", interval=1, exit_action="continue"):
        self.env = env
        self.backend = backend
        self.interval = max(1, int(interval))
        self.exit_action = exit_action
        self.viewer = None
        self.inactive = False
        self.step_counter = 0

    def setup(self):
        if self.backend == "auto":
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            self.backend = "native" if has_display else "viser"

        print(f"[INFO] Launching training viewer backend: {self.backend}")

        if self.backend == "native":
            from mjlab.viewer import NativeMujocoViewer
            self.viewer = NativeMujocoViewer(self.env, _unused_policy)
        elif self.backend == "viser":
            from mjlab.viewer import ViserPlayViewer
            self.viewer = ViserPlayViewer(self.env, _unused_policy)
        else:
            raise ValueError(f"Unsupported viewer backend: {self.backend}")

        self.viewer.setup()

    def render_if_needed(self):
        if self.inactive or self.viewer is None:
            return

        self.step_counter += 1
        if self.step_counter % self.interval != 0:
            return

        if self.backend == "native" and not self.viewer.is_running():
            self.inactive = True
            if self.exit_action == "stop":
                print("[INFO] Viewer closed, stopping training.")
                sys.exit(0)
            return

        # 保持内部计数的更新，以防不走 viewer.run() 时数据不同步
        self.viewer._step_count = self.step_counter
        self.viewer._stats_frames += 1
        self.viewer.sync_env_to_viewer()

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass