"""Video writer implementations shared by reconstruction renderers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def _mp4_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".mp4":
        path = path.with_suffix(".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class OpenCVVideoWriter:
    def __init__(self, path: Path, fps: int, width: int, height: int):
        import cv2

        self.path = _mp4_path(path)
        self.cv2 = cv2
        self.writer = None
        for codec in ("mp4v", "avc1", "H264"):
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height))
            if writer.isOpened():
                self.writer = writer
                break
            writer.release()
        if self.writer is None:
            raise RuntimeError(f"OpenCV could not open VideoWriter for {self.path}")

    def append_data(self, image: np.ndarray) -> None:
        image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
        self.writer.write(self.cv2.cvtColor(np.ascontiguousarray(image), self.cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class FFmpegPipeWriter:
    def __init__(self, path: Path, fps: int, width: int, height: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg was not found on PATH.")

        self.path = _mp4_path(path)
        self.width = width
        self.height = height
        self.proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def append_data(self, image: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed.")
        image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
        if image.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected frame shape {(self.height, self.width, 3)}, got {image.shape}")
        self.proc.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        stderr = self.proc.stderr.read() if self.proc.stderr is not None else b""
        if self.proc.wait() != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed for {self.path}: {message}")


def announce(name: str, path) -> None:
    """Each step down this chain changes the codec, the quality, and at the end even the container.
    Say which encoder actually ran, instead of leaving the user to wonder why the file looks wrong."""
    print(f"[video] encoder={name}  ->  {path}", flush=True)


def make_default_writer(path: Path, fps: int, width: int, height: int):
    path = _mp4_path(path)
    try:
        w = OpenCVVideoWriter(path, fps, width, height)
        announce("OpenCV (mp4v)", path)
        return w, path
    except Exception:
        try:
            import imageio.v2 as imageio

            w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
            announce("imageio (libx264)", path)
            return w, path
        except Exception:
            try:
                import imageio.v2 as imageio

                gif_path = path.with_suffix(".gif")
                w = imageio.get_writer(gif_path, mode="I", fps=fps)
                announce("imageio GIF -- no MP4 encoder was available", gif_path)
                return w, gif_path
            except Exception as imageio_exc:
                raise RuntimeError(
                    "No video writer available. Install ffmpeg (best), or opencv-python, or imageio."
                ) from imageio_exc


def make_ffmpeg_first_writer(path: Path, fps: int, width: int, height: int):
    try:
        writer = FFmpegPipeWriter(path, fps, width, height)
        announce("ffmpeg (H.264)", writer.path)
        return writer, writer.path
    except Exception:
        return make_default_writer(path, fps, width, height)
