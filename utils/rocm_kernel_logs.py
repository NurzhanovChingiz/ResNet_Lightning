"""Keep Composable Kernel GEMM grid-descriptor dumps off the console."""

from __future__ import annotations

import os
import sys
import threading

_UNSET_NAMES = ("CK_LOGGING",)
_QUIET_VALUES = {
    "HIPBLASLT_LOG_LEVEL": "0",
    "HIPBLASLT_LOG_MASK": "0",
    "MIOPEN_ENABLE_LOGGING": "0",
    "MIOPEN_ENABLE_LOGGING_CMD": "0",
    "MIOPEN_LOG_LEVEL": "1",
    "ROCBLAS_LAYER": "0",
}
_GRID_DESC_MARKERS = ("_grid_desc_", "_container_")
_INSTALLED_STREAM_FILTERS: list[bool] = []


def is_ck_grid_desc_line(line: str) -> bool:
    """Return True when ``line`` is a CK GEMM grid-descriptor dump.

    Args:
        line: One host-log line, with or without a trailing newline.

    Returns:
        True if the line matches CK ``*_grid_desc_*_container_`` dumps.
    """
    return all(marker in line for marker in _GRID_DESC_MARKERS)


def silence_rocm_kernel_logs(*, filter_host_streams: bool = True) -> None:
    """Quiet ROCm host logs that print CK GEMM grid descriptors.

    Purpose:
        Stop ``a_grid_desc_m_ak_container_``-style lines from appearing
        during convolution and matmul launches.

    Strategy:
        Unset presence-gated ``CK_LOGGING``, set MIOpen/hipBLASLt/rocBLAS
        to quiet, then optionally copy C++ stdout/stderr through a filter
        because CK ``Argument::Print`` writes to ``std::cout`` when
        ``StreamConfig.log_level_`` is greater than zero.

    Args:
        filter_host_streams: If True, drop matching lines from process
            stdout and stderr file descriptors.

    Returns:
        None. Updates ``os.environ`` and may redirect host streams.

    Flow:
        1. Unset presence-gated CK logging
        2. Apply quiet library log levels
        3. Optionally filter host streams
    """
    # 1. Unset presence-gated CK logging
    for name in _UNSET_NAMES:
        os.environ.pop(name, None)
    # 2. Apply quiet library log levels
    os.environ.update(_QUIET_VALUES)
    # 3. Optionally filter host streams
    if filter_host_streams:
        _install_host_stream_filters()


def _install_host_stream_filters() -> None:
    """Redirect libc stdout and stderr through a CK grid-desc line filter.

    Returns:
        None. No-ops when filters are already installed.
    """
    if _INSTALLED_STREAM_FILTERS:
        return
    _redirect_fd(1, "stdout")
    _redirect_fd(2, "stderr")
    _INSTALLED_STREAM_FILTERS.append(True)


def _redirect_fd(fd: int, stream_name: str) -> None:
    """Send C++ writes on ``fd`` through a dropping copy thread.

    Purpose:
        Intercept ``std::cout``/``std::cerr`` while leaving Python's
        ``sys.stdout``/``sys.stderr`` on the original terminal.

    Args:
        fd: Process file descriptor, ``1`` for stdout or ``2`` for stderr.
        stream_name: ``stdout`` or ``stderr`` attribute name on ``sys``.

    Returns:
        None. Starts a daemon copier thread.
    """
    original_fd = os.dup(fd)
    python_fd = os.dup(original_fd)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, fd)
    os.close(write_fd)
    thread = threading.Thread(
        target=_copy_filtered_lines,
        args=(read_fd, original_fd),
        name=f"ck-grid-desc-filter-{stream_name}",
        daemon=True,
    )
    thread.start()
    wrapped = os.fdopen(python_fd, "w", buffering=1, encoding="utf-8", errors="replace")
    if stream_name == "stdout":
        sys.stdout = wrapped
    else:
        sys.stderr = wrapped


def _copy_filtered_lines(read_fd: int, write_fd: int) -> None:
    """Copy bytes from ``read_fd`` to ``write_fd``, dropping CK dump lines.

    Args:
        read_fd: Pipe that now receives C++ host writes.
        write_fd: Original terminal or capture descriptor.

    Returns:
        None. Runs until the read end closes.
    """
    buf = b""
    try:
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            buf += chunk
            buf = _flush_complete_lines(buf, write_fd)
            if buf and b"grid_desc_" not in buf:
                os.write(write_fd, buf)
                buf = b""
    finally:
        if buf and not is_ck_grid_desc_line(buf.decode("utf-8", errors="replace")):
            os.write(write_fd, buf)
        os.close(read_fd)


def _flush_complete_lines(buf: bytes, write_fd: int) -> bytes:
    """Write non-CK newline-terminated lines from ``buf`` to ``write_fd``.

    Args:
        buf: Accumulated unread host output.
        write_fd: Original terminal or capture descriptor.

    Returns:
        The unterminated remainder of ``buf``.
    """
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        text = line.decode("utf-8", errors="replace")
        if not is_ck_grid_desc_line(text):
            os.write(write_fd, line + b"\n")
    return buf
