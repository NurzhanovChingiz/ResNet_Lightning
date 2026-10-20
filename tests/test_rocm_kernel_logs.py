"""Contracts for silencing Composable Kernel GEMM grid-descriptor dumps."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from utils.rocm_kernel_logs import (
    _copy_filtered_lines,
    disable_miopen_winograd_fury_solvers,
    is_ck_grid_desc_line,
    silence_rocm_kernel_logs,
    use_miopen_fast_find_mode,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "line",
    [
        "a_grid_desc_m_ak_container_{18432, 1152}",
        "b_grid_desc_n_bk_container_{128, 1152}",
        "e_grid_desc_mblock_mperblock_nblock_nperblock_container_{18432, 128}",
    ],
)
def test_is_ck_grid_desc_line_matches_host_dumps(line: str) -> None:
    """CK grid-descriptor dump lines are classified as log spam."""
    assert is_ck_grid_desc_line(line)


@pytest.mark.parametrize(
    "line",
    [
        "Using device: cuda",
        "Epoch 1: 100% loss=0.12",
    ],
)
def test_is_ck_grid_desc_line_ignores_training_output(line: str) -> None:
    """Ordinary training console lines are not treated as CK dumps."""
    assert not is_ck_grid_desc_line(line)


def test_copy_filtered_lines_drops_ck_dumps_and_keeps_other_text() -> None:
    """The host-stream copier must drop CK dumps and forward other lines."""
    read_in, write_in = os.pipe()
    read_out, write_out = os.pipe()
    os.write(
        write_in,
        b"a_grid_desc_m_ak_container_{18432, 1152}\nkeep me\n"
        b"e_grid_desc_mblock_mperblock_nblock_nperblock_container_{18432, 128}\n",
    )
    os.close(write_in)
    _copy_filtered_lines(read_in, write_out)
    os.close(write_out)
    data = os.read(read_out, 1024)
    os.close(read_out)
    assert data == b"keep me\n"


def test_silence_rocm_kernel_logs_unsets_ck_logging_and_quiets_miopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CK_LOGGING must be removed and MIOpen must be set to quiet."""
    monkeypatch.setenv("CK_LOGGING", "1")
    monkeypatch.delenv("MIOPEN_LOG_LEVEL", raising=False)
    silence_rocm_kernel_logs(filter_host_streams=False)
    assert "CK_LOGGING" not in os.environ
    assert os.environ["MIOPEN_LOG_LEVEL"] == "1"
    assert os.environ["MIOPEN_ENABLE_LOGGING"] == "0"
    assert os.environ["HIPBLASLT_LOG_LEVEL"] == "0"
    assert os.environ["ROCBLAS_LAYER"] == "0"


def test_silence_without_stream_filter_leaves_sys_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env quieting must not wrap streams when filtering is disabled."""
    original_stdout = sys.stdout
    monkeypatch.delenv("CK_LOGGING", raising=False)
    silence_rocm_kernel_logs(filter_host_streams=False)
    assert sys.stdout is original_stdout


def test_disable_miopen_winograd_fury_solvers_sets_debug_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Winograd Fury solvers must be turned off before convolution find."""
    monkeypatch.delenv("MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3", raising=False)
    monkeypatch.delenv("MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2", raising=False)
    disable_miopen_winograd_fury_solvers()
    assert os.environ["MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3"] == "0"
    assert os.environ["MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2"] == "0"


def test_use_miopen_fast_find_mode_sets_find_mode_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAST find must be selected so gfx1151 skips live solver timing."""
    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
    use_miopen_fast_find_mode()
    assert os.environ["MIOPEN_FIND_MODE"] == "2"


def test_prepare_runtime_quiets_rocm_kernel_logs_before_gpu_setup() -> None:
    """prepare_runtime must silence CK dumps before probing the GPU."""
    source = (REPO_ROOT / "config.py").read_text(encoding="utf-8")
    start = source.index("def prepare_runtime")
    body = source[start : source.index("def ensure_download_path")]
    quiet_call = body.index("silence_rocm_kernel_logs()")
    wino_call = body.index("disable_miopen_winograd_fury_solvers()")
    find_call = body.index("use_miopen_fast_find_mode()")
    gpu_call = body.index("resolve_gpu_runtime()")
    math_call = body.index("enable_medium_float32_matmul_precision()")
    bench_call = body.index("disable_cudnn_benchmark()")
    assert quiet_call < wino_call < find_call < gpu_call < math_call < bench_call


def test_makefile_train_unsets_ck_logging() -> None:
    """Make train must unset CK_LOGGING before Python starts."""
    source = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = source[source.index("train:") :]
    assert "env -u CK_LOGGING" in recipe
    assert "MIOPEN_LOG_LEVEL=1" in recipe
    assert "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3=0" in recipe
    assert "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2=0" in recipe
    assert "MIOPEN_FIND_MODE=2" in recipe
