#!/usr/bin/env python3
"""运行带源码与输入指纹的固定 nonlinear 官方对照基准。"""

from __future__ import annotations

import argparse
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from dwi2cond_xp.preprocessing import (
    ReferenceArtifact,
    run_reference_command,
    summarize_fixture_inputs,
)


def _positive_float(value: str) -> float:
    """解析严格大于零的浮点参数。"""

    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("数值必须大于零")
    return parsed


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """在同一目录原子写入 JSON，避免长任务留下半份合同。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """以固定分块计算文件内容哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    """记录当前提交；非 Git 源码包返回空值。"""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _cpu_affinity() -> list[int] | None:
    """读取 Linux 进程亲和性；其他平台明确返回空值。"""

    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    return sorted(int(cpu) for cpu in getter(0))


def _physical_cores(cpus: list[int] | None) -> list[list[int]] | None:
    """把 Linux 逻辑 CPU 映射为 package/core 对，供 8 物理核门禁审计。"""

    if cpus is None:
        return None
    cores: set[tuple[int, int]] = set()
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((topology / "physical_package_id").read_text().strip())
            core = int((topology / "core_id").read_text().strip())
        except (OSError, ValueError):
            return None
        cores.add((package, core))
    return [[package, core] for package, core in sorted(cores)]


def _simnibs_external_directory(explicit: Path | None) -> Path:
    """解析 SimNIBS external 源码目录，不依赖当前 shell 的 PATH。"""

    if explicit is not None:
        return explicit.resolve()
    spec = find_spec("simnibs")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(
            "当前环境找不到 SimNIBS；请用 --simnibs-external 指定 4.6 external 目录"
        )
    return Path(next(iter(spec.submodule_search_locations))) / "external"


def _source_files(
    root: Path,
    fsl_dir: Path,
    simnibs_external: Path,
) -> tuple[Path, ...]:
    """返回决定当前 nonlinear 算法合同的上游与本地源码。"""

    candidates = (
        simnibs_external / "dwi2cond",
        simnibs_external / "dwi2cond.t1reg.source.sh",
        fsl_dir / "src/fnirt/fnirt.cpp",
        fsl_dir / "src/fnirt/fnirtfns.cpp",
        fsl_dir / "src/fnirt/fnirt_costfunctions.cpp",
        fsl_dir / "src/fdt/vecreg.cc",
        fsl_dir / "src/avwutils/fslmaths.cc",
        root / "src/dwi2cond_xp/preprocessing/fnirt.py",
        root / "src/dwi2cond_xp/preprocessing/fnirt_topology.py",
        root / "src/dwi2cond_xp/preprocessing/nonlinear.py",
        root / "src/dwi2cond_xp/preprocessing/tensor_ops.py",
    )
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少 nonlinear 源码审计文件：" + ", ".join(str(path) for path in missing)
        )
    return candidates


def _source_summary(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    """冻结源码名称、大小和哈希，不把私有绝对路径写入合同。"""

    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _build_contract(args: argparse.Namespace, root: Path) -> dict[str, object]:
    """构建计时外的同输入、同算法源码合同。"""

    affinity = _cpu_affinity()
    physical = _physical_cores(affinity)
    simnibs_external = _simnibs_external_directory(args.simnibs_external)
    source_files = _source_files(root, args.fsl_dir.resolve(), simnibs_external)
    return {
        "schema_version": 1,
        "boundary": "SimNIBS 4.6 fixed FNIRT plus FSL vecreg PPD stage",
        "implementation": args.implementation,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "input_contract": summarize_fixture_inputs(
            {
                "affine_matrix": args.affine,
                "brain_mask": args.brain_mask,
                "fa": args.fa,
                "reference": args.reference,
                "tensor": args.tensor,
            },
            nifti_aliases=("brain_mask", "fa", "reference", "tensor"),
            mask_aliases=("brain_mask",),
            include_digests=True,
        ),
        "algorithm_sources": _source_summary(source_files),
        "algorithm_flow": [
            "dwi2cond: FLIRT 12-DOF matrix",
            "FNIRT: subsamp=8,4,2,2 with unchanged defaults",
            "VECREG: nonlinear trilinear tensor sampling plus PPD",
            "fslmaths: apply T1 brain mask after vecreg",
            "fslmaths: tensor_decomp derived FA and V1",
        ],
        "runtime": {
            "git_commit": _git_commit(root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_affinity": affinity,
            "physical_core_ids": physical,
            "strict_worker_affinity": affinity is not None
            and physical is not None
            and len(affinity) == args.workers
            and len(physical) == args.workers,
        },
    }


def _validate_fresh_output(path: Path) -> None:
    """拒绝复用已有 artifact，保证 fresh-output 计时边界。"""

    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"基准输出目录必须不存在或为空：{path}")


def _python_artifacts() -> tuple[ReferenceArtifact, ...]:
    """列出 Python nonlinear 产品边界的必需输出。"""

    return (
        ReferenceArtifact("FA2T1_warp.nii.gz", "nifti"),
        ReferenceArtifact("FA2T1_field.nii.gz", "nifti"),
        ReferenceArtifact("FA2T1_jacobian.nii.gz", "nifti"),
        ReferenceArtifact("DTI_FA_nonlin.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_tensor.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_FA.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_V1.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_jacobian.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_valid_mask.nii.gz", "nifti", mask=True),
        ReferenceArtifact("DTI_coregT1_nonlinear_qa.json"),
        ReferenceArtifact("nonlinear_registration_qa.json"),
    )


def _fsl_artifacts() -> tuple[ReferenceArtifact, ...]:
    """列出原始 FSL nonlinear 分支实际写出的共同科学输出。"""

    return (
        ReferenceArtifact("FA2T1_warp.nii.gz", "nifti"),
        ReferenceArtifact("FA2T1_field.nii.gz", "nifti"),
        ReferenceArtifact("FA2T1_jacobian.nii.gz", "nifti"),
        ReferenceArtifact("DTI_FA_nonlin.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_tensor.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_FA.nii.gz", "nifti"),
        ReferenceArtifact("DTI_coregT1_V1.nii.gz", "nifti"),
        ReferenceArtifact("fnirt.log"),
    )


def main() -> int:
    """冻结合同，并按显式实现运行一次 fresh-output nonlinear 阶段。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("python", "fsl"), required=True)
    parser.add_argument("--fa", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--brain-mask", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--fsl-dir", type=Path, default=Path("/usr/local/fsl"))
    parser.add_argument("--simnibs-external", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=7200.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="仅供预检或 smoke；正式性能运行不得使用",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers 必须大于零")

    root = Path(__file__).resolve().parents[1]
    contract_path = (
        args.contract.resolve()
        if args.contract is not None
        else args.manifest.resolve().with_name(f"{args.manifest.stem}.contract.json")
    )
    contract = _build_contract(args, root)
    _atomic_json(contract_path, contract)
    if args.preflight_only:
        print(
            json.dumps(
                {"status": "preflight-completed", "contract": str(contract_path)}
            )
        )
        return 0

    if not contract["runtime"]["strict_worker_affinity"] and not args.allow_unpinned:
        parser.error(
            "正式运行要求进程已被限制为与 --workers 相同数量的独立物理核；"
            "请先使用 taskset 绑定，smoke 才可传 --allow-unpinned"
        )
    work = args.work.resolve()
    _validate_fresh_output(work)
    fsl_dir = args.fsl_dir.resolve()
    environment = {
        "FSLDIR": str(fsl_dir),
        "FSLOUTPUTTYPE": "NIFTI_GZ",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": str(args.workers),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": os.pathsep.join((str(fsl_dir / "bin"), os.environ.get("PATH", ""))),
    }
    source_files = _source_files(
        root,
        fsl_dir,
        _simnibs_external_directory(args.simnibs_external),
    ) + (contract_path,)
    if args.implementation == "python":
        arguments = (
            "-m",
            "dwi2cond_xp",
            "register-t1-nonlinear",
            str(args.fa.resolve()),
            str(args.tensor.resolve()),
            str(args.reference.resolve()),
            str(args.affine.resolve()),
            str(work),
            "--brain-mask",
            str(args.brain_mask.resolve()),
            "--workers",
            str(args.workers),
            "--progress",
            "off",
        )
        artifacts = _python_artifacts()
        stage = "dwi2cond-xp-v030-fixed-nonlinear"
        version = "dwi2cond-xp current source"
        threads = args.workers
    else:
        runner = root / "tools/run_fsl_fnirt_reference.py"
        arguments = (
            str(runner),
            "--worker",
            "--fa",
            str(args.fa.resolve()),
            "--tensor",
            str(args.tensor.resolve()),
            "--reference",
            str(args.reference.resolve()),
            "--affine",
            str(args.affine.resolve()),
            "--brain-mask",
            str(args.brain_mask.resolve()),
            "--work",
            str(work),
            "--fsl-dir",
            str(fsl_dir),
        )
        artifacts = _fsl_artifacts()
        stage = "simnibs46-fsl604-v030-fixed-nonlinear"
        version = "SimNIBS 4.6.0 dwi2cond 0.4 / FSL 6.0.4:ddd0a010"
        threads = 1

    manifest = run_reference_command(
        stage=stage,
        executable=sys.executable,
        arguments=arguments,
        working_directory=work,
        manifest_path=args.manifest,
        artifacts=artifacts,
        environment=environment,
        reference_version=version,
        script_paths=source_files,
        threads=threads,
        timeout_seconds=args.timeout_seconds,
        include_output_digests=True,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "contract": str(contract_path),
                "manifest": str(args.manifest.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
