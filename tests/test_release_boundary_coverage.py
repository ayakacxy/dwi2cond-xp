"""核验发布新增配置、文件竞态及下游消费前的失败边界。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from dwi2cond_xp import leadfield, simulation
from dwi2cond_xp.preprocessing import pipeline, topup_eddy, workflow


def _save_nifti(path: Path, values: np.ndarray) -> None:
    """仅创建测试自己的合成体数据，保留有效空间 header。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)


def _make_subject(directory: Path) -> Path:
    """创建用于参数预检的最小合成 CHARM 目录，不执行 FEM。"""

    subject = directory / "m2m_boundary"
    values = np.ones((2, 3, 4), dtype=np.float32)
    for name in (
        "T1.nii.gz",
        "final_tissues.nii.gz",
        "segmentation/labeling.nii.gz",
        "segmentation/T1_bias_corrected.nii.gz",
    ):
        _save_nifti(subject / name, values)
    (subject / "boundary.msh").write_text("", encoding="utf-8")
    cap = subject / "eeg_positions/EEG10-10_UI_Jurak_2007.csv"
    cap.parent.mkdir()
    cap.write_text("Electrode\n", encoding="utf-8")
    return subject


@pytest.mark.parametrize("builder", [leadfield.build_tdcs_leadfield, simulation.build_tdcs_session])
def test_scalar_builders_reject_invalid_eigensystem_before_tensor_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builder,
) -> None:
    """标量模式同样必须拒绝未知特征分解模式，且不创建输出。"""

    simnibs = ModuleType("simnibs")
    simnibs.__version__ = "4.6.0"
    structure = ModuleType("simnibs.sim_struct")
    structure.TDCSLEADFIELD = SimpleNamespace
    structure.SESSION = SimpleNamespace
    simnibs.sim_struct = structure
    monkeypatch.setitem(sys.modules, "simnibs", simnibs)
    monkeypatch.setitem(sys.modules, "simnibs.sim_struct", structure)
    contract = {
        "subpath": str(tmp_path / "m2m_boundary"),
        "head_mesh": str(tmp_path / "boundary.msh"),
        "eeg_cap": str(tmp_path / "cap.csv"),
        "mode": "scalar",
        "tensor": None,
    }
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="eigensystem_mode must be stable or simnibs46-literal"):
        builder(contract, output, eigensystem_mode="unknown")
    assert not output.exists()


@pytest.mark.parametrize("runner", [leadfield.run_tdcs_leadfield, simulation.run_tdcs])
def test_scalar_runners_reject_invalid_eigensystem_before_attempt_directory(
    tmp_path: Path, runner,
) -> None:
    """真实输入预检通过后，无效模式仍必须在发布事务开始之前报错。"""

    subject = _make_subject(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="eigensystem_mode must be stable or simnibs46-literal"):
        runner(subject, output, mode="scalar", dry_run=True, eigensystem_mode="unknown")
    assert not output.exists()
    assert not list(tmp_path.glob(".*.attempt-*"))


def test_file_fingerprint_rejects_real_mutation_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """哈希读取后实际替换输入，不能把旧摘要与新元数据组合成可复用指纹。"""

    source = tmp_path / "input.txt"
    source.write_bytes(b"original")
    hash_file = pipeline._sha256_file

    def hash_then_change(path: Path) -> str:
        digest = hash_file(path)
        path.write_bytes(b"changed-longer-content")
        return digest

    monkeypatch.setattr(pipeline, "_sha256_file", hash_then_change)
    with pytest.raises(ValueError, match="file changed while hashing"):
        pipeline.fingerprint_file(source)
    assert source.read_bytes() == b"changed-longer-content"


@pytest.mark.parametrize("workers", [0, 1.5])
def test_artifact_validation_rejects_invalid_worker_budget(workers) -> None:
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        pipeline.validate_artifacts((), workers=workers)


@pytest.mark.parametrize("workers", [0, 1.5])
def test_pipeline_runner_rejects_invalid_worker_budget(tmp_path: Path, workers) -> None:
    manifests = tmp_path / "manifests"
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        pipeline.PipelineRunner(manifests, workers=workers)
    assert not manifests.exists()


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("bvec_mode", "unknown", "bvec_mode must be corrected or compat46"),
        ("component_mode", "unknown", "component_mode must be physical"),
        ("jacobian_mode", "unknown", "jacobian_mode must be physical"),
        ("basis_mode", "unknown", "basis_mode must be physical"),
        ("eigensystem_mode", "unknown", "eigensystem_mode must be stable"),
        ("bvec_mode", "compat46", "only consumed by legacy preprocessing"),
    ],
)
def test_workflow_rejects_invalid_or_unused_geometry_modes(
    tmp_path: Path, setting: str, value: str, message: str,
) -> None:
    """不读取或创建计算产物，先拒绝未知或当前分支不消费的坐标选项。"""

    config = workflow.Dwi2CondPipelineConfig(
        data=None, bvals=None, bvecs=None,
        m2m_directory=tmp_path / "m2m_boundary",
        output_directory=tmp_path / "output",
        preprocessing_mode="nomoco", t1_mode="affine",
    )
    with pytest.raises(ValueError, match=message):
        workflow.run_dwi2cond_pipeline(replace(config, **{setting: value}))
    assert not config.output_directory.exists()


@pytest.mark.parametrize("component_mode", ["physical", "simnibs46-literal"])
def test_prefit_import_rejects_nonfinite_tensor_before_decomposition(
    tmp_path: Path, component_mode: str,
) -> None:
    """实际重排后的非有限张量不能继续产生 FA、有效性掩膜或完成报告。"""

    source = tmp_path / "tensor.nii.gz"
    values = np.zeros((2, 3, 4, 6), dtype=np.float32)
    values[0, 0, 0, 0] = np.nan
    _save_nifti(source, values)
    before = source.read_bytes()
    output = tmp_path / "prefit"
    with pytest.raises(ValueError, match="pre-fitted tensor contains NaN or Inf"):
        workflow._import_prefit_tensor(source, output, component_mode=component_mode)
    assert source.read_bytes() == before
    assert not (output / "DTI_FA.nii.gz").exists()
    assert not (output / "DTI_valid_mask.nii.gz").exists()
    assert not (output / "prefit_qa.json").exists()


def test_prefit_import_defensively_rejects_malformed_producer_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """故障注入重排函数的错误产物，检查独立形状防线；不标记导入成功。"""

    source = tmp_path / "tensor.nii.gz"
    _save_nifti(source, np.zeros((2, 3, 4, 6), dtype=np.float32))
    before = source.read_bytes()

    def malformed_output(_source, destination, **_kwargs):
        _save_nifti(destination, np.zeros((2, 3, 4, 5), dtype=np.float32))
        return destination

    monkeypatch.setattr(workflow, "write_fsl_reoriented", malformed_output)
    output = tmp_path / "prefit"
    with pytest.raises(ValueError, match="pre-fitted tensor must have shape XxYxZx6"):
        workflow._import_prefit_tensor(source, output)
    assert source.read_bytes() == before
    assert not (output / "DTI_FA.nii.gz").exists()
    assert not (output / "prefit_qa.json").exists()


def test_topup_eddy_rejects_unknown_component_mode_before_image_io(tmp_path: Path) -> None:
    output = tmp_path / "topup_eddy"
    with pytest.raises(ValueError, match="component_mode must be physical or simnibs46-literal"):
        topup_eddy.run_topup_eddy_nifti(
            tmp_path / "missing-dwi", tmp_path / "missing-bvals",
            tmp_path / "missing-bvecs", tmp_path / "missing-reverse", output,
            readout_seconds=0.05, phase_encoding_direction="y", component_mode="unknown",
        )
    assert not output.exists()


def test_eddy_field_is_tracked_as_stage_input_output_and_passed_to_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """检查真实合成外部 Hz 场的依赖与参数传递，在求解入口明确停止。"""

    subject = _make_subject(tmp_path)
    dwi = tmp_path / "dwi.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    field = tmp_path / "field-hz.nii.gz"
    _save_nifti(dwi, np.ones((2, 3, 4, 2), dtype=np.float32))
    _save_nifti(mask, np.ones((2, 3, 4), dtype=np.uint8))
    _save_nifti(field, np.zeros((2, 3, 4), dtype=np.float32))
    bvals = tmp_path / "bvals"
    bvecs = tmp_path / "bvecs"
    bvals.write_text("0 1000\n", encoding="utf-8")
    bvecs.write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
    config = workflow.Dwi2CondPipelineConfig(
        data=dwi, bvals=bvals, bvecs=bvecs,
        m2m_directory=subject, output_directory=tmp_path / "output",
        preprocessing_mode="eddy", t1_mode="affine", workers=1,
        dwi_brain_mask=mask, susceptibility_field=field,
        readout_seconds=0.05, phase_encoding_direction="y",
    )

    class ConsumerBoundary(RuntimeError):
        """刻意在昂贵求解之前停止，不伪造已完成的科学产物。"""

    def check_consumer(data, _bvals, _bvecs, input_mask, output, **kwargs):
        assert data == dwi
        assert input_mask == mask
        assert kwargs["susceptibility_field_file"] == field
        assert kwargs["component_mode"] == "physical"
        assert output == config.output_directory / "preprocess/eddy"
        raise ConsumerBoundary("field reached EDDY consumer")

    def inspect_dag(_runner, stages):
        stage = next(item for item in stages if item.name == "preprocess_eddy")
        assert field in stage.inputs
        assert pipeline.ArtifactContract(
            config.output_directory / "preprocess/eddy/susceptibility_field_hz.nii.gz",
            "nifti", ndim=3,
        ) in stage.outputs
        stage.action()

    monkeypatch.setattr(workflow, "run_eddy_nifti", check_consumer)
    monkeypatch.setattr(pipeline.PipelineRunner, "run", inspect_dag)
    with pytest.raises(ConsumerBoundary, match="field reached EDDY consumer"):
        workflow.run_dwi2cond_pipeline(config)
    assert not (subject / "DTI_coregT1_tensor.nii.gz").exists()
    assert not (config.output_directory / "qa/pipeline_qa.json").exists()


def test_nomoco_reuse_binds_prepared_gradient_deviation_to_producer_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实际配置中的 grad-dev 必须同时绑定准备产物及其生产者，避免错误缓存复用。"""

    subject = _make_subject(tmp_path)
    dwi = tmp_path / "dwi.nii.gz"
    grad_dev = tmp_path / "grad-dev.nii.gz"
    _save_nifti(dwi, np.ones((2, 3, 4, 2), dtype=np.float32))
    _save_nifti(grad_dev, np.zeros((2, 3, 4, 9), dtype=np.float32))
    bvals = tmp_path / "bvals"
    bvecs = tmp_path / "bvecs"
    bvals.write_text("0 1000\n", encoding="utf-8")
    bvecs.write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
    config = workflow.Dwi2CondPipelineConfig(
        data=dwi, bvals=bvals, bvecs=bvecs, grad_dev=grad_dev,
        m2m_directory=subject, output_directory=tmp_path / "output",
        preprocessing_mode="nomoco", t1_mode="affine", workers=1,
    )

    class DagBoundary(RuntimeError):
        """检查完依赖即停止，避免用假拟合结果代替科学运行。"""

    def inspect_dag(_runner, stages):
        producer = next(stage for stage in stages if stage.name == "fit_raw_dti_qa")
        consumer = next(stage for stage in stages if stage.name == "preprocess_nomoco")
        prepared = config.output_directory / "preprocess/raw_dti_qa/grad_dev.nii"
        assert grad_dev in producer.inputs
        assert pipeline.ArtifactContract(prepared, "nifti", ndim=4, final_axis=9) in producer.outputs
        assert grad_dev in consumer.inputs
        assert prepared in consumer.inputs
        assert consumer.dependencies == (producer.name,)
        assert consumer.parameters["reuse_raw_preparation"] is True
        raise DagBoundary("prepared grad-dev dependency verified")

    monkeypatch.setattr(pipeline.PipelineRunner, "run", inspect_dag)
    with pytest.raises(DagBoundary, match="prepared grad-dev dependency verified"):
        workflow.run_dwi2cond_pipeline(config)
    assert not (config.output_directory / "manifests").exists()
    assert not (subject / "DTI_coregT1_tensor.nii.gz").exists()
