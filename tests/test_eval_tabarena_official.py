import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_tabarena_official", ROOT / "eval_tabarena_official.py")
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


def test_checkpoint_metadata_rejects_wrong_step_or_penalty(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    torch = ModuleType("torch")
    torch.load = lambda *args, **kwargs: {"config": {}, "state_dict": {}, "step": 2000,
                                          "experiment": {"lambda_fg": .5}}
    monkeypatch.setitem(sys.modules, "torch", torch)
    info = official.checkpoint_info(checkpoint, "penalized", 2000)
    assert info["step"] == 2000 and len(info["sha256"]) == 64
    with pytest.raises(ValueError, match="lambda_fg"):
        official.checkpoint_info(checkpoint, "control", 2000)
    with pytest.raises(ValueError, match="expected 1900"):
        official.checkpoint_info(checkpoint, "penalized", 1900)


def test_official_runner_uses_lite_classification_and_three_named_models(tmp_path, monkeypatch):
    state = {}

    class FakeGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeBundle:
        def __init__(self, **kwargs):
            state["bundle"] = kwargs

        def build_experiments(self, **kwargs):
            state["resources"] = kwargs
            return [SimpleNamespace(name=model[0].kwargs["model_cls"].ag_name) for model in state["bundle"]["models"]]

    class FakeContext:
        task_metadata_collection = object()

        def build_and_run_jobs(self, experiments, **kwargs):
            state["run"] = kwargs

        def register(self, methods):
            state["registered"] = methods

        def compare(self, **kwargs):
            state["compare"] = kwargs
            return [1, 2, 3]

    class FakeEndToEnd:
        @staticmethod
        def from_path_raw(path, **kwargs):
            state.setdefault("processed", []).append(kwargs)
            return SimpleNamespace(to_method_metadata_lst=lambda **_: [kwargs["method"]])

    models = ModuleType("official_tabarena_models")
    models.MODEL_CLASSES = tuple(type(f"Model{i}", (), {"ag_name": f"variant{i}"}) for i in range(3))
    benchmark = ModuleType("tabarena.benchmark.experiment")
    benchmark.TabArenaV0pt1ExperimentBundle = FakeBundle
    config = ModuleType("tabarena.utils.config_utils")
    config.ConfigGenerator = FakeGenerator
    contexts = ModuleType("tabarena.contexts")
    contexts.TabArenaContext = FakeContext
    end_to_end = ModuleType("tabarena.end_to_end")
    end_to_end.EndToEnd = FakeEndToEnd
    for name, module in {"official_tabarena_models": models, "tabarena.benchmark.experiment": benchmark,
                         "tabarena.utils.config_utils": config, "tabarena.contexts": contexts,
                         "tabarena.end_to_end": end_to_end}.items():
        monkeypatch.setitem(sys.modules, name, module)
    checkpoints = [{"path": str(tmp_path / name)} for name in official.LABELS]
    experiments = official.build_experiments(checkpoints)
    assert [e.name for e in experiments] == ["variant0", "variant1", "variant2"]
    assert state["bundle"]["default_seed_config"] == "static"
    assert all(count == 0 for _, count in state["bundle"]["models"])
    assert state["resources"] == {"num_gpus": 1}
    for experiment in experiments:
        for task in range(38):
            path = tmp_path / "experiments" / "data" / experiment.name / str(task) / "0_0" / "results.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    official.run_official(experiments, output=tmp_path)
    assert state["run"]["subset"] == "lite"
    assert state["run"]["build_kwargs"] == {"problem_types": ["binary", "multiclass"]}
    assert state["run"]["register"] is False
    assert len(state["registered"]) == 3
    assert len({item["model_key"] for item in state["processed"]}) == 3
    assert state["compare"]["subset"] == ["lite", "classification"]
    assert state["compare"]["new_methods_only"] is True
    state.pop("run")
    official.run_official(experiments, output=tmp_path, report_only=True)
    assert "run" not in state
    official.run_official(experiments, output=tmp_path, full=True)
    assert state["run"]["subset"] is None
