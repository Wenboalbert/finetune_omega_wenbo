"""Immutable four-arm endpoint barrier. Reads no geometry ground truth."""
import datetime
from pathlib import Path
from .io import read_json,write_json,sha256
from .trust_policy import TERMINATIONS

FROZEN_FILES=("baseline/predictions.npz","adapted/post14_D02_registers/residual.pt",
              "adapted/post14_D02_registers/predictions.npz",
              "adapted/post14_D02_registers/optimization.json")


def freeze_endpoint(run, reason, counters, extra):
    run=Path(run)
    if reason not in TERMINATIONS: raise ValueError("unknown termination")
    data=dict(status="FROZEN",termination_reason=reason,counters=counters,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files={p:sha256(run/p) for p in FROZEN_FILES},
        config_sha256=sha256(run/"resolved_config.json"),
        study_sha256=sha256(run/"study_manifest.json"),**extra)
    write_json(run/"logs/ladder_completion.json",data)
    return data


def assert_study_frozen(run):
    run=Path(run)
    study=read_json(run/"study_manifest.json")
    if len(study["runs"])!=4: raise ValueError("frozen ladder requires four arms")
    paths=[Path(x["path"]).resolve() for x in study["runs"]]
    if len(set(paths))!=4 or run.resolve() not in paths: raise ValueError("invalid study members")
    endpoints=[]
    for item,path in zip(study["runs"],paths):
        if read_json(path/"study_manifest.json")!=study: raise ValueError("study manifest mismatch")
        if sha256(path/"resolved_config.json")!=item["config_sha256"]: raise ValueError("config changed")
        done=read_json(path/"logs/ladder_completion.json")
        if done["config_sha256"]!=sha256(path/"resolved_config.json") or done["study_sha256"]!=sha256(path/"study_manifest.json"):
            raise ValueError("endpoint configuration/study changed")
        if done["status"]!="FROZEN" or done["termination_reason"] not in TERMINATIONS:
            raise RuntimeError("not all four endpoints are frozen")
        if set(done["files"])!=set(FROZEN_FILES): raise ValueError("incomplete endpoint hashes")
        for rel,digest in done["files"].items():
            if sha256(path/rel)!=digest: raise ValueError("frozen endpoint changed: "+str(path/rel))
        endpoints.append(dict(run=str(path),completion_sha256=sha256(path/"logs/ladder_completion.json"),
                              termination_reason=done["termination_reason"]))
    return dict(study_id=study["study_id"],endpoints=endpoints,all_four_frozen_before_gt=True)
