import subprocess
from pathlib import Path

import structlog
from job_wrapper.job import job_wrapper
from pydantic import BaseModel, field_validator


class SlimParams(BaseModel):
    K: int
    SD: float
    SI: float
    SM: float
    L: int
    W: float
    G: float
    MU: int
    R: float
    SIM_END: float
    IND_RECORD_LIM: int
    IND_RECORD_LAG: int
    IND_RECORD_FREQ: float
    OUTPATH: Path

    @field_validator("OUTPATH")
    @classmethod
    def contains_units(cls, v: Path) -> Path:
        assert not v.exists(), f"{v} already exists"
        return v


class SLiMJobArgs(BaseModel):
    in_files: None
    out_files: Path
    params: SlimParams


@job_wrapper(SLiMJobArgs)
def main(args: SLiMJobArgs) -> None:
    log = structlog.get_logger(module=__name__)
    log.info("starting")

    params_file = Path("params.json")
    if params_file.exists():
        raise Exception(f"{params_file} already exists")

    with open(params_file, "w") as fp:
        fp.write(args.params.model_dump_json())

    log.info("running SLiM")
    subprocess.run(
        ["slim", "-d", "PARAM_FILE='params.json'", "/opt/main.slim"], check=True
    )
