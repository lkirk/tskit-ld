import subprocess
from pathlib import Path

import structlog
import tskit
import tszip
from htcluster.job_wrapper.job import job_wrapper
from htcluster.validator_base import BaseModel
from pydantic import field_validator


class SlimParams(BaseModel):
    K: int
    SD: float
    SI: float
    SM: float
    L: int
    W: float
    H: float
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

    params_file = Path("params.json")
    if params_file.exists():
        raise Exception(f"{params_file} already exists")

    with open(params_file, "w") as fp:
        fp.write(args.params.model_dump_json())
    log.info("wrote input params", file=str(params_file))

    log.info("running SLiM")
    subprocess.run(
        ["slim", "-d", "PARAM_FILE='params.json'", "/opt/main.slim"], check=True
    )

    uncompressed_path = args.params.OUTPATH
    compressed_path = uncompressed_path.with_suffix(f"{uncompressed_path.suffix}.tsz")
    log.info(
        "SLiM has completed, compressing output tree",
        tree_path=str(uncompressed_path),
        compressed_path=str(compressed_path),
    )
    ts = tskit.load(uncompressed_path)
    tszip.compress(ts, compressed_path)
    log.info("compression complete, removing uncompressed tree")
    compressed_path.unlink()
