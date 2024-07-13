from time import sleep

import structlog
from htcluster.validators_3_9_compat import JobArgs

def main(args: JobArgs) -> None:
    log = structlog.get_logger()
    log.info("starting", workflow="test")
    assert args.in_files is not None  # mypy
    assert args.out_files is not None  # mypy
    with open(args.in_files, "r") as fp:
        log.info("reading in file", file=args.in_files)
        fp.readlines()
    with open(args.out_files, "w") as fp:
        fp.write("test out")
    log.info("got params", params=args.params)
    sleep(60)
