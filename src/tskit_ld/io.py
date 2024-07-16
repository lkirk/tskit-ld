import json
import re
from pathlib import Path
from typing import Optional, cast

import polars as pl
import pyarrow as pa
import pyarrow.dataset
import pyarrow.parquet as pq

## General Data IO functionality, used in most steps


def parse_id(p: Path | str, pat: str) -> str:
    result = re.search(pat, str(p))
    if result is None:
        raise ValueError(f"Could not find {pat} in {p}")
    return result.group(1)


def glob_and_parse_id(
    path: Path, glob: str, pat: str, return_ids=False
) -> tuple[list[str], dict[str, Path]] | dict[str, Path]:
    paths = list(path.glob(glob))
    out = {parse_id(p, pat): p for p in paths}
    if len(paths) != len(out):
        raise ValueError("run ids not unique")
    if return_ids is True:
        ids = sorted(out, key=lambda k: tuple(map(int, k.split("-"))))
        return ids, out
    return out


def write_parquet(
    data: pl.DataFrame | pa.Table,
    out_path: Path,
    metadata: Optional[dict[str, str] | dict[bytes, bytes]] = None,
    metadata_from: Optional[Path] = None,
    compression: str = "ZSTD",
    compression_level: Optional[int] = None,
) -> None:
    """
    Write parquet file from either a polars dataframe or a pyarrow table.
    Include user specified metadata in the form of a flat dictionary.
    We use ZSTD compression with the default compression level.
    """
    match data:
        case pl.DataFrame():
            table = data.to_arrow()
        case pa.Table():
            table = data
        case _:
            raise ValueError

    if metadata is None and metadata_from is None:
        raise ValueError("One of `metadata` or `metadata_from` is required")
    elif metadata is not None and metadata_from is not None:
        raise ValueError("`metadata` and `metadata_from` are mutually exclusive")

    if metadata_from is not None:
        metadata = cast(dict[bytes, bytes], pq.read_metadata(metadata_from).metadata)
        del metadata[b"ARROW:schema"]

    # preserve polars enum fields
    schema_fields = {f.name: f for f in table.schema}
    enum_fields = [
        c for c, d in zip(data.columns, data.dtypes) if isinstance(d, pl.Enum)
    ]
    for f in enum_fields:
        schema_fields[f] = schema_fields[f].with_metadata(
            {b"POLARS.CATEGORICAL_TYPE": b"ENUM"}
        )
    schema = pa.schema(schema_fields.values(), metadata)

    with pq.ParquetWriter(
        out_path,
        schema,
        compression=compression,
        compression_level=compression_level,
    ) as writer:
        writer.write_table(table)


def read_metadata_df(in_path: Path) -> pl.DataFrame:
    file_meta = pq.read_metadata(in_path).metadata
    del file_meta[b"ARROW:schema"]
    file_meta = sorted(
        file_meta.items(),
        key=lambda k: tuple(map(int, k[0].decode("utf-8").split("-"))),
    )
    meta = pl.DataFrame(
        [
            {**dict(run_id=k.decode("utf-8")), **json.loads(v.decode("utf-8"))}
            for k, v in file_meta
        ]
    )
    run_id_enum = pl.Enum(meta["run_id"])
    meta = meta.with_columns(run_id=pl.col("run_id").cast(run_id_enum))
    return meta


def read_parquet_file(
    in_path: Path,
    collect: bool = False,
    metadata: bool = False,
) -> tuple[pl.DataFrame, pl.LazyFrame | pl.DataFrame] | pl.LazyFrame | pl.DataFrame:
    """
    Read parquet file and associated metadata. We have to work around the
    fact that serialization into parquet format converts our run_id to a
    categorical variable. Once this is fixed, we won't have to perform our
    cast to the enum type.
    """
    data = pl.scan_parquet(in_path)
    if collect:
        data = data.collect()
    if metadata is False:
        return data
    meta = read_metadata_df(in_path)
    if not collect:
        # hack for now
        return meta, data.cast({"run_id": meta["run_id"].dtype})
    return meta, data


def create_pyarrow_dataset(
    in_dir: Path,
    out_dir: Path,
    partition_keys: list[str],
    flavor: str = "hive",
    compression: str = "ZSTD",
    compression_level: Optional[int] = None,
) -> None:
    assert in_dir.is_dir(), "in_dir must be a directory"
    dataset = pyarrow.dataset.dataset(in_dir, format="parquet")
    file_options = pyarrow.dataset.ParquetFileFormat().make_write_options(
        compression=compression, compression_level=compression_level
    )
    partitioning = pyarrow.dataset.partitioning(
        pa.schema([dataset.schema.field(k) for k in partition_keys]), flavor=flavor
    )
    pyarrow.dataset.write_dataset(
        dataset,
        max_open_files=0,
        base_dir=out_dir,
        format="parquet",
        file_options=file_options,
        partitioning=partitioning,
        existing_data_behavior="error",
    )
