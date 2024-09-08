import logging
import sys
from collections.abc import Iterator
from typing import Self

import demes
import msprime
import tskit
from htcluster.api import BaseModel
from more_itertools import zip_equal
from pydantic import (
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    field_validator,
    model_validator,
)
from pydantic_core import from_json
from typing_extensions import Annotated

type OptionalNumber = float | int | None


# TODO: what about multiple ancestry seeds?
# prints simulation logging, doesn't match the rest of the logs, but fine for now
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)


def validate_demes(val):
    """Demes deserialization function"""
    match val:
        case str():  # parsing from json
            return demes.Graph.fromdict(from_json(val))
        case dict():  # from python
            return demes.Graph.fromdict(val)
        case _:
            return val


DemesGraph = Annotated[
    demes.Graph,
    BeforeValidator(validate_demes),
    PlainSerializer(lambda g: g.asdict(), return_type=dict),
]


class AncestryParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    samples: int  # TODO: SampleSet objects
    demography: DemesGraph | None = None
    sequence_length: OptionalNumber = None
    discrete_genome: bool | None = None
    recombination_rate: float | None = None  # TODO: RateMap objects
    population_size: OptionalNumber = None
    ploidy: int | None = None
    start_time: OptionalNumber = None
    end_time: OptionalNumber = None
    record_migrations: bool | None = None
    record_full_arg: bool | None = None
    coalescing_segments_only: bool | None = None
    num_labels: int | None = None
    random_seed: int | None = None
    num_replicates: int | None = None
    replicate_index: int | None = None
    record_provenance: bool | None = None

    @field_validator("random_seed")
    @classmethod
    def validate_num_replicates(cls, v: int | None) -> int:
        if v is None:
            return 1  # we always want to return an iterator
        else:
            assert v > 0, f"num replicates must be > 0, got: {v}"
        return v


class MutationParams(BaseModel):
    rate: int | float
    random_seeds: list[int] | None = None
    model: str | None = None  # TODO: model obj
    start_time: OptionalNumber = None
    end_time: OptionalNumber = None
    discrete_genome: bool | None = None
    keep: bool | None = None
    record_provenance: bool = True


class SimulationParams(BaseModel):
    ancestry_params: AncestryParams
    mutation_params: MutationParams | None

    @model_validator(mode="after")
    def validate_seeds(self) -> Self:
        n_reps = self.ancestry_params.num_replicates
        assert n_reps is not None  # mypy
        if (
            self.mutation_params is not None
            and (mut_seeds := self.mutation_params.random_seeds) is not None
        ):
            if n_reps > 1:
                msg = (
                    "must specify as many mutation seeds as ancestry replicates "
                    "or one ancestry rep and many mutation seeds, got "
                    f"{len(mut_seeds)} mut seeds and {n_reps} ancestry replicates"
                )
                assert isinstance(mut_seeds, list), msg
                assert n_reps == len(mut_seeds), msg
            assert len(mut_seeds) >= n_reps, (
                "cannot specify more mut seeds than ancestry reps, got "
                f"{len(mut_seeds)} mut seeds and {n_reps} ancestry replicates"
            )
        return self


def sim_ancestry(params: AncestryParams) -> Iterator[tskit.TreeSequence]:
    demography = (
        None
        if params.demography is None
        else msprime.Demography.from_demes(params.demography)
    )
    return msprime.sim_ancestry(
        samples=params.samples,
        demography=demography,
        sequence_length=params.sequence_length,
        discrete_genome=params.discrete_genome,
        recombination_rate=params.recombination_rate,
        population_size=params.population_size,
        ploidy=params.ploidy,
        start_time=params.start_time,
        end_time=params.end_time,
        record_migrations=params.record_migrations,
        record_full_arg=params.record_full_arg,
        coalescing_segments_only=params.coalescing_segments_only,
        num_labels=params.num_labels,
        random_seed=params.random_seed,
        num_replicates=params.num_replicates,
        replicate_index=params.replicate_index,
        record_provenance=params.record_provenance,
    )


def run_msprime(
    params: SimulationParams,
) -> Iterator[tuple[int, int, tskit.TreeSequence]]:
    mut_params = params.mutation_params
    anc_params = params.ancestry_params
    tss = sim_ancestry(anc_params)
    assert anc_params.num_replicates is not None  # mypy
    if mut_params is not None:
        if mut_params.random_seeds is None:
            rep_seeds = [[None]] * anc_params.num_replicates
        elif anc_params.num_replicates == 1:
            rep_seeds = [mut_params.random_seeds]
        else:
            rep_seeds = [[s] for s in mut_params.random_seeds]
        for i, (ts, seeds) in enumerate(zip_equal(tss, rep_seeds)):
            for j, seed in enumerate(seeds):
                yield (
                    i,
                    j,
                    msprime.sim_mutations(
                        ts,
                        rate=mut_params.rate,
                        random_seed=seed,
                        model=mut_params.model,
                        start_time=mut_params.start_time,
                        end_time=mut_params.end_time,
                        discrete_genome=mut_params.discrete_genome,
                        keep=mut_params.keep,
                        record_provenance=mut_params.record_provenance,
                    ),
                )
    else:
        return ((i, None, ts) for i, ts in enumerate(tss))
