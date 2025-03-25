from __future__ import annotations

import dask.distributed
from dask_jobqueue import SLURMCluster  # type: ignore[attr-defined]


def get_cluster(
    walltime: int | None = None,
    memory: int | None = None,
    cores: int | None = None,
    port: int | None = None,
    scale: int | None = None,
) -> dask.distributed.deploy.cluster.Cluster:
    if walltime is None and memory is None and cores is None and port is None:
        cluster: dask.distributed.deploy.cluster.Cluster = (
            dask.distributed.LocalCluster()  # type: ignore[no-untyped-call]
        )
    else:
        if walltime is None:
            raise ValueError("--walltime <hours> is required for running on cleps")
        if memory is None:
            raise ValueError("--memory <GB> is required for running on cleps")
        if cores is None:
            raise ValueError("--cores <N> is required for running on cleps")
        if port is None:
            raise ValueError("--port <N> is required for running on cleps")
        if scale is None:
            raise ValueError("--scale <N> is required for running on cleps")
        cluster = SLURMCluster(
            account="inria",
            queue="cpu_devel",
            cores=cores,
            memory=f"{memory}GB",
            walltime=f"{walltime}:00:00",
            scheduler_options={"dashboard_address": f":{port}"},
        )
    if scale is not None:
        cluster.scale(scale)
    return cluster
