from __future__ import annotations

from multiprocessing import freeze_support

import typer

from gospel.scripts.aces import Method, cli

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    walltime: int | None = None,
    memory: int | None = None,
    cores: int | None = None,
    port: int | None = None,
    scale: int | None = None,
) -> None:
    for method in Method:
        cli(
            method=method,
            ncircuits=50000,
            shots=200000,
            target=f"aces-{method}.png",
            walltime=walltime,
            memory=memory,
            cores=cores,
            port=port,
            scale=scale,
        )


if __name__ == "__main__":
    freeze_support()
    app()
