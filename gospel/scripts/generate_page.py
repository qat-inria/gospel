import string
import tarfile
from datetime import datetime
from pathlib import Path

import git

from gospel.sampling_circuits.experiments import run_sample_circuits
from gospel.scripts.qasm2brickwork_state import (
    convert_circuit_directory_to_brickwork_state,
)
from gospel.scripts.qasm2img import convert_circuit_directory_to_svg


def generate_page() -> None:
    path_pages_meta = Path("pages.meta")
    path_pages = Path("pages")

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    shortsha = sha[0:7]
    committed_date = datetime.fromtimestamp(repo.head.object.committed_date)

    circuits_dirname = f"circuits-{shortsha}"
    path_circuits = path_pages / circuits_dirname

    run_sample_circuits(path_circuits)

    circuits_tarball = f"{circuits_dirname}.tar.gz"
    with tarfile.open(path_pages / circuits_tarball, "w:gz") as tar:
        tar.add(path_circuits, arcname=circuits_dirname)

    circuits_svg_dirname = f"circuits-svg-{shortsha}"
    path_circuits_svg = path_pages / circuits_svg_dirname
    convert_circuit_directory_to_svg(path_circuits, path_circuits_svg)

    brickwork_state_svg_dirname = f"brickwork-state-svg-{shortsha}"
    path_brickwork_state_svg = path_pages / brickwork_state_svg_dirname
    convert_circuit_directory_to_brickwork_state(
        path_circuits, path_brickwork_state_svg
    )

    with (path_pages_meta / "index.html").open("r") as f:
        template = string.Template(f.read())

    result = template.substitute(
        {
            "sha": sha,
            "committed_date": committed_date.astimezone()
            .replace(microsecond=0)
            .isoformat(),
            "circuits_dirname": circuits_dirname,
            "circuits_tarball": circuits_tarball,
            "circuits_svg_dirname": circuits_svg_dirname,
            "brickwork_state_svg_dirname": brickwork_state_svg_dirname,
        }
    )

    with (path_pages / "index.html").open("w") as f:
        f.write(result)


if __name__ == "__main__":
    generate_page()
