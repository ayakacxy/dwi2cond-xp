"""Generate a CycloneDX source SBOM and SHA256SUMS for built distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
import uuid
from pathlib import Path


def _dependency_name(requirement: str) -> str:
    match = re.match(r"^[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise ValueError(f"Cannot parse dependency requirement: {requirement}")
    return match.group(0)


def write_sbom(project_root: Path, dist: Path) -> Path:
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    package_ref = f"pkg:pypi/{project['name']}@{project['version']}"
    components = []
    dependency_refs = []
    for requirement in project.get("dependencies", []):
        name = _dependency_name(requirement)
        reference = f"pkg:pypi/{name}"
        dependency_refs.append(reference)
        components.append(
            {
                "type": "library",
                "name": name,
                "bom-ref": reference,
                "properties": [
                    {"name": "dwi2cond-xp:declared-requirement", "value": requirement}
                ],
            }
        )
    serial_material = "\n".join([package_ref, *sorted(dependency_refs)])
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, serial_material)}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": project["name"],
                "version": project["version"],
                "bom-ref": package_ref,
                "licenses": [{"license": {"id": project["license"]}}],
                "purl": package_ref,
            }
        },
        "components": components,
        "dependencies": [{"ref": package_ref, "dependsOn": dependency_refs}],
    }
    output = dist / f"{project['name'].replace('-', '_')}.cdx.json"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return output


def write_checksums(dist: Path) -> Path:
    output = dist / "SHA256SUMS"
    candidates = sorted(
        path for path in dist.iterdir() if path.is_file() and path.name != output.name
    )
    lines = []
    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dist = args.dist.resolve()
    if not list(dist.glob("*.whl")) or not list(dist.glob("*.tar.gz")):
        raise SystemExit("Build a wheel and sdist before generating release metadata")
    sbom = write_sbom(project_root, dist)
    checksums = write_checksums(dist)
    print(f"Wrote {sbom.name} and {checksums.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
