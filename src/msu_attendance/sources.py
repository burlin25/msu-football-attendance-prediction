"""Download public inputs and record the exact files used by the parser."""

import hashlib
import subprocess
import tempfile
from pathlib import Path


def fetch(url: str, destination: Path, *, refresh: bool = False) -> Path:
    if destination.is_file() and destination.stat().st_size and not refresh:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        # The official legacy athletics hostname contains an underscore. curl can
        # validate its wildcard certificate with normal HTTPS verification enabled.
        subprocess.run(
            ["curl", "--disable", "--fail", "--silent", "--show-error", "--location",
             "--proto", "=https", "--proto-redir", "=https", "--retry", "2",
             "--connect-timeout", "15", "--max-time", "60", "--output", str(temporary), url],
            check=True, capture_output=True, text=True,
        )
        if not temporary.stat().st_size:
            raise ValueError(f"Source returned an empty file for {url}")
        temporary.replace(destination)
        return destination
    except FileNotFoundError as error:
        raise RuntimeError("Downloading raw sources requires curl on PATH.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Could not download {url}: {error.stderr.strip()}") from error
    finally:
        temporary.unlink(missing_ok=True)


def describe(url: str, file: Path) -> dict:
    with file.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"url": url, "path": file.as_posix(), "bytes": file.stat().st_size, "sha256": digest}
