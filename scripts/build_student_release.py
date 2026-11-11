"""Build, check, or export the public student starter from an explicit manifest."""

import argparse
from pathlib import Path
from shutil import copyfile
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "student"
ARCHIVE = ROOT / "mujoco-racing-student.zip"
ARCHIVE_FILES = (
    "README.md", "environment.yml", "requirements.txt", "train.py", "evaluate.py",
    "racing/__init__.py", "racing/env.py", "racing/car.xml", "racing/track.py",
    "racing/reward.py", "racing/visuals.py", "racing/vehicle.py",
)
GIT_FILES = (".gitignore", *ARCHIVE_FILES)


def check_archive(path=ARCHIVE):
    with ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(ARCHIVE_FILES) or archive.testzip() is not None:
            raise ValueError(f"Unexpected file manifest or corrupt archive: {path}")
        for name in ARCHIVE_FILES:
            if archive.read(name) != (STUDENT / name).read_bytes():
                raise ValueError(f"Archive does not match student/{name}")


def build_archive():
    with NamedTemporaryFile(dir=ARCHIVE.parent, prefix=".student-release-", delete=False) as temp:
        path = Path(temp.name)
    try:
        with ZipFile(path, "w") as archive:
            for name in ARCHIVE_FILES:
                archive.writestr(ZipInfo(name), (STUDENT / name).read_bytes(), compress_type=ZIP_DEFLATED)
        check_archive(path)
        path.replace(ARCHIVE)
    finally:
        path.unlink(missing_ok=True)


def export_git(destination):
    for name in GIT_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(STUDENT / name, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Verify the existing student ZIP")
    modes.add_argument("--export", type=Path, metavar="DIRECTORY", help="Copy the GitHub starter files")
    args = parser.parse_args()
    if args.export is not None:
        export_git(args.export)
        print(f"Exported {len(GIT_FILES)} student files to {args.export}")
    elif args.check:
        check_archive()
        print(f"Verified {len(ARCHIVE_FILES)} student files in {ARCHIVE.name}")
    else:
        build_archive()
        print(f"Built {ARCHIVE.name} with {len(ARCHIVE_FILES)} student files")


if __name__ == "__main__":
    main()
