"""Every job that reads a secret must run in the branch-restricted "secure" environment."""
from pathlib import Path

import yaml


def test_secret_jobs_use_secure_environment():
    for path in Path(".github/workflows").glob("*.yml"):
        for name, job in yaml.safe_load(path.read_text())["jobs"].items():
            if "secrets." in yaml.safe_dump(job):
                assert job.get("environment") == "secure", f"{path.name}:{name}"
