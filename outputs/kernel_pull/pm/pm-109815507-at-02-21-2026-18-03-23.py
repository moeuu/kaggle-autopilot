import subprocess
import sys
import os
import shutil

preamble_configs = """#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

export PIP_DISABLE_PIP_VERSION_CHECK=true
export PIP_FIND_LINKS=$SCRIPT_DIR
export PIP_NO_INDEX=true

"""

github_configs = """
export PIP_NO_BUILD_ISOLATION=false

"""

zip_file_installs = """
for file in "$SCRIPT_DIR"/*.zip; do
    if [ -f "$file" ]; then
        pip install "$file"
    fi
done
"""

def copy_requirements_file(src="/kaggle/requirements/input_requirements.txt", dst="/kaggle/working/input_requirements.txt"):
    shutil.copy(src, dst)
    print(f"Successfully copied requirements file to {dst}")


def download_pip_files(filename, target_dir="/kaggle/working"):
    with open(filename, "r") as f:
        for line in f:
            command = line.strip()

            if command.startswith("pip install"):
                if "-r" in command or "--requirement" in command:
                    raise ValueError(f"Requirements files (-r or --requirement) not supported: {command}")

                # Convert pip install commands to pip download commands.
                download_command = command.replace("install", "download")

                # By default, pip downloads fetches the latest version
                download_command = download_command.replace("-U", "").replace("--upgrade", "")
                download_command += f" -d {target_dir}"

                print(f"Downloading package to {target_dir}: {download_command}")

                try:
                    subprocess.check_call(download_command.split())
                except subprocess.CalledProcessError:
                    raise ValueError(f"Failed to download package with command: {command}")

            else:
                raise ValueError(f"Only install commands are supported: {command}")


def create_install_script(requirements_file, script_name="install_requirements.sh"):
    with open(script_name, "w") as f:
        f.write(preamble_configs)
        with open(requirements_file, "r") as req_file:
            for line in req_file:
                package = line.strip()
                # We install packages from github last
                if "git+https" not in package:
                    f.write(f"{package}\n")

        f.write(github_configs)
        f.write(zip_file_installs)


copy_requirements_file()
download_pip_files("/kaggle/requirements/input_requirements.txt")
create_install_script("/kaggle/requirements/input_requirements.txt", script_name="install_requirements.sh")