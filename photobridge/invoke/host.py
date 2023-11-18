"""
Module with docker commands
"""

import invoke


@invoke.task
def build_app_container(context):
    """
    Build app container

    :param context: invoke.Context instance
    """

    command = (
        "DOCKER_BUILDKIT=1 docker build "
        "--tag photobridge/multiperson:latest "
        "-f ./docker/app.Dockerfile ."
    )

    context.run(command, echo=True)


@invoke.task
def run(context, config_path):
    """
    Run app container

    Args:
        context (invoke.Context): invoke context instance
        config_path (str): path to configuration file
    """

    import os

    import photobridge.host.utilities

    config = photobridge.host.utilities.read_yaml(config_path)

    os.makedirs(config.data_dir_on_host, exist_ok=True)
    os.makedirs(config.logging_output_directory_on_host, exist_ok=True)

    # Define run options that need a bit of computations
    run_options = {
        # Use gpu runtime if host has cuda installed
        "gpu_capabilities": "--gpus all" if "/cuda/" in os.environ["PATH"] else ""
    }

    command = (
        "docker run -it --rm "
        "{gpu_capabilities} "
        "-v $PWD:/app "
        f"-v {config.logging_output_directory_on_host}:{os.path.dirname(config.logging_path)} "
        f"-v {os.path.abspath(config['data_dir_on_host'])}:{config['data_dir']} "
        "photobridge/multiperson:latest /bin/bash"
    ).format(**run_options)

    context.run(command, pty=True, echo=True)
