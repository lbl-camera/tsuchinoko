"""Headless CLI for Tsuchinoko adaptive experiment service."""

import click
from loguru import logger


@click.group()
@click.version_option()
def main():
    """Tsuchinoko — adaptive experiment service."""
    pass


@main.command()
@click.option("--nats-url", default="", help="NATS broker URL (e.g. nats://localhost:4222). Empty = no NATS.")
@click.option("--lucid-prefix", default="als.7011", help="LUCID instance topic prefix.")
@click.option("--tiled-url", default="", help="Tiled server URL. Empty = no Tiled publication.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="YAML config file.")
def run(nats_url, lucid_prefix, tiled_url, config_path):
    """Run the Tsuchinoko adaptive experiment service."""
    from tsuchinoko.config import AppConfig, set_config
    from tsuchinoko.core import Core

    if config_path:
        import yaml
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        config = AppConfig(**(raw or {}))
    else:
        config = AppConfig()

    # CLI flags override config file
    if nats_url:
        config.nats.url = nats_url
    if lucid_prefix != "als.7011":
        config.nats.lucid_prefix = lucid_prefix
    if tiled_url:
        config.tiled.url = tiled_url

    set_config(config)

    # Create adaptive engine from config
    ac = config.adaptive
    if ac.engine_type == "gpcam":
        from tsuchinoko.adaptive.gpCAM_in_process import GPCAMInProcessEngine
        adaptive_engine = GPCAMInProcessEngine(
            dimensionality=ac.dimensionality,
            parameter_bounds=ac.parameter_bounds,
        )
    elif ac.engine_type == "random":
        from tsuchinoko.adaptive.random_in_process import RandomInProcess
        adaptive_engine = RandomInProcess(
            dimensionality=ac.dimensionality,
            parameter_bounds=ac.parameter_bounds,
        )
    else:
        raise click.BadParameter(f"Unknown engine type: {ac.engine_type}")

    logger.info(f"Tsuchinoko starting (engine={ac.engine_type}, dim={ac.dimensionality})")
    logger.info(f"  NATS: {config.nats.url or 'disabled'}")
    if config.nats.url:
        logger.info(f"  LUCID prefix: {config.nats.lucid_prefix}")
    logger.info(f"  Tiled: {config.tiled.url or 'disabled'}")

    core = Core(
        adaptive_engine=adaptive_engine,
        nats_config=config.nats,
    )
    try:
        core.main()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


@main.command()
def version():
    """Print version information."""
    from tsuchinoko import __version__
    click.echo(f"tsuchinoko {__version__}")
