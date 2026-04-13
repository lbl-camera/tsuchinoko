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
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="YAML config file.")
def run(nats_url, lucid_prefix, config_path):
    """Run the Tsuchinoko adaptive experiment service."""
    from tsuchinoko.config import AppConfig, set_config
    from tsuchinoko.nats.config import NATSConfig
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

    set_config(config)

    logger.info(f"Tsuchinoko starting (NATS: {'enabled' if config.nats.url else 'disabled'})")
    if config.nats.url:
        logger.info(f"  NATS URL: {config.nats.url}")
        logger.info(f"  LUCID prefix: {config.nats.lucid_prefix}")

    core = Core(nats_config=config.nats)
    try:
        core.main()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


@main.command()
def version():
    """Print version information."""
    from tsuchinoko import __version__
    click.echo(f"tsuchinoko {__version__}")
