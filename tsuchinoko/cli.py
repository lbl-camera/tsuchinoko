"""Headless CLI for Tsuchinoko adaptive experiment service."""

import click
from loguru import logger


@click.group()
@click.version_option()
def main():
    """Tsuchinoko — adaptive experiment service."""
    pass


@main.command()
@click.argument('config_path', required=False, default=None)
def run(config_path):
    """Run the adaptive experiment service (placeholder for Phase 2)."""
    logger.info("Tsuchinoko headless service")
    if config_path:
        logger.info(f"Config: {config_path}")
    logger.info("NATS integration not yet implemented (Phase 2)")


@main.command()
def version():
    """Print version information."""
    from tsuchinoko import __version__
    click.echo(f"tsuchinoko {__version__}")
