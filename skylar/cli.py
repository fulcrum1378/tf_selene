import click

from skylar import __version__
from skylar.utils import load_path, parse_configs_and_run


@click.command()
@click.version_option(__version__)
@click.argument('path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--lr', type=float, help='If training, the optimizer learning rate', show_default=True)
def main(path, lr):
    configs = load_path(path, instantiate=False)
    parse_configs_and_run(configs, lr=lr)


if __name__ == "__main__":
    main()
