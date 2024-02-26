from jsonargparse import ArgumentParser, Namespace

import torchlft.scripts.train as train

parser = ArgumentParser(prog="lft")

subcommands = parser.add_subcommands()
subcommands.add_subcommand("train", train.parser)


def main(config: Namespace) -> None:

    if config.subcommand == "train":
        train.main(config.train)


def cli():
    config = parser.parse_args()
    main(config)


if __name__ == "__main__":
    cli()
