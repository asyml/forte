"""
Script to facilitate generating ontology given a root JSON config,
and cleaning a folder out of generated ontologies.
"""
import os
import argparse
import logging
from forte.data.ontology.ontology_code_generator import OntologyCodeGenerator

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def create(args_):
    """
    Function for the `create` mode. Generates the ontology.
    Args:
        args_: parsed args for the `create` mode
    """
    generator = OntologyCodeGenerator(args_.config_paths,
                                      args_.top_dir)
    if args_.no_dry_run is None:
        log.info("Ontology will be generated in a temporary directory as "
                 "--no_dry_run is not specified by the user.")
        args_.no_dry_run = False

    is_dry_run = not args_.no_dry_run
    generated_folder = generator.generate_ontology(args_.config,
                                                   args_.dest_path,
                                                   is_dry_run)
    log.info(f"Ontology generated in the directory {generated_folder}.")


def clean(args_):
    """
        Function for the `clean` mode. Cleans the given directory of generated
        files.
        Args:
            args_: parsed args for the `clean` mode
        """
    generator = OntologyCodeGenerator()
    is_empty = generator.cleanup_generated_ontology(args_.dir, args_.force)
    if not is_empty:
        log.info(f"Directory {args_.dir} not empty, cannot delete completely")


create_example = "python generate_ontology.py create --config " \
                 "forte/data/ontology/configs/example_ontology_config.json"
clean_example = "python generate_ontology.py clean --dir generated-files"

file_description = f"Utility to automatically generate or create ontologies.\n" \
                   f"\n*create*: Generate ontology given a root JSON config.\n" \
                   f"Example: {create_example} --no_dry_run\n\n" \
                   f"*clean*: Clean a folder of generated ontologies.\n" \
                   f"Example: {clean_example}"

parser = argparse.ArgumentParser(description=file_description,
                                 formatter_class=argparse.RawTextHelpFormatter)
subs = parser.add_subparsers()

# Parser for creating the ontology.
create_parser = subs.add_parser('create')
create_parser.add_argument('--config',
                           type=str,
                           required=True,
                           help='Root JSON config.')

create_parser.add_argument('--no_dry_run',
                           required=False,
                           default=None,
                           action='store_true',
                           help='Generates the package tree in a temporary '
                                'directory if true, ignores the argument '
                                '`--dest_path`')

create_parser.add_argument('--dest_path',
                           type=str,
                           required=False,
                           default=None,
                           help='Destination directory provided by the user. '
                                'Only used when --no_dry_run is specified. The '
                                'default directory is the current working '
                                'directory.')

create_parser.add_argument('--top_dir',
                           type=str,
                           required=False,
                           default=None,
                           help='Top level directory to be created inside the '
                                'generated directory.')

create_parser.add_argument('--config_paths',
                           type=str,
                           nargs='*',
                           required=False,
                           default=None,
                           help='Paths in which the root and imported config '
                                'files are to be searched.')

create_parser.set_defaults(func=create)

# Parsing for cleaning.
clean_parser = subs.add_parser('clean')

clean_parser.add_argument('--force', '-f', default=True, action='store_true')

clean_parser.add_argument('--dir',
                          type=str,
                          required=True,
                          help='Generated files to be cleaned from the '
                               'directory path.')

clean_parser.set_defaults(func=clean)

args = parser.parse_args()
args.func(args)
