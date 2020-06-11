"""
Script to facilitate generating ontology given a root JSON config,
and cleaning a folder out of generated ontology classes.
"""
import os
import sys
import logging
import argparse
from argparse import RawTextHelpFormatter
from forte.data.ontology.ontology_code_generator import OntologyCodeGenerator

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def normalize_path(path):
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))


def create(args_):
    """
    Function for the `create` mode. Generates the ontology.
    Args:
        args_: parsed args for the `create` mode
    """
    spec_path = normalize_path(args_.spec)
    dest_path = normalize_path(args_.dest_path)
    spec_paths = [normalize_path(config) for config in args_.spec_paths] \
        if args_.spec_paths is not None else None

    merged_path = normalize_path(args_.merged_path)

    generator = OntologyCodeGenerator(spec_paths, args_.gen_all)
    if args_.no_dry_run is None:
        log.info("Ontology will be generated in a temporary directory as "
                 "--no_dry_run is not specified by the user.")
        args_.no_dry_run = False

    is_dry_run = not args_.no_dry_run
    include_init = not args_.exclude_init
    generated_folder = generator.generate(spec_path, dest_path, is_dry_run,
                                          include_init, merged_path)
    log.info("Ontology generated in the directory %s.", generated_folder)


def clean(args_):
    """
        Function for the `clean` mode. Cleans the given directory of generated
        files.
        Args:
            args_: parsed args for the `clean` mode
        """
    dir_ = normalize_path(args_.dir)
    generator = OntologyCodeGenerator()
    is_empty, del_dir = generator.cleanup_generated_ontology(dir_, args_.force)
    if not is_empty:
        log.info("Directory %s not empty, cannot delete completely.", dir_)
    else:
        log.info("Directory %s deleted.", dir_)
    if not args_.force:
        log.info("Deleted files moved to %s.", del_dir)


class OntologyGenerationParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def main():
    create_example = "generate_ontology create --spec " \
                     "ontology_specs/example/pet_shop.json"
    clean_example = "generate_ontology clean --dir generated-files"

    file_description = '\n'.join([
        "Utility to automatically generate or create Python classes given "
        "the ontology.",
        "\n*create*: Generate ontology given a JSON specification.",
        f"Example: {create_example} --no_dry_run\n",
        "*clean*: Clean a folder of generated Python classes.",
        f"Example: {clean_example}"])

    parser = OntologyGenerationParser(description=file_description,
                                      formatter_class=RawTextHelpFormatter)
    subs = parser.add_subparsers(title='required commands',
                                 help='Select one of:')

    # Parser for creating the ontology.
    create_parser = subs.add_parser('create', help='to create new ontology')
    create_parser.add_argument('-i', '--spec',
                               type=str,
                               required=True,
                               help='The main input JSON specification.')

    create_parser.add_argument('-r', '--no_dry_run',
                               required=False,
                               default=None,
                               action='store_true',
                               help='Generates the package tree in a temporary '
                                    'directory if true, ignores the argument '
                                    '`--dest_path`')

    create_parser.add_argument('-o', '--dest_path',
                               type=str,
                               required=False,
                               default=os.getcwd(),
                               help='Destination directory provided by the user'
                                    '. Only used when --no_dry_run is '
                                    'specified. The'
                                    ' default directory is the current working '
                                    'directory.')

    create_parser.add_argument('-s', '--spec_paths',
                               type=str,
                               nargs='*',
                               required=False,
                               default=None,
                               help='Paths in which the root and imported '
                                    'spec files are to be searched.')

    create_parser.add_argument('-m', '--merged_path',
                               type=str,
                               required=False,
                               default=None,
                               help='The destination file path for the merged'
                                    'file path.')

    create_parser.add_argument('-e', '--exclude_init',
                               required=False,
                               default=None,
                               action='store_true',
                               help='Excludes generation of `__init__.py` files'
                                    ' in the already existing directories, if'
                                    '`__init__.py` not already present.')

    create_parser.add_argument('-a', '--gen_all',
                               required=False,
                               default=False,
                               action='store_true',
                               help='If True, will generate all the ontology,'
                                    'including the existing ones shipped with '
                                    'Forte.')

    create_parser.set_defaults(func=create)

    # Parsing for cleaning.
    clean_parser = subs.add_parser('clean', help='to clean up ontologies')

    clean_parser.add_argument('-d', '--dir',
                              type=str,
                              required=True,
                              help='Generated files to be cleaned from the '
                                   'directory path.')

    clean_parser.add_argument('-f', '--force',
                              default=False,
                              action='store_true',
                              help='If true, skips the interactive deleting of '
                                   'folders. Use with caution.')

    clean_parser.set_defaults(func=clean)

    options = parser.parse_args()
    options_func = getattr(options, "func", None)
    if options_func is not None:
        options.func(options)
    else:
        sys.stderr.write('Error: %s\n' % "wrong usage of the script.")
        OntologyGenerationParser().print_help()
        sys.exit(2)

    if __name__ == "__main__":
        main()
