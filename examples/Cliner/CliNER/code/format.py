######################################################################
#  CliNER - format.py                                                #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Script to convert among different data formats.          #
######################################################################


import argparse
import os
import sys
import tempfile

from examples.Cliner.CliNER.code.notes.note import Note

cliner_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp_dir = os.path.join(cliner_dir, 'data', 'tmp')


def create_filename(odir, bfile, extension):
    fname = os.path.basename(bfile) + extension
    return os.path.join(odir, fname)


def main():
    # Argument Parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-txt",
        dest="txt",
        help="The files that contain the training examples",
    )

    parser.add_argument("-annotations",
        dest="annotations",
        help="The files that contain the labels for the training examples",
    )

    parser.add_argument("-out",
        dest="out",
        default=None,
        help="Directory to output data",
    )

    parser.add_argument("-format",
        dest="format",
        help="Output format (%s)" % str(' or '.join(Note.supportedFormats())),
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Parse arguments
    txt = args.txt
    annotations = args.annotations
    out_file = args.out
    format = args.format

    # pylint: disable=pointless-statement
    # Ensure annotations are specified
    if not txt:
        print >> sys.stderr, '\n\tError: Must supply text file'
        print >> sys.stderr
        sys.exit()
    elif not os.path.exists(txt):
        print >> sys.stderr, '\n\tError: Given text file does not exist'
        print >> sys.stderr
        sys.exit()

    # Ensure annotations are specified
    extensions = Note.supportedFormatExtensions()
    if not annotations:
        print >> sys.stderr, '\n\tError: Must supply annotations'
        print >> sys.stderr
        sys.exit()
    elif not os.path.exists(txt):
        print >> sys.stderr, '\n\tError: Given annotation file does not exist'
        print >> sys.stderr
        sys.exit()
    elif os.path.splitext(annotations)[1][1:] not in extensions:
        print >> sys.stderr, '\n\tError: annotation must be a supported format'
        # pylint: disable=expression-not-assigned
        print >> sys.stderr, '\t\t(.%s)' % str(' or .'.join(extensions))
        # pylint: disable=expression-not-assigned
        print >> sys.stderr
        sys.exit()

    # Ensure output format is specified
    if (not format) or (format not in Note.supportedFormats()):
        print >> sys.stderr, '\n\tError: Must specify supported output format'
        # pylint: disable=expression-not-assigned
        print >> sys.stderr, '\t\t(%s)' % str(
            ' or '.join(Note.supportedFormats()))
        # pylint: disable=expression-not-assigned
        print >> sys.stderr
        sys.exit()

    # Automatically find the input file format
    in_extension = os.path.splitext(annotations)[1][1:]
    for f, ext in Note.dictOfFormatToExtensions().items():
        if ext == in_extension:
            in_format = f

    # Read input data into note object
    in_note = Note(in_format)
    in_note.read(txt, annotations)

    # Convert data to standard format
    internal_output = in_note.write_standard()

    os_handle, tmp_file = tempfile.mkstemp(dir=tmp_dir, suffix="format_temp")
    with open(tmp_file, 'w') as f:
        f.write(internal_output)
    os.close(os_handle)

    # print internal_output

    # Read internal standard data into new file with given output format
    out_note = Note(format)
    out_note.read_standard(txt, tmp_file)

    # Output data
    out = out_note.write()
    if out_file:
        with open(out_file, 'w') as out_f:
            out_f.write(out)
    else:
        sys.stdout.write(out)

    # Clean up
    os.remove(tmp_file)
    if out_file:
        out_f.close()


if __name__ == '__main__':
    main()
