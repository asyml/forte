# database.py creates a .db file for performing umls searches.
import atexit
import os
import sqlite3
import sys

from read_config import enabled_modules

features_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if features_dir not in sys.path:
    sys.path.append(features_dir)

# find where umls tables are located

enabled = enabled_modules()
umls_tables = enabled['UMLS']

# set to True when create_db() is succesful
success = False
db_path = None
conn = None

MRSTY_TABLE_FILE = None
MRCON_TABLE_FILE = None
MRREL_TABLE_FILE = None
LRABR_TABLE_FILE = None


# this ensure files are closed properly and umls.db is removed if not succesful
@atexit.register
def umls_db_cleanup():
    # pylint: disable=global-statement
    global success
    global conn
    global db_path
    global MRSTY_TABLE_FILE
    global MRCON_TABLE_FILE
    global MRREL_TABLE_FILE
    global LRABR_TABLE_FILE

    if conn is not None:
        conn.close()

    if MRSTY_TABLE_FILE is not None:
        MRSTY_TABLE_FILE.close()

    if MRCON_TABLE_FILE is not None:
        MRCON_TABLE_FILE.close()

    if MRREL_TABLE_FILE is not None:
        MRREL_TABLE_FILE.close()

    if LRABR_TABLE_FILE is not None:
        LRABR_TABLE_FILE.close()

    if success is False:

        # remove umls.db, it is junk now
        if db_path is not None:
            os.remove(db_path)


def create_db():
    # pylint: disable=global-statement
    global success
    global conn
    global db_path

    global MRSTY_TABLE_FILE
    global MRCON_TABLE_FILE
    global MRREL_TABLE_FILE
    global LRABR_TABLE_FILE

    print("\ncreating umls.db")
    # connect to the .db file we are creating.
    db_path = os.path.join(umls_tables, 'umls.db')
    conn = sqlite3.connect(db_path)
    conn.text_factory = str

    print("opening files")
    # load data in files.
    try:
        mrsty_path = os.path.join(umls_tables, 'MRSTY.RRF')
        MRSTY_TABLE_FILE = open(mrsty_path, "r")
    except IOError:
        print("\nNo file to use for creating MRSTY.RRF table\n")
        sys.exit()

    try:
        mrcon_path = os.path.join(umls_tables, 'MRCONSO.RRF')
        MRCON_TABLE_FILE = open(mrcon_path, "r")
    except IOError:
        print("\nNo file to use for creating MRCONSO.RRF table\n")
        sys.exit()

    try:
        mrrel_path = os.path.join(umls_tables, 'MRREL.RRF')
        MRREL_TABLE_FILE = open(mrrel_path, "r")
    except IOError:
        print("\nNo file to use for creating MRREL.RRF table\n")
        sys.exit()

    try:
        lrabr_path = os.path.join(umls_tables, 'LRABR')
        LRABR_TABLE_FILE = open(lrabr_path, "r")
    except IOError:
        print("\nNo file to use for creating LRABR table\n")
        sys.exit()

    print("creating tables")
    c = conn.cursor()

    # create tables.
    c.execute("CREATE TABLE MRSTY( CUI, TUI, STN, STY, ATUI, CVF  ) ;")
    c.execute(
        "CREATE TABLE MRCON( CUI, LAT, TS, LUI, STT, SUI, ISPREF, AUI, SAUI, \
        SCUI, SDUI, SAB, TTY, CODE, STR, SRL, SUPPRESS, CVF ) ;")
    c.execute(
        "CREATE TABLE MRREL( CUI1, AUI1, STYPE1, REL, CUI2, AUI2, STYPE2, \
        RELA, RUI, SRUI, SAB, SL, RG, DIR, SUPPRESS, CVF );")
    c.execute("CREATE TABLE LRABR( EUI1, ABR, TYPE, EUI2, STR);")

    print("inserting data into MRSTY table")
    for line in MRSTY_TABLE_FILE:
        line = line.strip('\n')

        line = line.split('|')

        # end will always be empty str
        line.pop()

        assert len(line) == 6

        c.execute("INSERT INTO MRSTY( CUI, TUI, STN, STY, ATUI, CVF ) \
        values( ?, ?, ?, ?, ?, ?)", tuple(line))

    print("inserting data into MRCON table")
    for line in MRCON_TABLE_FILE:
        line = line.strip('\n')

        line = line.split('|')

        # end will always be empty str
        line.pop()

        assert len(line) == 18

        c.execute(
            "INSERT INTO MRCON( CUI, LAT, TS, LUI, STT, SUI, ISPREF, AUI, \
            SAUI, SCUI, SDUI, SAB, TTY, CODE, STR, SRL, SUPPRESS, CVF ) \
            values ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
            tuple(line))

    print("inserting data into MRREL table")
    for line in MRREL_TABLE_FILE:
        line = line.strip('\n')

        line = line.split('|')

        # end will always be empty str
        line.pop()

        assert len(line) == 16

        c.execute(
            "INSERT INTO MRREL(  CUI1, AUI1, STYPE1, REL, CUI2, AUI2, STYPE2, \
            RELA, RUI, SRUI, SAB, SL, RG, DIR, SUPPRESS, CVF ) \
            values( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )",
            tuple(line))

    print("inserting into LRABR table")
    for line in LRABR_TABLE_FILE:
        line = line.strip('\n')

        line = line.split('|')

        line.pop()

        assert len(line) == 5

        c.execute("INSERT INTO LRABR( EUI1, ABR, TYPE, EUI2, STR) \
        values( ?, ?, ?, ?,?)", tuple(line))

    print("creating indices")

    # create indices for faster queries
    c.execute("CREATE INDEX mrsty_cui_map ON MRSTY(CUI)")
    c.execute("CREATE INDEX mrcon_str_map ON MRCON(STR)")
    c.execute("CREATE INDEX mrcon_cui_map ON MRCON(CUI)")
    c.execute("CREATE INDEX mrrel_cui2_map ON MRREL( CUI2 )")
    c.execute("CREATE INDEX mrrel_cui1_map on MRREL( CUI1 ) ")
    c.execute("CREATE INDEX mrrel_rel_map on MRREL( REL )")
    c.execute("CREATE INDEX lrabr_abr_map on LRABR(ABR)")
    c.execute("CREATE INDEX lrabr_str_map on LRABR(STR)")

    # save changes to .db
    conn.commit()

    success = True
    print("\nsqlite database created")


if __name__ == "__main__":
    create_db()
