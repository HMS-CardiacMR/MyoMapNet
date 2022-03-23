#!/usr/bin/python3

from server import Server

import argparse
import logging
import sys
import os

defaults = {
    'host':           '0.0.0.0',
    'port':           9004,
    'savedataFolder': '/tmp/share/saved_data'
}

def main(args):
    # Start a multi-threaded dispatcher to handle incoming connections
    server = Server(args.host, args.port, args.savedata, args.savedataFolder, args.multiprocessing)
    server.serve()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example server for MRD streaming format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--port',            type=int,            help='Port')
    parser.add_argument('-H', '--host',            type=str,            help='Host')
    parser.add_argument('-v', '--verbose',         action='store_true', help='Verbose output.')
    parser.add_argument('-l', '--logfile',         type=str,            help='Path to log file')
    parser.add_argument('-s', '--savedata',        action='store_true', help='Save incoming data')
    parser.add_argument('-S', '--savedataFolder',  type=str,            help='Folder to save incoming data')
    parser.add_argument('-m', '--multiprocessing', action='store_true', help='Use multiprocessing')

    parser.set_defaults(**defaults)

    args = parser.parse_args()

    if args.logfile:
        print("Logging to file: ", args.logfile)

        if not os.path.exists(os.path.dirname(args.logfile)):
            os.makedirs(os.path.dirname(args.logfile))

        logging.basicConfig(filename=args.logfile, format='%(asctime)s - %(message)s', level=logging.WARNING)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        print("No logfile provided")
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    main(args)
