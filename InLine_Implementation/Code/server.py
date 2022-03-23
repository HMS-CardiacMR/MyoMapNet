
import constants
from connection import Connection

import socket
import logging
import multiprocessing
import ismrmrd.xsd
import importlib

import simplefft
import invertcontrast
import analyzeflow
import t1mapping
import myomapnet

class Server:
    """
    Something something docstring.
    """

    def __init__(self, address, port, savedata, savedataFolder, multiprocessing):
        logging.info("Starting server and listening for data at %s:%d", address, port)
        if (savedata is True):
            logging.debug("Saving incoming data is enabled.")

        if (multiprocessing is True):
            logging.debug("Multiprocessing is enabled.")

        self.multiprocessing = multiprocessing
        self.savedata = savedata
        self.savedataFolder = savedataFolder
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((address, port))

    def serve(self):
        logging.debug("Serving... ")
        self.socket.listen(0)

        while True:
            sock, (remote_addr, remote_port) = self.socket.accept()

            logging.info("Accepting connection from: %s:%d", remote_addr, remote_port)

            if (self.multiprocessing is True):
                process = multiprocessing.Process(target=self.handle, args=[sock])
                process.daemon = True
                process.start()
                logging.debug("Spawned process %d to handle connection.", process.pid)
            else:
                self.handle(sock)

    def handle(self, sock):

        try:
            connection = Connection(sock, self.savedata, "", self.savedataFolder, "dataset")

            # First message is the config (file or text)
            config = next(connection)

            # Break out if a connection was established but no data was received
            if ((config is None) & (connection.is_exhausted is True)):
                logging.info("Connection closed without any data received")
                return

            # Second messages is the metadata (text)
            metadata_xml = next(connection)
            logging.debug("XML Metadata: %s", metadata_xml)
            try:
                metadata = ismrmrd.xsd.CreateFromDocument(metadata_xml)
                if (metadata.acquisitionSystemInformation.systemFieldStrength_T != None):
                    logging.info("Data is from a %s %s at %1.1fT", metadata.acquisitionSystemInformation.systemVendor, metadata.acquisitionSystemInformation.systemModel, metadata.acquisitionSystemInformation.systemFieldStrength_T)
            except:
                logging.warning("Metadata is not a valid MRD XML structure.  Passing on metadata as text")
                metadata = metadata_xml

            # Decide what program to use based on config
            # If not one of these explicit cases, try to load file matching name of config
            if (config == "simplefft"):
                logging.info("Starting simplefft processing based on config")
                simplefft.process(connection, config, metadata)
            elif (config == "invertcontrast"):
                logging.info("Starting invertcontrast processing based on config")
                invertcontrast.process(connection, config, metadata)
            elif (config == "analyzeflow"):
                logging.info("Starting analyzeflow processing based on config")
                analyzeflow.process(connection, config, metadata)
            elif (config == "t1mapping"):
                logging.info("Starting t1mapping processing based on config")
                t1mapping.process(connection, config, metadata)
            elif (config == "myomapnet"):
                logging.info("Starting myomapnet processing based on config")
                myomapnet.process(connection, config, metadata)
            elif (config == "null"):
                logging.info("No processing based on config")
                try:
                    for msg in connection:
                        if msg is None:
                            break
                finally:
                    connection.send_close()
            elif (config == "savedataonly"):
                # Dummy loop with no processing
                try:
                    for msg in connection:
                        if msg is None:
                            break
                finally:
                    connection.send_close()
            else:
                try:
                    # Load module from file having exact name as config
                    module = importlib.import_module(config)
                    logging.info("Starting config %s", config)
                    module.process(connection, config, metadata)
                except ImportError:
                    logging.info("Unknown config '%s'.  Falling back to 'invertcontrast'", config)
                    invertcontrast.process(connection, config, metadata)

        except Exception as e:
            logging.exception(e)

        finally:
            # Encapsulate shutdown in a try block because the socket may have
            # already been closed on the other side
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            sock.close()
            logging.info("Socket closed")

            # Dataset may not be closed properly if a close message is not received
            if connection.savedata is True:
                try:
                    connection.dset.close()
                except:
                    pass
                logging.info("Incoming data was saved at %s", connection.mrdFilePath)
