
import struct

MRD_MESSAGE_INT_ID_MIN                             =    0 # CONTROL
MRD_MESSAGE_CONFIG_FILE                            =    1
MRD_MESSAGE_CONFIG_TEXT                            =    2
MRD_MESSAGE_METADATA_XML_TEXT                      =    3
MRD_MESSAGE_CLOSE                                  =    4
MRD_MESSAGE_TEXT                                   =    5
MRD_MESSAGE_INT_ID_MAX                             =  999 # CONTROL
MRD_MESSAGE_EXT_ID_MIN                             = 1000 # CONTROL
MRD_MESSAGE_ACQUISITION                            = 1001 # DEPRECATED
MRD_MESSAGE_NEW_MEASUREMENT                        = 1002 # DEPRECATED
MRD_MESSAGE_END_OF_SCAN                            = 1003 # DEPRECATED
MRD_MESSAGE_IMAGE_CPLX_FLOAT                       = 1004 # DEPRECATED
MRD_MESSAGE_IMAGE_REAL_FLOAT                       = 1005 # DEPRECATED
MRD_MESSAGE_IMAGE_REAL_USHORT                      = 1006 # DEPRECATED
MRD_MESSAGE_EMPTY                                  = 1007 # DEPRECATED
MRD_MESSAGE_ISMRMRD_ACQUISITION                    = 1008
MRD_MESSAGE_ISMRMRD_IMAGE_CPLX_FLOAT               = 1009 # DEPRECATED
MRD_MESSAGE_ISMRMRD_IMAGE_REAL_FLOAT               = 1010 # DEPRECATED
MRD_MESSAGE_ISMRMRD_IMAGE_REAL_USHORT              = 1011 # DEPRECATED
MRD_MESSAGE_DICOM                                  = 1012 # DEPRECATED
MRD_MESSAGE_CLOUD_JOB                              = 1013 # UNSUPPORTED
MRD_MESSAGE_GADGETCLOUD_JOB                        = 1014 # UNSUPPORTED
MRD_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_CPLX_FLOAT     = 1015 # DEPRECATED
MRD_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_REAL_FLOAT     = 1016 # DEPRECATED
MRD_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_REAL_USHORT    = 1017 # DEPRECATED
MRD_MESSAGE_DICOM_WITHNAME                         = 1018 # UNSUPPORTED
MRD_MESSAGE_DEPENDENCY_QUERY                       = 1019 # UNSUPPORTED
MRD_MESSAGE_ISMRMRD_IMAGE_REAL_SHORT               = 1020 # DEPRECATED
MRD_MESSAGE_ISMRMRD_IMAGEWITHATTRIB_REAL_SHORT     = 1021 # DEPRECATED
MRD_MESSAGE_ISMRMRD_IMAGE                          = 1022
MRD_MESSAGE_RECONDATA                              = 1023 # UNSUPPORTED
MRD_MESSAGE_ISMRMRD_WAVEFORM                       = 1026
MRD_MESSAGE_EXT_ID_MAX                             = 4096 # CONTROL

MrdMessageLength = struct.Struct('<I')
SIZEOF_MRD_MESSAGE_LENGTH = len(MrdMessageLength.pack(0))

MrdMessageIdentifier = struct.Struct('<H')
SIZEOF_MRD_MESSAGE_IDENTIFIER = len(MrdMessageIdentifier.pack(0))

MrdMessageConfigurationFile = struct.Struct('<1024s')
SIZEOF_MRD_MESSAGE_CONFIGURATION_FILE = len(MrdMessageConfigurationFile.pack(b''))

MrdMessageAttribLength = struct.Struct('<Q')
SIZEOF_MRD_MESSAGE_ATTRIB_LENGTH = len(MrdMessageAttribLength.pack(0))
