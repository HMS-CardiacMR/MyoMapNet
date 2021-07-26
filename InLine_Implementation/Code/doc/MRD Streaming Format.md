# MR Data Message Format
The streaming ISMRM data format (MRD) consists of a series of messages sent asynchronously through a standard TCP/IP socket. Each message starts with an ID that identifies the message type and how subsequent stream data should be parsed. The term “server” refers to the process doing the image reconstruction or processing, while “client” refers to the process sending raw data or images.

<style>
.basic-styling td,
.basic-styling th {
  border: 1px solid #999;
  padding: 0.5rem;
}
.collapsed table {
  border-collapse: collapse;
 text-align:       center
}
</style>

## ID 1: MRD_MESSAGE_CONFIG_FILE

<div class="basic-styling collapsed">

| ID             | Config File Name |
| --             | --               |
| 2 bytes        | 1024 bytes       |
| unsigned short | char             |

This message type is used to send the file name of a configuration file (local on the server file system) to be used during reconstruction. The file name must not exceed 1023 characters and is formatted as a null terminated, UTF-8 encoded char string.

## ID 2: MRD_MESSAGE_CONFIG_TEXT

| ID             | Length   | Config Text     |
| --             | --       | --              |
| 2 bytes        | 4 bytes  | length * 1 byte |
| unsigned short | uint32_t | char            |

Alternatively, the text contents of a configuration file can be sent directly via the data stream. The length is sent as an uint32_t. Configuration text is sent as a null terminated char string.

## ID 3: MRD_MESSAGE_PARAMETER_HEADER
| ID             | Length   | XML Header Text |
| --             | --       | --              |
| 2 bytes        | 4 bytes  | length * 1 byte |
| unsigned short | uint32_t | char            |

Metadata for MRD datasets are stored in a flexible XML scheme, as detailed in http://ismrmrd.github.io/#flexible-data-header. The header length is sent as an uint32_t and the text is sent as a null terminated char string.

## ID 4: MRD_MESSAGE_CLOSE
| ID             |
| --             |
| 2 bytes        |
| unsigned short |

This message type consists only of an ID with no following data. It is used to indicate that all data related to an acquisition/reconstruction has been sent. IceFire will send this message after receiving a scan marked MDH_ACQEND. After receiving this message from the server, IceFire no longer accepts new data and finishes after completing processing of any already-received data.

## ID 5: MRD_MESSAGE_TEXT
| ID             | Length   | Text            |
| --             | --       | --              |
| 2 bytes        | 4 bytes  | length * 1 byte |
| unsigned short | uint32_t | char            |

Informational text can be sent using this message type, typically from the reconstruction side to the acquisition/client side. The length of message text is sent as an uint32_t while the text is sent as a null terminated char string. For the FIRE implementation of this message type, messages of this type are added to logviewer and the first three characters control how text is flagged.
- DBG: Debug (normal) text (white)
- WRN: Warning text (yellow)
- ERR: Error text (red)

## ID 1008: MRD_MESSAGE_ISMRMRD_ACQUISITION
| ID             | Fixed Raw Data Header   | Trajectory       | Raw Data |
| --             | --                      | --               | --       |
| 2 bytes        | 340 bytes               | number_of_samples * trajectory_dimensions * 4 bytes | number_of_channels * number_of_samples * 8 bytes |
| unsigned short | mixed                   | float             | float |

This message type is used to send raw (k-space) acquisition data. A separate message is sent for each readout. A fixed data header contains metadata such as encoding counters, is defined in http://ismrmrd.github.io/#fixed-data-structures. Three fields of the data header must be parsed in order to read the rest of the message:
- **trajectory_dimensions**: defines the number of dimensions in the k-space trajectory data component. For 2D acquisitions (kx, ky), this is set to 2, while for 3D acquisitions (kx, ky, kz), this is set to 3. If set to 0, the trajectory component is omitted.
- **number_of_samples**: number of readout samples.
- **active_channels**: number of channels for which raw data is acquired.

Trajectory data is organized by looping through the dimensions first then the samples:
e.g. for 2D trajectory data: Sample 1 Sample 2 Sample 3 … Sample n kx ky kx ky kx ky kx ky
e.g. for 3D trajectory data: Sample 1 Sample 2 Sample 3 … Sample n kx ky kz kx ky kz kx ky kz kx ky kz
Raw data is organized by looping through real/imaginary data, samples, then channels: Channel 1 Channel 2 Channel n Sample 1 … Sample n Sample 1 … Sample n … Sample 1 … Sample n Re Im Re Im Re Im Re Im Re Im Re Im Re Im Re Im Re Im

## 1022: MRD_MESSAGE_ISMRMRD_IMAGE
| ID             | Fixed Image Header  | Attribute Length | Attribute Data  |Image Data |
| --             | --                  | --               | --              | --        |
| 2 bytes        | 198 bytes           | 8 bytes          | length * 1 byte | matrix_size[0] * matrix_size[1] * matrix_size[2] * channels * sizeof(data_type) |
| unsigned short | mixed               | uint_64          | char            | data_type |

Image data is sent using this message type. The fixed image header contains metadata including fields such as the image type (magnitude, phase, etc.) and indices such as slice and repetition number. It is defined by the ImageHeader struct as detailed in http://ismrmrd.github.io/#fixed-data-structures. Within this header, there are 3 fields that must be interpreted to parse the rest of the message:
- **matrix_size**: This 3 element array indicates the size of each dimension of the image data.
- **channels**: This value indicates the number of (receive) channels for which image data is sent
- **data_type**: This value is an MRD_DataTypes enum that indicates the type of data sent. The following types are supported:

| Value        | Name         | Type           | Size        |
| --           | --           | --             | --          |
| 1            | MRD_USHORT   | uint16_t       |     2 bytes |
| 2            | MRD_SHORT    | int16_t        |     2 bytes |
| 3            | MRD_UINT     | uint32_t       |     4 bytes |
| 4            | MRD_INT      | int32_t        |     4 bytes |
| 5            | MRD_FLOAT    | float          |     4 bytes |
| 6            | MRD_DOUBLE   | double         |     8 bytes |
| 7            | MRD_CXFLOAT  | complex float  | 2 * 4 bytes |
| 8            | MRD_CXDOUBLE | complex double | 2 * 8 bytes |

Attributes are used to declare additional image metadata that is not present in the fixed image header. In general, this data is sent as a char string (not null-terminated), with the length sent first as an uint_64 (not uint_32!). The IceFire implementation interprets attribute data as an ISMRMRD MetaContainer, as defined by https://ismrmrd.github.io/api/class_i_s_m_r_m_r_d_1_1_meta_container.html.

Image data is organized by looping through matrix_size[0], matrix_size[1], matrix_size[2], then channels. For example, 2D image data would be formatted as: Channel 1 … Channel n Y1 … Yn Y1 … Yn X1 … Xn X1 … Xn X1 … Xn X1 … Xn X1 … Xn X1 … Xn

## ID 1026: MRD_MESSAGE_ISMRMRD_WAVEFORM
<style>
.collapsed table {
  border-collapse: collapse;
 text-align:       center
}
</style>

<div class="ox-hugo-table collapsed basic-styling">

| ID             | Fixed Waveform Header | Waveform Data                        |
| --             | --                    | --                                   |
| 2 bytes        | 240 bytes             | channels * number of samples * bytes |
| unsigned short | mixed                 | uint32_t                             |
</div>

This message type is used to send arbitrary waveform data (e.g. physio signals, gradient waveforms, etc.). The fixed waveform data header is defined by the MRD_WaveformHeader and contains the following members:

| Member Name       | Description                                   | Type     | Size    |
| --                | --                                            | --       | --      |
| version           | Version number                                | uint16_t | 2 bytes |
| flags             | Bit field with flags                          | uint64_t | 8 bytes |
| measurement_uid   | Unique ID for this measurement                | uint32_t | 4 bytes |
| scan_counter      | Number of the acquisition after this waveform | uint32_t | 4 bytes |
| time_stamp        | Starting imestamp of this waveform            | uint32_t | 4 bytes |
| number_of_samples | Number of samples acquired                    | uint16_t | 2 bytes |
| channels          | Active channels                               | uint16_t | 2 bytes |
| sample_time_us    | Time between samples in microseconds          | float    | 4 bytes |
| waveform_id       | ID matching types specified in XML header     | uint16_t | 2 bytes |

The **channels** and **number_of_samples** members fields must be parsed in order to read the rest of the message. Waveform data is sent as an uint32_t array, ordered by looping through samples and then through channels: Channel 1 Channel 2 … Channel n w1 … wn w1 … wn w1 … wn


# ISMRM MetaContainer Format
Image metadata is transmitted using XML text conforming to the ISMRM MetaContainer format, which may look like:

    <ismrmrdMeta>
        <meta>
            <name>DataRole</name>
            <value>Image</value>
            <value>AVE</value>
            <value>NORM</value>
            <value>MAGIR</value>
        </meta>
        <meta>
            <name>ImageNumber</name>
            <value>1</value>
        </meta>
    </ismrmrdMeta>
A variable number of “meta” elements can be defined, each with a single name and one or more value sub-elements. The following table lists named meta elements interpreted by IceFire:

| Element Name      | Format       | Interpretation                                      |
| --                | --           | --                                                  |
| DataRole          | text array   | Characteristics of the image. <br><br> A value of “Quantitative” indicates that pixel values in the image are parametric and to be interpreted directly (e.g. T1 values, velocity, etc.). If this role is present, pixel values are not further modified in the ICE chain, e.g. by normalization. |
| ImageComment      | text array   | Remarks about the image. This array of values is stored in the DICOM ImageComment (0020,4000) field, delimited by “_” (underscores). |
| SeriesDescription | text array   | Brief characteristics of the image. This array of values is appended to the DICOM SeriesDescription (0008,103E) field, delimited by “_” (underscores). |
| ImageType         | text array   | Characteristics of the image. This array of values is appended to the DICOM ImageType (0008,0008) field starting in position 4, delimited by “\” (backslash). |
| RescaleIntercept  | double       | Intercept for image pixel values, used in conjunction with RescaleSlope. Pixel values are to be interpreted as: ***value = RescaleSlope\*pixelValue + RescaleIntercept***. This value is set in the DICOM RescaleIntercept (0028,1052) field. |
| RescaleSlope      | double       | Scaling factor for image pixel values, used in conjunction with RescaleIntercept. Pixel values are to be interpreted as: ***value = RescaleSlope\*pixelValue + RescaleIntercept***. This value is set in the DICOM RescaleSlope (0028,1053) field. |
| WindowCenter      | long         | The window center in the rendered image, used in conjunction with WindowWidth. If RescaleIntercept and RescaleSlope are defined, WindowCenter and WindowWidth are applied to rescaled values. This value is set in the DICOM WindowCenter (0028,1050) field. |
| WindowWidth       | long         | The window center in the rendered image, used in conjunction with WindowCenter. If RescaleIntercept and RescaleSlope are defined, WindowCenter and WindowWidth are applied to rescaled values. This value is set in the DICOM WindowWidth (0028,1051) field. |
| LUTFileName       | text         | Path to a color lookup table file to be used for this image. LUT files must be in Siemens .pal format and stored in C:\MedCom\config\MRI\ColorLUT. If a value is provided, the DICOM field PhotometricInterpretation (0028,0004) is set to “PALETTE COLOR” |
| EchoTime          | double       | Echo time of the image. This value is set in the DICOM EchoTime (0018,0081) field.
| InversionTime     | double       | Inversion time of the image. This value is set in the DICOM InversionTime (0018,0082) field.
| ROI               | double array | Region of interest polygon. For multiple ROIs, the MetaAttribute element names shall start with “ROI_”. These ROIs are stored in a format compatible with the Siemens syngo viewer. The first 6 values are meta attributes of the ROI:
|                   |              |   1. Red color (normalized to 1)
|                   |              |   2. Green color (normalized to 1)
|                   |              |   3. Blue color (normalized to 1)
|                   |              |   4. Line thickness (default is 1)
|                   |              |   5. Line style (0 = solid, 1 = dashed)
|                   |              |   6. Visibility (0 = false, 1 = true)
|                   |              | The remaining values are (row,col) coordinates for each ROI point, with values between 0 and the number of rows/columns. Data is organized as (point 1row, point 1col, point2row, point 2col, etc). The last point should be a duplicate of the first point if a closed ROI is desired.

</div>