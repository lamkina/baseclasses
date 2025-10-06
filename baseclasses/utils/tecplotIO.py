import re
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum, EnumMeta
from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Optional, Set, TextIO, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound="TecplotZone")


# ==============================================================================
# ENUMS
# ==============================================================================
class ZoneType(Enum):
    """Tecplot finite element zone types"""

    UNSET = -1
    ORDERED = 0
    FELINESEG = 1
    FETRIANGLE = 2
    FEQUADRILATERAL = 3
    FETETRAHEDRON = 4
    FEBRICK = 5


FEZones = [ZoneType.FELINESEG, ZoneType.FETRIANGLE, ZoneType.FEQUADRILATERAL, ZoneType.FETETRAHEDRON, ZoneType.FEBRICK]


class DataPacking(Enum):
    """Tecplot data packing formats"""

    BLOCK = 0
    POINT = 1


class VariableLocation(Enum):
    """Grid location of the variable data"""

    NODAL = 0
    CELLCENTERED = 1


class DataPrecision(Enum):
    """Tecplot data precision"""

    SINGLE = 6
    DOUBLE = 12


class BinaryDataPrecisionCodes(Enum):
    """Binary data precision codes"""

    SINGLE = 1
    DOUBLE = 2


class DataType(Enum):
    """Data types for variable data"""

    SINGLE = np.float32
    DOUBLE = np.float64
    LONGINT = np.int32
    SHORTINT = np.int16
    BYTE = np.int8
    BIT = bool


class FileType(Enum):
    """Tecplot file types"""

    FULL = 0
    GRID = 1
    SOLUTION = 2


class SectionMarkers(Enum):
    """Tecplot section markers"""

    ZONE = 299.0  # V11.2 marker
    DATA = 357.0


class BinaryFlags(Enum):
    """Binary boolean flags"""

    NONE = -1
    FALSE = 0
    TRUE = 1


class StrandID(Enum):
    """Strand ID default codes"""

    PENDING = -2
    STATIC = -1


class SolutionTime(Enum):
    """Solution time default codes"""

    UNSET = -1


class Separator(Enum):
    """Separator characters"""

    SPACE = " "
    COMMA = ","
    TAB = "\t"
    NEWLINE = "\n"
    CARRIAGE_RETURN = "\r"


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
class TecplotZone:
    def __init__(
        self,
        title: str,
        data: npt.NDArray,
        variables: List[str],
        sharedVariables: List[int],
        passiveVariables: List[bool],
        zoneType: ZoneType = ZoneType.UNSET,
        connectivity: Optional[npt.NDArray] = None,
        connectivityShareZone: Optional[int] = None,
        dataType: DataType = DataType.DOUBLE,
        dataPacking: DataPacking = DataPacking.POINT,
        varLocation: VariableLocation = VariableLocation.NODAL,
        strandID: Optional[int] = None,
        solutionTime: Optional[Union[float, int]] = None,
        auxData: Optional[Dict[str, str]] = None,
    ):
        """Create a tecplot zone object."""
        if not isinstance(title, str):
            raise TypeError("Title must be a string.")

        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy ndarray.")

        if not isinstance(variables, list) or not all(isinstance(var, str) for var in variables):
            raise TypeError("Variables must be a list of strings.")

        if not isinstance(sharedVariables, list) or not all(isinstance(var, int) for var in sharedVariables):
            raise TypeError("Shared variables must be a list of integers.")

        if not isinstance(passiveVariables, list) or not all(isinstance(var, bool) for var in passiveVariables):
            raise TypeError("Passive variables must be a list of booleans.")

        if zoneType not in ZoneType:
            raise TypeError(f"Zone type must be one of {list(ZoneType)}.")

        if dataType not in DataType:
            raise TypeError(f"Data type must be one of {list(DataType)}.")

        if dataPacking not in DataPacking:
            raise TypeError(f"Data packing must be one of {list(DataPacking)}.")

        if varLocation not in VariableLocation:
            raise TypeError(f"Variable location must be one of {list(VariableLocation)}.")

        if strandID is not None:
            if not isinstance(strandID, int):
                raise TypeError("Strand ID must be an integer.")

        if solutionTime is not None:
            if not isinstance(solutionTime, (float, int)):
                raise TypeError("Solution time must be a float or int.")

        if connectivity is not None and zoneType not in FEZones:
            raise TypeError(f"Connectivty provided for non-FE zone type: {zoneType.name}.")

        # Connectivity must be either None or a 2D numpy array of integers
        if connectivity is not None:
            if not isinstance(connectivity, np.ndarray):
                raise TypeError("Connectivity must be a numpy ndarray.")
            if connectivity.ndim != 2 or connectivity.dtype.kind not in "iu":
                raise TypeError("Connectivity must be a 2D array of integers.")

        if connectivityShareZone is not None:
            if not isinstance(connectivityShareZone, int):
                raise TypeError("Connectivity share zone must be an integer.")

        if zoneType in FEZones:
            if connectivityShareZone is None and connectivity is None:
                raise ValueError("Either 'connectivityShareZone' or 'connectivity' must be provided for FE zones.")
            elif connectivityShareZone is not None and connectivity is not None:
                raise ValueError("If 'connectivityShareZone' is provided, 'connectivity' must be None.")

        if auxData is not None:
            if not isinstance(auxData, dict):
                raise TypeError("Auxiliary data must be a dictionary.")
            if not all(isinstance(key, str) and isinstance(value, str) for key, value in auxData.items()):
                raise TypeError("Auxiliary data keys and values must be strings.")

        self.title = title
        self.data = data
        self.zoneType = zoneType
        self.passiveVariables = passiveVariables

        # --- Validate and set connectivity ---
        self.connectivity = connectivity
        self.connectivityShareZone = connectivityShareZone
        self.uniqueIndices = None
        self.uniqueConnectivity = None
        if connectivity is not None:
            self._validateConnectivity()
            self.uniqueIndices = np.unique(connectivity.flatten())
            self.uniqueConnectivity = self._remapConnectivity()

        # --- Header attributes ---
        self.dataType = dataType
        self.dataPacking = dataPacking
        self.varLocation = varLocation
        self.strandID = strandID
        self.solutionTime = solutionTime
        self.auxData = auxData

        # --- Internal attributes ---
        self._variables = variables  # Point to the dataset variables

        # Get the global indices of the variables local to this zone
        sharedVariablesIndices = np.array(sharedVariables, dtype=int).nonzero()[0]
        self._localVariables = set(range(len(variables))) - set(sharedVariablesIndices)
        self._localVariables = sorted(list(self._localVariables))
        self._localToGlobalVariableMap = {il: ig for il, ig in enumerate(self._localVariables)}

        # Validate the local variables and shared variables are consistent with the total variables
        if len(sharedVariablesIndices) + len(self._localVariables) != len(variables):
            raise ValueError(
                f"Zone '{self.title}' has {len(sharedVariablesIndices)} shared variables and "
                f"{len(self._localVariables)} local variables, but the total number of variables is {len(variables)}. "
                "The sum of shared and local variables must match the total number of variables."
            )

        # Create a mapping of shared variables to their parent zone indices
        self.sharedVariables = {}
        for ivar, izone in enumerate(sharedVariables):
            if izone not in self.sharedVariables:
                self.sharedVariables[izone] = []

            self.sharedVariables[izone].append(ivar)

    @property
    def triConnectivity(self) -> npt.NDArray:
        if self.zoneType == ZoneType.FETRIANGLE:
            return self.connectivity
        elif self.zoneType == ZoneType.FEQUADRILATERAL:
            return np.row_stack((self.connectivity[:, [0, 1, 2]], self.connectivity[:, [0, 2, 3]]))
        else:
            raise TypeError(f"'triConnectivity' not supported for {self.zoneType.name} zone type.")

    def _validateConnectivity(self) -> None:
        if self.zoneType == ZoneType.FELINESEG:
            assert self.connectivity.shape[1] == 2, "Connectivity shape does not match zone type."
        elif self.zoneType == ZoneType.FETRIANGLE:
            assert self.connectivity.shape[1] == 3, "Connectivity shape does not match zone type."
        elif self.zoneType == ZoneType.FEQUADRILATERAL:
            assert self.connectivity.shape[1] == 4, "Connectivity shape does not match zone type."
        elif self.zoneType == ZoneType.FETETRAHEDRON:
            assert self.connectivity.shape[1] == 4, "Connectivity shape does not match zone type."
        elif self.zoneType == ZoneType.FEBRICK:
            assert self.connectivity.shape[1] == 8, "Connectivity shape does not match zone type."
        else:
            # Prior validation step should ensure we don't reach this point
            # but raise an error just in case.
            raise TypeError(
                f"Zone type: {self.zoneType.name} and "
                f"connectivity of shape {self.connectivity.shape} are not compatible."
            )

    def _remapConnectivity(self) -> npt.NDArray:
        remap = np.full(self.uniqueIndices.max() + 1, -1, dtype=np.int64)
        remap[self.uniqueIndices] = np.arange(len(self.uniqueIndices))
        return remap[self.connectivity]


class TecplotData:
    def __init__(self, title: str, variables: List[str], zones: Optional[List[TecplotZone]] = None):
        self.title = title
        self.variables = variables
        self._lowercaseVariables = [var.lower() for var in variables]
        self.zones = zones or []
        self._validateAllZones()
        self._convertSharedVariablesToIndicesAllZones()
        self._validateSharedZoneVariables()

        # --- Internal attributes ---
        # Map to relate zones to shared variables in other zones {zone_index: {shared_zone_index: {variable_indices}}}
        # The 'variable_indices' are indices of the variables in the self.variables list.
        self._sharedVariableMap: Dict[int, Dict[int, Set[int]]] = {}
        self._zoneVariableOwnership: Dict[int, Set[int]] = {}

        # The shapes of the data arrays in each zone
        self._dataShapes: List[Tuple[int, ...]] = [zone.data.shape for zone in self.zones]

    def __len__(self) -> int:
        return len(self.zones)

    def __getitem__(self, index: int) -> TecplotZone:
        if index < 0 or index >= len(self.zones):
            raise IndexError("Zone index out of range.")
        return self.zones[index]

    def __setitem__(self, index: int, value: TecplotZone) -> None:
        if not isinstance(value, TecplotZone):
            raise TypeError("Value must be a TecplotZone instance.")

        self._validateZone(value.zoneType)

        if index < 0 or index >= len(self.zones):
            raise IndexError("Zone index out of range.")

        self.zones[index] = value

    def __iter__(self):
        """Iterate over the zones in the Tecplot data."""
        for zone in self.zones:
            yield zone

    def _validateAllZones(self) -> None:
        """Validate that all the current zones have the same type."""
        if not self.zones:
            # No zones to validate against, so we can accept any zone type
            return

        # Check if all zones have the same zone type
        if not all(z.zoneType == self.zones[0].zoneType for z in self.zones[1:]):
            raise TypeError("All zones must have the same zone type.")

    @staticmethod
    def _getVariableIndices(variables: List[str], variableNames: List[str]) -> List[int]:
        """Get the indices of the variables in the variableNames list."""
        variableIndices = []
        for var in variables:
            if var not in variableNames:
                raise ValueError(f"Variable '{var}' not found in the variable names.")
            variableIndices.append(variableNames.index(var))
        return variableIndices

    def _convertSharedVariablesToIndices(self, zone: TecplotZone) -> None:
        if zone.sharedVariables:
            # Convert shared variables from strings to indices
            sharedVars = {}
            for zoneIdx, variables in zone.sharedVariables.items():
                sharedVars[zoneIdx] = self._getVariableIndices(variables, self.variables)
            zone.sharedVariables = sharedVars

    def _convertSharedVariablesToIndicesAllZones(self) -> None:
        for zone in self.zones:
            self._convertSharedVariablesToIndices(zone)

    def _validateZone(self, zone: TecplotZone) -> None:
        """Validate that the zone type is one of the supported types."""
        if not self.zones:
            # No zones to validate against, so we can accept any zone type
            return

        if zone.zoneType != self.zones[0].zoneType:
            raise TypeError(
                f"Zone type {zone.zoneType.name} does not match the existing zone type {self.zones[0].zoneType.name}."
            )

    def _validateSharedZoneVariables(self) -> None:
        """Validate shared variables for all zones."""
        for zone in self.zones:
            if not zone.sharedVariables:
                continue

            if zone.zoneType in FEZones:
                validZones = [z for z in self.zones if z.zoneType in FEZones]
                errorMsg = "Finite element zones can only share data with other finite element zones."
            elif zone.zoneType == ZoneType.ORDERED:
                validZones = [z for z in self.zones if z.zoneType == ZoneType.ORDERED]
                errorMsg = "Ordered zones can only share data with other ordered zones."
            else:
                raise TypeError(f"Zone type {zone.zoneType.name} does not support shared variables.")

            if not validZones:
                raise TypeError(f"No valid zones found to use shared variables with Zone: {zone.title}. {errorMsg}")

            sharedZones: list[TecplotZone] = [self.zones[i] for i in zone.sharedVariables.keys()]
            sharedVars = list(zone.sharedVariables.values())
            localVars = set(range(len(self.variables))) - set(sharedVars)
            localShape = zone.data.shape

            if len(localVars) != localShape[-1]:
                raise ValueError(
                    f"Zone {zone.title} has {len(localVars)} local variables, "
                    f"but the data shape is {localShape}. "
                    "The number of local variables must match the last dimension of the data."
                )

            for sharedZone in sharedZones:
                if sharedZone not in validZones:
                    raise TypeError(
                        f"Zone type {zone.zoneType.name} does not support shared variables with incompatible zones."
                    )
                if sharedZone.data.shape[:-1] != localShape[:-1]:
                    raise ValueError(
                        f"Zone {zone.title} has a shape of {localShape}, "
                        f"but shared zone {sharedZone.title} has a shape of {sharedZone.data.shape}. "
                        "The leading dimensions must match for shared variables."
                    )

            totalVars = len(localVars) + len(sharedVars)
            if totalVars != len(self.variables):
                raise ValueError(
                    f"Zone {zone.title} has {totalVars} total variables (local + shared), "
                    f"but the total number of variables is {len(self.variables)}. "
                    "The total number of variables must match the number of variables in the Tecplot data."
                )

    def addZone(self, zone: TecplotZone) -> None:
        if not isinstance(zone, TecplotZone):
            raise TypeError("zone must be an instance of TecplotZone.")

        self._validateZone(zone)
        self._convertSharedVariablesToIndices(zone)

        self.zones.append(zone)

    def addZones(self, zones: List[TecplotZone]) -> None:
        for zone in zones:
            if not isinstance(zone, TecplotZone):
                raise TypeError("All items must be instances of TecplotZone.")
            self._validateZone(zone)
            self._convertSharedVariablesToIndices(zone)

            self.zones.append(zone)

    def getZoneDataDict(self, index: int) -> Dict[str, npt.NDArray]:
        zone = self.zones[index]
        data = {}

        # We need to figure out what data this zone contains and what data it shares with other zones
        sharedVariables = set()
        sharedData = {}

        if zone.sharedVariables:
            for zoneIdx, varIdxs in zone.sharedVariables.items():
                zoneData = self.zones[zoneIdx].data
                sharedVariables.add(varIdxs)
                for vi in varIdxs:
                    sharedData[vi] = zoneData[..., vi]  # Last dimension is the variable index

        # Get the variables that are unique to this zone
        uniqueVariables = set(range(len(self.variables))) - sharedVariables
        uniqueVariablesSorted = sorted(list(uniqueVariables))

        # Create a dictionary of the data local to this zone
        localData = {variableIdx: zone.data[..., variableIdx] for variableIdx in uniqueVariablesSorted}

        # Combine local and shared data
        data.update(localData)
        data.update(sharedData)

        # Sort the data dictionary by variable index
        data = {self.variables[k]: data[k] for k in sorted(data.keys())}

        return data

    def getZoneDataArray(self, index: int) -> npt.NDArray:
        zone = self.zones[index]
        data = {}

        # We need to figure out what data this zone contains and what data it shares with other zones
        sharedVariables = set()
        sharedData = {}

        if zone.sharedVariables:
            for zoneIdx, varIdxs in zone.sharedVariables.items():
                zoneData = self.zones[zoneIdx].data
                sharedVariables.add(varIdxs)
                for vi in varIdxs:
                    sharedData[vi] = zoneData[..., vi]  # Last dimension is the variable index

        # Get the variables that are unique to this zone
        uniqueVariables = set(range(len(self.variables))) - sharedVariables
        uniqueVariablesSorted = sorted(list(uniqueVariables))

        # Create a dictionary of the data local to this zone
        localData = {variableIdx: zone.data[..., variableIdx] for variableIdx in uniqueVariablesSorted}

        # Combine local and shared data
        data.update(localData)
        data.update(sharedData)

        # Create a numpy array of the data
        dataArray = np.stack([data[ivar] for ivar in range(len(self.variables))], axis=-1)

        return dataArray

    def getZoneHeader(self, index: int) -> Dict[str, Any]:
        zone = self.zones[index]
        return {
            "title": zone.title,
            "zoneType": zone.zoneType,
            "sharedVariables": zone.sharedVariables,
            "dataType": zone.dataType,
            "dataPacking": zone.dataPacking,
            "varLocation": zone.varLocation,
            "strandID": zone.strandID,
            "solutionTime": zone.solutionTime,
            "passiveVariables": zone.passiveVariables,
            "auxData": zone.auxData,
        }

    def getZoneTitle(self, index: int) -> str:
        return self.zones[index].title

    def getVariableAllZones(self, variable: str) -> List[npt.NDArray]:
        assert variable in self.variables, f"Variable '{variable}' not found in Tecplot data."
        varData = []
        for i in range(len(self.zones)):
            data = self.getZoneData(i)
            varData.append(data[variable])  # Get the variable data for each zone

        return varData

    def getVariableZone(self, variable: str, index: int) -> npt.NDArray:
        assert variable in self.variables, f"Variable '{variable}' not found in Tecplot data."
        data = self.getZoneData(index)
        return data[variable]

    def getConnectivityZone(self, index: int) -> npt.NDArray:
        return self.zones[index].connectivity

    def getUniqueConnectivityZone(self, index: int) -> npt.NDArray:
        return self.zones[index].uniqueConnectivity

    def getTriConnectivityZone(self, index: int) -> npt.NDArray:
        return self.zones[index].triConnectivity


# ==============================================================================
# ASCII WRITERS
# ==============================================================================
def writeArrayToFile(
    arr: npt.NDArray[np.float64],
    handle: TextIO,
    maxLineWidth: int = 4000,
    precision: int = 6,
    separator: Separator = Separator.SPACE,
) -> None:
    """
    Write a 2D numpy array to a file using numpy.array_str with custom
    formatting.

    Parameters
    ----------
    arr : npt.NDArray[np.float64]
        2D numpy array to write.
    file : TextIO
        A file-like object (e.g., opened with 'open()') to write to.
    maxLineWidth : int, optional
        Maximum width of each line in characters, by default 4000.
    precision : int, optional
        Number of decimal places for floating-point numbers, by default
        6.
    separator : Literal[" ", ",", "\\t", "\\n", "\\r"], optional
        Separator to use between elements, by default Separator.SPACE

    Raises
    ------
    ValueError
        If the input array is not 2-dimensional.
    """
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")

    kwargs = {
        "max_line_width": maxLineWidth,
        "separator": separator.value,
        "formatter": {"float_kind": lambda x: f"{x:.{precision}E}"},
        "threshold": np.inf,
    }

    def array2stringClean(val: npt.NDArray) -> str:
        """Clean the output of np.array2string to remove extra spaces."""
        return np.array2string(val, **kwargs).strip("[]").strip().replace("\n ", "\n")

    rows = [array2stringClean(row) for row in arr]
    handle.writelines(rows)


class TecplotZoneWriterASCII:
    def __init__(
        self,
        zone: Optional[TecplotZone] = None,
        separator: Separator = Separator.SPACE,
        maxLineWidth: int = 32000,
    ) -> None:
        """Abstract base class for writing Tecplot zones to ASCII files.

        Parameters
        ----------
        zone : T
            The Tecplot zone to write.
        datapacking : Literal["BLOCK", "POINT"]
            The data packing format. BLOCK is row-major, POINT is
            column-major.
        precision : Literal["SINGLE", "DOUBLE"]
            The floating point precision to write the data.
        """
        self.zone = zone
        self.separator = separator
        self.maxLineWidth = maxLineWidth

    def writeHeader(self, handle: TextIO):
        header = self.zone.header

        headerList = [f'ZONE T="{header.TITLE}"']

        # If we have an ordered zone type, write the dimensions
        if header.ZONETYPE == ZoneType.ORDERED:
            headerList += f"I={header.I}"

            if header.J:
                headerList += f"J={header.J}"

            if header.K:
                headerList += f"K={header.K}"

        if header.ZONETYPE in FEZones:
            headerList += f"NODES={header.NODES}"
            headerList += f"ELEMENTS={header.ELEMENTS}"
            headerList += f"ZONETYPE={header.ZONETYPE.name}"

        # Write the strand ID and solution time
        if header.STRANDID != StrandID.STATIC.value and header.STRANDID != StrandID.PENDING.value:
            # ASCII format does not support the -1 or -2 strand IDs
            # So we only write the strand ID if it is not -1
            headerList += f"STRANDID={header.STRANDID}"

        # Only write the solution time if it is set
        if header.SOLUTIONTIME != SolutionTime.UNSET.value:
            headerList += f"SOLUTIONTIME={header.SOLUTIONTIME}"

        # Write the datapacking
        headerList += f"DATAPACKING={header.DATAPACKING.value}\n"

        # Write all remaining metadata if not None
        if header.TOTALNUMFACENODES:
            headerList += f"TOTALNUMFACENODES={header.TOTALNUMFACENODES}"

        if header.NUMCONNECTEDBOUNDARYFACES:
            headerList += f"NUMCONNECTEDBOUNDARYFACES={header.NUMCONNECTEDBOUNDARYFACES}"

        if header.TOTALNUMBOUNDARYCONNECTIONS:
            headerList += f"TOTALNUMBOUNDARYCONNECTIONS={header.TOTALNUMBOUNDARYCONNECTIONS}"

        if header.FACENEIGHBORMODE:
            headerList += f"FACENEIGHBORMODE={header.FACENEIGHBORMODE.value}"

        if header.FACENEIGHBORCONNECTIONS:
            headerList += f"FACENEIGHBORCONNECTIONS={header.FACENEIGHBORCONNECTIONS}"

        if header.DT:
            headerList += f"DT={header.DT.name}"

        if header.VARLOCATION:
            headerList += f"VARLOCATION={header.VARLOCATION.value}"

        if header.VARSHARELIST:
            headerList += f"VARSHARELIST={','.join(header.VARSHARELIST)}"

        if header.NV:
            headerList += f"NV={header.NV}"

        if header.CONNECTIVITYSHAREZONE:
            headerList += f"CONNECTIVITYSHAREZONE={header.CONNECTIVITYSHAREZONE}"

        if header.PASSIVEVARLIST:
            headerList += f"PASSIVEVARLIST={','.join(header.PASSIVEVARLIST)}"

        if header.AUXDATA:
            for key, value in header.AUXDATA.items():
                headerList += f"AUXDATA {key}={value}"

        # Join the header into a single string with a line break every 5 elements
        chunks = [headerList[i : i + 5] for i in range(0, len(headerList), 5)]
        headerString = "\n".join([", ".join(chunk) for chunk in chunks])

        # Write the header and newline to the file
        handle.write(headerString)
        handle.write("\n")

    def writeFooter(self, handle: TextIO):
        # Only write a footer if there is connectivity data and the zone is FE
        if self.zone.header.ZONETYPE not in FEZones:
            handle.write("\n")
        else:
            if self.zone.connectivity.size > 0:
                connectivity = self.zone.connectivity + 1
                # Get the max characters in the connectivity
                maxChars = len(str(connectivity.max()))

                np.savetxt(handle, connectivity, fmt=f"%{maxChars}d")

                handle.write("\n")

    def writeData(self, handle: TextIO):
        data = np.stack([self.zone.data[var] for var in self.zone.data.keys()], axis=-1)

        if self.zone.header.DATAPACKING == DataPacking.POINT:
            data = data.reshape(-1, len(self.zone.data.keys()))
        else:
            data = data.reshape(-1, len(self.zone.data.keys())).T

        precision = DataPrecision[self.zone.header.DT.name].value
        writeArrayToFile(data, handle, maxLineWidth=self.maxLineWidth, precision=precision, separator=self.separator)


class TecplotWriter(ABC):
    def __init__(self, tecplotData: TecplotData):
        self.tecplotData = tecplotData

    @abstractmethod
    def write(self, filename: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def _writeZone(self, handle: TextIO, zone: TecplotZone) -> None:
        pass


class TecplotWriterASCII(TecplotWriter):
    def __init__(
        self,
        tecplotData: TecplotData,
        maxLineWidth: int = 32000,
        separator: Separator = Separator.SPACE,
    ) -> None:
        """Writer for Tecplot files in ASCII format.

        Parameters
        ----------
        title : str
            The title of the Tecplot file.
        zones : List[TecplotZone]
            A list of Tecplot zones to write.
        datapacking : Literal["BLOCK", "POINT"]
            The data packing format. BLOCK is row-major, POINT is
            column-major.
        precision : Literal["SINGLE", "DOUBLE"]
            The floating point precision to write the data.
        separator : Separator, optional
            Separator to use between elements. The Separator
            is an enum defined in :meth:`Separator <baseclasses.utils.tecplotIO.Separator>`,
            by default Separator.SPACE
        """
        super().__init__(tecplotData)
        self.maxLineWidth = maxLineWidth
        self.separator = separator

    def _writeVariables(self, handle: TextIO) -> None:
        """Write the variable names to the file.

        Parameters
        ----------
        handle : TextIO
            The file handle.
        """
        variables = [f'"{var}"' for var in self.tecplotData.variables]  # Wrap variable names in quotes
        variableString = ", ".join(variables)  # Join the variables into a single string
        handle.write(f"VARIABLES = {variableString}")
        handle.write("\n")

    def _writeHeader(self, handle: TextIO, zone: TecplotZone) -> None:
        pass

    def _writeZone(self, handle: TextIO, zone: TecplotZone) -> None:
        """Write a Tecplot zone to the file.

        Parameters
        ----------
        handle : TextIO
            The file handle.
        zone : TecplotZone
            The zone to write.

        Raises
        ------
        ValueError
            If the zone type is invalid.
        """
        self._zoneWriter.zone = zone
        self._zoneWriter.writeHeader(handle)
        self._zoneWriter.writeData(handle)
        self._zoneWriter.writeFooter(handle)

    def write(self, filename: Union[str, Path]) -> None:
        """Write the Tecplot file to disk.

        Parameters
        ----------
        filename : Union[str, Path]
            The filename as a string or pathlib.Path object.
        """
        with open(filename, "w") as handle:
            handle.write(f'TITLE = "{self.tecplotData.title}"\n')
            self._writeVariables(handle)
            for zone in self.tecplotData.zones:
                self._writeZone(handle, zone)


# ==============================================================================
# BINARY WRITERS
# ==============================================================================
def _writeInteger(handle: TextIO, value: int) -> None:
    """Write an integer to a binary file as int32.

    Parameters
    ----------
    handle : TextIO
        The file handle to write to.
    value : int
        Integer value to write.
    """
    handle.write(struct.pack("i", value))


def _writeFloat32(handle: TextIO, value: float) -> None:
    """Write a float to a binary file as float32.

    Parameters
    ----------
    handle : TextIO
        The file handle to write to.
    value : float
        Float value to write.
    """
    handle.write(struct.pack("f", value))


def _writeFloat64(handle: TextIO, value: float) -> None:
    """Write a float to a binary file as float64.

    Parameters
    ----------
    handle : TextIO
        The file handle to write to.
    value : float
        Float value to write.
    """
    handle.write(struct.pack("d", value))


def _writeString(handle: TextIO, value: str) -> None:
    """Write a string to a binary file as a string.

    Parameters
    ----------
    handle : TextIO
        The file handle to write to.
    value : str
        String value to write.
    """
    for char in value:
        asciiValue = ord(char)
        handle.write(struct.pack("i", asciiValue))

    handle.write(struct.pack("i", 0))


class TecplotZoneWriterBinary:
    def __init__(self, zone: Optional[TecplotZone] = None) -> None:
        """Abstract base class for writing Tecplot zones to binary
        files.

        Parameters
        ----------
        title : str
            The title of the Tecplot file.
        zone : T
            The Tecplot zone to write.
        precision : Literal["SINGLE", "DOUBLE"]
            The floating point precision to write the data.
        """
        self.zone = zone

    def writeHeader(self, handle: TextIO) -> None:
        """Write the common header information for all zones.

        Parameters
        ----------
        handle : TextIO
            The file handle.
        """
        if self.zone is None:
            raise ValueError("Zone object must be set.")

        header = self.zone.header

        # Write the zone marker
        _writeFloat32(handle, SectionMarkers.ZONE.value)  # Write the zone marker
        _writeString(handle, header.TITLE)  # Write the zone name
        _writeInteger(handle, BinaryFlags.NONE.value)  # Write the parent zone
        _writeInteger(handle, header.STRANDID)  # Write the strand ID
        _writeFloat64(handle, header.SOLUTIONTIME)  # Write the solution time
        _writeInteger(handle, BinaryFlags.NONE.value)  # Write the default color
        _writeInteger(handle, header.ZONETYPE.value)  # Write the zone type
        _writeInteger(handle, DataPacking.BLOCK.value)  # Data Packing (Always block for binary)
        _writeInteger(handle, header.VARLOCATION.value)  # Specify the variable location
        _writeInteger(handle, BinaryFlags.FALSE.value)  # Are raw 1-1 face neighbors supplied

        # Ordered zone header data
        if header.ZONETYPE == ZoneType.ORDERED:
            _writeInteger(handle, header.I)  # Write the I dimension
            _writeInteger(handle, header.J)  # Write the J dimension
            _writeInteger(handle, header.K)  # Write the K dimension

        # FE zone header data
        if header.ZONETYPE in FEZones:
            _writeInteger(handle, header.NODES)  # Write the number of nodes
            _writeInteger(handle, header.ELEMENTS)  # Write the number of elements
            _writeInteger(handle, 0)  # iCellDim (future use, set to 0)
            _writeInteger(handle, 0)  # jCellDim (future use, set to 0)
            _writeInteger(handle, 0)  # kCellDim (future use, set to 0)

        # Does the zone have aux data
        if header.AUXDATA:
            pass

    def writeData(self, handle: TextIO):
        """Write the zone data to the file.

        Parameters
        ----------
        handle : TextIO
            The file handle.
        """
        # Get the data into a single array
        data = np.stack([self.zone.data[var] for var in self.zone.variables], axis=-1)

        # Flatten the data such that each variable is a row
        data = data.reshape(-1, len(self.zone.variables)).T

        _writeFloat32(handle, SectionMarkers.ZONE.value)  # Write the zone marker

        # Write the variable data format for each variable
        for _ in range(len(self.zone.variables)):
            _writeInteger(handle, BinaryDataPrecisionCodes[self.precision].value)

        _writeInteger(handle, BinaryFlags.FALSE.value)  # No passive variables
        _writeInteger(handle, BinaryFlags.FALSE.value)  # No variable sharing
        _writeInteger(handle, BinaryFlags.NONE.value)  # No connectivity sharing

        # Write the min/max values for the variables
        for i in range(len(self.zone.variables)):
            _writeFloat64(handle, data[i, ...].min())
            _writeFloat64(handle, data[i, ...].max())

        # Write the data using the specified data format (single or double)
        data.astype(DataType[self.zone.header.DT].value).tofile(handle)

    def writeFooter(self, handle: TextIO):
        if self.zone.header.ZONETYPE in FEZones:
            self.zone.connectivity.astype("int32").tofile(handle)


class TecplotWriterBinary(TecplotWriter):
    def __init__(self, tecplotData: TecplotData) -> None:
        """Writer for Tecplot files in binary format.

        This writer only supports files formatted using the format
        designated by magic number ``#!TDV112``.

        See the Tecplot 360 User's Manual for more information on the
        binary file format and the magic number.

        Parameters
        ----------
        title : str
            The title of the Tecplot file.
        zones : List[TecplotZone]
            A list of Tecplot zones to write.
        precision : Literal["SINGLE", "DOUBLE"]
            The floating point precision to write the data.
        """
        super().__init__(tecplotData)
        self._magicNumber = b"#!TDV112"  # Magic number for Tecplot binary files
        self._zoneWriter = TecplotZoneWriterBinary()

    def write(self, filename: Union[str, Path]) -> None:
        """Write the Tecplot file to disk.

        Parameters
        ----------
        filename : Union[str, Path]
            The filename as a string or pathlib.Path object.
        """
        with open(filename, "wb") as handle:
            # Write the header information
            handle.write(self._magicNumber)  # Magic number
            _writeInteger(handle, 1)  # Byte order
            _writeInteger(handle, self.tecplotData.filetype.value)  # Full filetype
            _writeString(handle, self.tecplotData.title)  # Write the title
            _writeInteger(handle, len(self.tecplotData.variables))  # Write the number of variables

            for var in self.tecplotData.variables:
                _writeString(handle, var)  # Write the variable names

            # Write the zone headers
            for zone in self.tecplotData.zones:
                self._zoneWriter.zone = zone
                self._zoneWriter.writeHeader(handle)

            # Write the data marker
            _writeFloat32(handle, SectionMarkers.DATA.value)

            # Write the data and footer for each zone
            for zone in self.tecplotData.zones:
                self._zoneWriter.zone = zone
                self._zoneWriter.writeData(handle)
                self._zoneWriter.writeFooter(handle)


# ==============================================================================
# ASCII READERS
# ==============================================================================
def readArrayData(filename: str, iCurrent: int, nVals: int, nVars: int) -> Tuple[npt.NDArray, int]:
    """Read array data from a Tecplot ASCII file.

    Parameters
    ----------
    filename : str
        The filename of the Tecplot file.
    iCurrent : int
        The current line number in the file.
    nVals : int
        The number of values to read.
    nVars : int
        The number of variables in the data.

    Returns
    -------
    Tuple[npt.NDArray, int]
        The data and the number of lines read.
    """
    with open(filename, "r") as handle:
        lines = handle.readlines()

    pattern = r"[\s,\t\n\r]+"  # Separator pattern

    data = []
    nLines = 0
    while len(data) < nVals * nVars:
        line = lines[iCurrent].strip()  # Remove leading/trailing whitespace
        vals = [float(val) for val in re.split(pattern, line) if val]  # Split the line into values
        data.extend(vals)  # Add the values to the data list
        iCurrent += 1  # Increment the line number
        nLines += 1

    data = np.array(data)

    return data, nLines


class TecplotASCIIReader:
    def __init__(self, filename: Union[str, Path]) -> None:
        """Reader for Tecplot files in ASCII format.

        Parameters
        ----------
        filename : Union[str, Path]
            The filename as a string or pathlib.Path object.
        """
        self.filename = filename

    def _readZoneHeader(self, lines: List[str], iCurrent: int) -> Tuple[Dict[str, Any], int]:
        """Read the zone header information from a line in a Tecplot file.

        Parameters
        ----------
        lines : List[str]
            The list of lines in the file.
        iCurrent : int
            The current line number in the file.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the parsed zone header information.
        """
        # Collect the header lines
        exitPattern = re.compile(r"^\s*\d")
        headerFlag = False
        headerLines = []
        while not exitPattern.match(lines[iCurrent]):
            line = lines[iCurrent]

            if line.lower().startswith("zone"):
                # This line starts the zone record
                headerFlag = True

                # Remove zone from the start of the line case-insensitive
                line = re.sub(r"^zone", "", line, flags=re.IGNORECASE)

            if headerFlag:
                headerLines.append(line)

            iCurrent += 1

        # Join the header lines into a single string
        headerString = "".join(headerLines)

        # Replace newlines with commas
        headerString = headerString.replace("\n", ",")

        # Remove trailing comma
        headerString = headerString.rstrip(",")

        # Split the header by commas unless the comma is inside quotes
        headerList = re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", headerString)

        # Create a dictionary by splitting keys and values by equal signs
        headerDictRaw = {key.strip().upper(): value.strip() for key, value in [item.split("=") for item in headerList]}

        # Remove mirrored quotes from the values, but keep quotes if they are not mirrored
        headerDict = {key: value.strip('"') for key, value in headerDictRaw.items()}

        # If the keys in the headerDict match the zoneHeaders, convert the values to the enum value type
        for key, value in headerDict.items():
            if "AUXDATA" in key:
                headerDict[key] = str(value)
                continue

            try:
                headerEnum = ZoneHeader[key.upper()]
                if isinstance(headerEnum.value, EnumMeta):
                    headerDict[key] = headerEnum.value[value.upper()]
                else:
                    headerDict[key] = headerEnum.value(value)

            except KeyError:
                raise ValueError(f"Invalid zone header: {key}")

        return headerDict, iCurrent

    def _readOrderedZoneData(
        self, iCurrent: int, variables: List[str], zoneHeaderDict: Dict[str, Any]
    ) -> Tuple[TecplotOrderedZone, int]:
        """Read the data for an ordered Tecplot zone.

        Parameters
        ----------
        iCurrent : int
            The current line number in the file.
        variables : List[str]
            The list of variable names.
        zoneHeaderDict : Dict[str, Any]
            The zone header information.

        Returns
        -------
        Tuple[TecplotOrderedZone, int]
            The ordered zone object and the number of lines read.
        """
        iMax = zoneHeaderDict[ZoneHeader.I.name]
        jMax = zoneHeaderDict[ZoneHeader.J.name]
        kMax = zoneHeaderDict[ZoneHeader.K.name]
        nNodes = iMax * jMax * kMax
        shape = (iMax, jMax, kMax, len(variables))

        if zoneHeaderDict[ZoneHeader.DATAPACKING.name] == DataPacking.POINT:
            # Point data is column-major
            nodalData, nodeOffset = readArrayData(self.filename, iCurrent, nNodes, len(variables))
            nodalData = nodalData.reshape(shape, order="C").squeeze()
        else:
            # Block data is row-major
            nodalData, nodeOffset = readArrayData(self.filename, iCurrent, nNodes, len(variables))
            nodalData = nodalData.reshape(shape, order="F").squeeze()

        data = {var: nodalData[..., i] for i, var in enumerate(variables)}
        zone = TecplotOrderedZone(data, zoneHeaderDict)

        return zone, nodeOffset

    def _readFEZoneData(
        self, iCurrent: int, variables: List[str], zoneHeaderDict: Dict[str, Any]
    ) -> Tuple[TecplotFEZone, int]:
        """Read the data for a finite element Tecplot zone.

        Parameters
        ----------
        iCurrent : int
            The current line number in the file.
        variables : List[str]
            The list of variable names.
        zoneHeaderDict : Dict[str, Any]
            The zone header information.

        Returns
        -------
        Tuple[TecplotFEZone, int]
            The finite element zone object and the number of lines read.
        """
        nNodes = zoneHeaderDict[ZoneHeader.NODES.name]
        nElements = zoneHeaderDict[ZoneHeader.ELEMENTS.name]

        if zoneHeaderDict[ZoneHeader.DATAPACKING.name] == DataPacking.POINT:
            # Point data is column-major
            nodalData, nodeOffset = readArrayData(self.filename, iCurrent, nNodes, len(variables))
            nodalData = nodalData.reshape(nNodes, len(variables), order="C")
        else:
            # Block data is row-major
            nodalData, nodeOffset = readArrayData(self.filename, iCurrent, nNodes, len(variables))
            nodalData = nodalData.reshape(nNodes, len(variables), order="F")

        connectivity = np.loadtxt(self.filename, skiprows=iCurrent + nodeOffset, max_rows=nElements, dtype=int)

        # Check if the nodal data is 1D
        if nodalData.ndim == 1:
            nodalData = nodalData.reshape(-1, len(variables))

        if connectivity.ndim == 1:
            connectivity = connectivity.reshape(nElements, -1)

        data = {var: nodalData[..., i] for i, var in enumerate(variables)}
        zone = TecplotFEZone(data, connectivity - 1, zoneHeaderDict)

        return zone, nodeOffset + nElements

    def _readZoneData(self, lines: List[str], iLine: int, variables: List[str]) -> Tuple[TecplotZone, int]:
        """Read the data for a Tecplot zone.

        Parameters
        ----------
        lines : List[str]
            The list of lines in the Tecplot file.
        iLine : int
            The current line number in the file.
        variables : List[str]
            The list of variable names.

        Returns
        -------
        Tuple[TecplotZone, int]
            The Tecplot zone object and the number of lines read.
        """
        zoneHeaderDict, iLine = self._readZoneHeader(lines, iLine)

        if zoneHeaderDict[ZoneHeader.ZONETYPE.name] == ZoneType.ORDERED:
            zone, iOffset = self._readOrderedZoneData(iLine, variables, zoneHeaderDict)
        else:
            zone, iOffset = self._readFEZoneData(iLine, variables, zoneHeaderDict)

        return zone, iLine + iOffset

    def read(self) -> Tuple[str, List[TecplotZone]]:
        """Read the Tecplot file and return the title and zones.

        Returns
        -------
        Tuple[str, List[TecplotZone]]
            The title of the Tecplot file and a list of Tecplot zones.

        Raises
        ------
        ValueError
            If the file is not a valid Tecplot file.
        ValueError
            If the title is missing.
        """
        with open(self.filename, "r") as handle:
            lines = handle.readlines()

        zones = []

        # Get the title
        title = re.search(r'title\s*=\s*"(.*)"', lines[0], re.IGNORECASE)
        if title is None:
            raise ValueError("Tecplot file must have a title on the first line.")
        title = title.group(1)

        # Get the variable names
        variables = re.findall(r'"([^"]*)"', lines[1])

        iLine = 2
        while iLine < len(lines):
            zone, iLine = self._readZoneData(lines, iLine, variables)
            zones.append(zone)

            # Skip any empty lines
            while iLine < len(lines) and not lines[iLine].strip():
                iLine += 1

        return title, zones


# ==============================================================================
# BINARY READERS
# ==============================================================================
class TecplotBinaryReader:
    def __init__(self, filename: Union[str, Path]) -> None:
        """Reader for Tecplot files in binary format.

        Parameters
        ----------
        filename : Union[str, Path]
            The filename as a string or pathlib.Path object
        """
        self.filename = filename
        self._nVariables = 0
        self._variables = []

    def _readString(self, handle: TextIO) -> str:
        """Read a string from a binary file that is null-terminated.

        Parameters
        ----------
        handle : TextIO
            The file handle to read from.

        Returns
        -------
        str
            The string read from the file.
        """
        result = []
        while True:
            data = handle.read(4)
            integer = struct.unpack_from("i", data, 0)[0]

            if integer == 0:
                break

            result.append(chr(integer))

        return "".join(result)

    def _readInteger(self, handle: TextIO, offset: int = 0) -> int:
        """Read an integer from a binary file as int32.

        Parameters
        ----------
        handle : TextIO
            The file handle to read from.
        offset : int, optional
            The offset (in bytes) from the file's current position,
            by default 0

        Returns
        -------
        int
            The integer read from the file.
        """
        return int(np.fromfile(handle, dtype=np.int32, count=1, offset=offset)[0])

    def _readIntegerArray(self, handle: TextIO, nValues: int, offset: int = 0) -> npt.NDArray[np.int32]:
        """Read an array of integers from a binary file as int32.

        Parameters
        ----------
        handle : TextIO
            The file handle to read from.
        nValues : int
            The number of values to read.
        offset : int, optional
            The offset (in bytes) from the file's current position,
            by default 0

        Returns
        -------
        npt.NDArray[np.int32]
            The integer array read from the file.
        """
        return np.fromfile(handle, dtype=np.int32, count=nValues, offset=offset)

    def _readFloat32(self, handle: TextIO, offset: int = 0) -> float:
        """Read a float from a binary file as float32.

        Parameters
        ----------
        handle : TextIO
            The file handle to read from.
        offset : int, optional
            The offset (in bytes) from the file's current position,
            by default 0

        Returns
        -------
        float
            The float read from the file.
        """
        return float(np.fromfile(handle, dtype=np.float32, count=1, offset=offset)[0])

    def _readFloat32Array(self, handle: TextIO, nValues: int, offset: int = 0) -> npt.NDArray[np.float32]:
        """Read an array of floats from a binary file as float32.

        Parameters
        ----------
        handle : TextIO
            The file handle to read from.
        nValues : int
            The number of values to read.
        offset : int, optional
            The offset (in bytes) from the file's current position,
            by default 0

        Returns
        -------
        npt.NDArray[np.float32]
            The float array read from the file.
        """
        return np.fromfile(handle, dtype=np.float32, count=nValues, offset=offset)

    def _readFloat64(self, handle: TextIO, offset: int = 0) -> float:
        """Read a float from a binary file as float64.

        Parameters
        ----------
        handle : TextIO
            The file handle to read from.
        offset : int, optional
            The offset (in bytes) from the file's current position,
            by default 0

        Returns
        -------
        float
            The float read from the file.
        """
        return float(np.fromfile(handle, dtype=np.float64, count=1, offset=offset)[0])

    def _readFloat64Array(self, handle: TextIO, nValues: int, offset: int = 0) -> npt.NDArray[np.float64]:
        """Read an array of floats from a binary file as float64.

        Parameters
        ----------
        handle : TextIO
            The file handle to read from.
        nValues : int
            The number of values to read.
        offset : int, optional
            The offset (in bytes) from the file's current position,
            by default 0

        Returns
        -------
        npt.NDArray[np.float64]
            The float array read from the file.
        """
        return np.fromfile(handle, dtype=np.float64, count=nValues, offset=offset)

    def _readOrderedZone(self, handle: TextIO, zoneName: str, strandID: int, solutionTime: float) -> TecplotOrderedZone:
        """Read an ordered Tecplot zone header from a binary file.

        Parameters
        ----------
        handle : TextIO
            The file handle to read from.
        zoneName : str
            The name of the zone.
        strandID : int
            The strand ID.
        solutionTime : float
            The solution time.

        Returns
        -------
        TecplotOrderedZone
            The ordered Tecplot zone object.
        """
        iMax = self._readInteger(handle)
        jMax = self._readInteger(handle)
        kMax = self._readInteger(handle)

        return TecplotOrderedZone(
            zoneName,
            {var: np.zeros((iMax, jMax, kMax)).squeeze() for _, var in enumerate(self._variables)},
            solutionTime=solutionTime,
            strandID=strandID,
        )

    def _readFEZone(
        self, handle: TextIO, zoneName: str, zoneType: int, strandID: int, solutionTime: float
    ) -> TecplotFEZone:
        """Read a finite element Tecplot zone header from a binary file.

        Parameters
        ----------
        handle : TextIO
            The file handle to read from.
        zoneName : str
            The name of the zone.
        zoneType : int
            The zone type.
        strandID : int
            The strand ID.
        solutionTime : float
            The solution time.

        Returns
        -------
        TecplotFEZone
            The finite element Tecplot zone object.

        Raises
        ------
        ValueError
            If the zone type is invalid.
        """
        nNodes = self._readInteger(handle)
        nElements = self._readInteger(handle)
        iCellDim = self._readInteger(handle)  # NOQA: F841
        jCellDim = self._readInteger(handle)  # NOQA: F841
        kCellDim = self._readInteger(handle)  # NOQA: F841

        if zoneType == ZoneType.FELINESEG.value:
            connectivity = np.zeros((nElements, 2), dtype=int)
        elif zoneType == ZoneType.FETRIANGLE.value:
            connectivity = np.zeros((nElements, 3), dtype=int)
        elif zoneType == ZoneType.FEQUADRILATERAL.value:
            connectivity = np.zeros((nElements, 4), dtype=int)
        elif zoneType == ZoneType.FETETRAHEDRON.value:
            connectivity = np.zeros((nElements, 4), dtype=int)
        elif zoneType == ZoneType.FEBRICK.value:
            connectivity = np.zeros((nElements, 8), dtype=int)
        else:
            raise ValueError("Invalid zone type.")

        return TecplotFEZone(
            zoneName,
            {var: np.zeros(nNodes) for _, var in enumerate(self._variables)},
            connectivity,
            zoneType=ZoneType(zoneType),
            solutionTime=solutionTime,
            strandID=strandID,
        )

    def _readZoneHeader(self, handle: TextIO) -> Union[TecplotOrderedZone, TecplotFEZone]:
        """Read a Tecplot zone header from a binary file.

        Parameters
        ----------
        handle : TextIO
            The file handle to read from.

        Returns
        -------
        Union[TecplotOrderedZone, TecplotFEZone]
            The Tecplot zone object, either ordered or finite element.
        """
        zoneName = self._readString(handle)
        parentZone = self._readInteger(handle)  # NOQA: F841
        strandID = self._readInteger(handle)
        solutionTime = self._readFloat64(handle)
        defaultColor = self._readInteger(handle)  # NOQA: F841
        zoneType = self._readInteger(handle)
        datapacking = self._readInteger(handle)  # NOQA: F841
        variableLocation = self._readInteger(handle)  # NOQA: F841
        rawFaceNeighbors = self._readInteger(handle)  # NOQA: F841

        headerDict = {
            ZoneHeader.TITLE.name: zoneName,
            ZoneHeader.STRANDID.name: strandID,
            ZoneHeader.SOLUTIONTIME.name: solutionTime,
            ZoneHeader.ZONETYPE.name: ZoneType(zoneType),
            ZoneHeader.DATAPACKING.name: DataPacking(datapacking),
            ZoneHeader.VARLOCATION.name: VariableLocation(variableLocation),
        }

        if zoneType == ZoneType.ORDERED.value:
            zone = self._readOrderedZone(handle, zoneName, strandID, solutionTime)
        else:
            zone = self._readFEZone(handle, zoneName, zoneType, strandID, solutionTime)

        return zone

    def read(self) -> Tuple[str, List[TecplotZone]]:
        """Read the Tecplot file and return the title and zones.

        Returns
        -------
        Tuple[str, List[TecplotZone]]
            The title of the Tecplot file and a list of Tecplot zones.

        Raises
        ------
        ValueError
            If the file is not a valid Tecplot file.
        """
        file = open(self.filename, "rb")
        file.seek(0, 2)
        fileSize = file.tell()
        file.seek(0)

        magic = file.read(8).decode("utf-8")
        if magic != "#!TDV112":
            raise ValueError("Invalid Tecplot binary file version.")

        byteOrder = self._readInteger(file)  # NOQA: F841
        filetype = self._readInteger(file)  # NOQA: F841
        title = self._readString(file)
        self._nVariables = self._readInteger(file)
        self._variables = [self._readString(file) for _ in range(self._nVariables)]
        zones: List[Union[TecplotOrderedZone, TecplotFEZone]] = []

        # Read all the zone headers
        while True:
            marker = self._readFloat32(file)

            if marker == SectionMarkers.ZONE.value:
                # Initialize zone from the header
                zone = self._readZoneHeader(file)
                zones.append(zone)
            if marker == SectionMarkers.DATA.value:
                break

        # Read the data for each zone
        izone = 0
        while file.tell() < fileSize:
            zoneMarker = self._readFloat32(file)  # NOQA: F841
            dataFormats = [self._readInteger(file) for _ in range(self._nVariables)]
            passiveVariables = self._readInteger(file)  # NOQA: F841
            variableSharing = self._readInteger(file)  # NOQA: F841
            connSharing = self._readInteger(file)  # NOQA: F841
            minMaxArray = self._readFloat64Array(file, 2 * self._nVariables).reshape(self._nVariables, 2)  # NOQA: F841

            if isinstance(zones[izone], TecplotOrderedZone):
                iMax = zones[izone].iMax
                jMax = zones[izone].jMax
                kMax = zones[izone].kMax

                for i in range(self._nVariables):
                    if dataFormats[i] == BinaryDataPrecisionCodes.SINGLE.value:
                        readData = self._readFloat32Array(file, iMax * jMax * kMax).reshape(iMax, jMax, kMax).squeeze()
                    else:
                        readData = self._readFloat64Array(file, iMax * jMax * kMax).reshape(iMax, jMax, kMax).squeeze()

                    zones[izone].data[self._variables[i]][...] = readData

            if isinstance(zones[izone], TecplotFEZone):
                nNodes = zones[izone].nNodes
                nElements = zones[izone].nElements

                for i in range(self._nVariables):
                    if dataFormats[i] == BinaryDataPrecisionCodes.SINGLE.value:
                        readData = self._readFloat32Array(file, nNodes)
                    else:
                        readData = self._readFloat64Array(file, nNodes)

                    zones[izone].data[self._variables[i]][...] = readData

                connectivitySize = zones[izone].connectivity.size
                connectivity = self._readIntegerArray(file, connectivitySize).reshape(nElements, -1)
                zones[izone].connectivity = connectivity

            izone += 1

        file.close()

        return title, zones


# ==============================================================================
# PUBLIC FUNCTIONS
# ==============================================================================
def writeTecplot(
    filename: Union[str, Path],
    title: str,
    zones: List[TecplotZone],
    datapacking: Literal["BLOCK", "POINT"] = "POINT",
    precision: Literal["SINGLE", "DOUBLE"] = "SINGLE",
    separator: Separator = Separator.SPACE,
) -> None:
    """Write a Tecplot file to disk. The file format is determined by the
    file extension. If the extension is .plt, the file will be written in
    binary format. If the extension is .dat, the file will be written in
    ASCII format.

    .. note::

        - ASCII files can be written with either BLOCK or POINT data packing.
        - Binary files are always written with BLOCK data packing.

    Parameters
    ----------
    filename : Union[str, Path]
        The filename as a string or pathlib.Path object.
    title : str
        The title of the Tecplot file.
    zones : List[TecplotZone]
        A list of Tecplot zones to write
    datapacking : Literal["BLOCK", "POINT"], optional
        The data packing format. BLOCK is row-major, POINT is
        column-major, by default "POINT"
    precision : Literal["SINGLE", "DOUBLE"], optional
        The floating point precision to write the data, by default
        "SINGLE"
    separator : Separator, optional
        The separator to use when writing ASCII files. The Separator
        is an enum defined in :meth:`Separator <baseclasses.utils.tecplotIO.Separator>`,
        by default Separator.SPACE

    Raises
    ------
    ValueError
        If the file extension is invalid.

    Examples
    --------
    .. code-block:: python

        from baseclasses import tecplotIO as tpio
        import numpy as np

        nx, ny, nz = 10, 10, 10
        X, Y, Z = np.meshgrid(np.random.rand(nx), np.random.rand(ny), np.random.rand(nz), indexing="ij")
        data = {"X": X, "Y": Y, "Z": Z}
        zone = tpio.TecplotOrderedZone("OrderedZone", data)

        # Write the Tecplot file in ASCII format
        tpio.writeTecplot("ordered.dat", "Ordered Zone", [zone], datapacking="BLOCK", precision="SINGLE", separator=tpio.Separator.SPACE)

        # Write the Tecplot file in binary format
        tpio.writeTecplot("ordered.plt", "Ordered Zone", [zone], precision="SINGLE")

    """
    filepath = Path(filename)
    if filepath.suffix == ".plt":
        writer = TecplotWriterBinary(title, zones, precision)
        writer.write(filepath)
    elif filepath.suffix == ".dat":
        writer = TecplotWriterASCII(title, zones, datapacking, precision, separator)
        writer.write(filename)
    else:
        raise ValueError("Invalid file extension. Must be .plt (binary) or .dat (ASCII).")


def readTecplot(filename: Union[str, Path]) -> Tuple[str, List[Union[TecplotOrderedZone, TecplotFEZone]]]:
    """Read a Tecplot file from disk. The file format is determined by
    the file extension. If the extension is .plt, the file will be read
    in binary format. If the extension is .dat, the file will be read in
    ASCII format.

    Parameters
    ----------
    filename : Union[str, Path]
        The filename as a string or pathlib.Path object.

    Returns
    -------
    Tuple[str, List[Union[TecplotOrderedZone, TecplotFEZone]]]
        The title of the Tecplot file and a list of Tecplot zones.

    Raises
    ------
    ValueError
        If the file extension is invalid.

    Examples
    --------
    .. code-block:: python

        from baseclasses import tecplotIO as tpio

        # Read a Tecplot file in ASCII format
        title, zones = tpio.readTecplot("ordered.dat")

        # Read a Tecplot file in binary format
        title, zones = tpio.readTecplot("ordered.plt")
    """
    filepath = Path(filename)
    if filepath.suffix == ".plt":
        reader = TecplotBinaryReader(filename)
        title, zones = reader.read()
    elif filepath.suffix == ".dat":
        reader = TecplotASCIIReader(filename)
        title, zones = reader.read()
    else:
        raise ValueError("Invalid file extension. Must be .plt (binary) or .dat (ASCII).")

    return title, zones
