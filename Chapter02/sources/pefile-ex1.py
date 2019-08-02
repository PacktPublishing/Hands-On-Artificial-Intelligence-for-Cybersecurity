import os

import pefile


notepad = pefile.PE("notepad.exe", fast_load=True)


dbgRVA = notepad.OPTIONAL_HEADER.DATA_DIRECTORY[6].VirtualAddress
 
imgver = notepad.OPTIONAL_HEADER.MajorImageVersion

expRVA = notepad.OPTIONAL_HEADER.DATA_DIRECTORY[0].VirtualAddress

iat = notepad.OPTIONAL_HEADER.DATA_DIRECTORY[12].VirtualAddress

sections = notepad.FILE_HEADER.NumberOfSections

dll = notepad.OPTIONAL_HEADER.DllCharacteristics


print("Notepad PE info: \n")

print ("Debug RVA: " + dbgRVA)

print ("\nImage Version: " + imgver)

print ("\nExport RVA: " + expRVA)

print ("\nImport Address Table: " + iat)

print ("\nNumber of Sections: " + sections)

print ("\nDynamic linking libraries: " + dll)

