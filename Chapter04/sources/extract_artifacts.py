import os
import pefile
import glob

csv = file('MalwareArtifacts.csv','w')

files = glob.glob('c:\\MalwareSamples\\*.exe')

csv.write("AddressOfEntryPoint,MajorLinkerVersion,MajorImageVersion,
MajorOperatingSystemVersion,,DllCharacteristics,SizeOfStackReserve,
NumberOfSections,ResourceSize,\n")

for file in files:
    suspect_pe = pefile.PE(file)

    csv.write( str(suspect_pe.OPTIONAL_HEADER.AddressOfEntryPoint) + ',')
    csv.write( str(suspect_pe.OPTIONAL_HEADER.MajorLinkerVersion) + ',')
    csv.write( str(suspect_pe.OPTIONAL_HEADER.MajorImageVersion) + ',')
    csv.write( str(suspect_pe.OPTIONAL_HEADER.MajorOperatingSystemVersion) + ',')
    csv.write( str(suspect_pe.OPTIONAL_HEADER.DllCharacteristics) + ',')
    csv.write( str(suspect_pe.OPTIONAL_HEADER.SizeOfStackReserve) + ',')
    csv.write( str(suspect_pe.FILE_HEADER.NumberOfSections) + ',')
    csv.write( str(suspect_pe.OPTIONAL_HEADER.DATA_DIRECTORY[2].Size) + "\n")

csv.close()

