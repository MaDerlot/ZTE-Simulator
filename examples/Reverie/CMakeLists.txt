set(reverie_mpi_libs)
if(${ENABLE_MPI})
  set(reverie_mpi_libs ${libmpi})
endif()

build_example(
  NAME reverie-evaluation-sigcomm2023
  SOURCE_FILES 
    reverie-evaluation-sigcomm2023.cc
    kira_functions.cc
  HEADER_FILES
    kira_functions.h
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libtraffic-control}
    ${libinternet}
    ${libapplications}
    ${reverie_mpi_libs}
)
build_example(
  NAME test-parallel
  SOURCE_FILES 
    test-parallel.cc
    kira_functions.cc
  HEADER_FILES
    kira_functions.h
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libtraffic-control}
    ${libinternet}
    ${libapplications}
)
set_source_files_properties(PROPERTIES SKIP_PRECOMPILE_HEADERS ON)