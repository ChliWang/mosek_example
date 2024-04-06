# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wcl/github_upload/mosek_example/SOCP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wcl/github_upload/mosek_example/SOCP/build

# Include any dependencies generated for this target.
include CMakeFiles/socp_mosek.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/socp_mosek.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/socp_mosek.dir/flags.make

CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o: CMakeFiles/socp_mosek.dir/flags.make
CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o: ../example/socp_mosek.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wcl/github_upload/mosek_example/SOCP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o -c /home/wcl/github_upload/mosek_example/SOCP/example/socp_mosek.cpp

CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wcl/github_upload/mosek_example/SOCP/example/socp_mosek.cpp > CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.i

CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wcl/github_upload/mosek_example/SOCP/example/socp_mosek.cpp -o CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.s

CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o.requires:

.PHONY : CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o.requires

CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o.provides: CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o.requires
	$(MAKE) -f CMakeFiles/socp_mosek.dir/build.make CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o.provides.build
.PHONY : CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o.provides

CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o.provides.build: CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o


# Object files for target socp_mosek
socp_mosek_OBJECTS = \
"CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o"

# External object files for target socp_mosek
socp_mosek_EXTERNAL_OBJECTS =

socp_mosek: CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o
socp_mosek: CMakeFiles/socp_mosek.dir/build.make
socp_mosek: /home/wcl/Solvers/mosek/10.1/tools/platform/linux64x86/bin/libmosek64.so.10.1
socp_mosek: /home/wcl/Solvers/mosek/10.1/tools/platform/linux64x86/bin/libfusion64.so.10.1
socp_mosek: CMakeFiles/socp_mosek.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wcl/github_upload/mosek_example/SOCP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable socp_mosek"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/socp_mosek.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/socp_mosek.dir/build: socp_mosek

.PHONY : CMakeFiles/socp_mosek.dir/build

CMakeFiles/socp_mosek.dir/requires: CMakeFiles/socp_mosek.dir/example/socp_mosek.cpp.o.requires

.PHONY : CMakeFiles/socp_mosek.dir/requires

CMakeFiles/socp_mosek.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/socp_mosek.dir/cmake_clean.cmake
.PHONY : CMakeFiles/socp_mosek.dir/clean

CMakeFiles/socp_mosek.dir/depend:
	cd /home/wcl/github_upload/mosek_example/SOCP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wcl/github_upload/mosek_example/SOCP /home/wcl/github_upload/mosek_example/SOCP /home/wcl/github_upload/mosek_example/SOCP/build /home/wcl/github_upload/mosek_example/SOCP/build /home/wcl/github_upload/mosek_example/SOCP/build/CMakeFiles/socp_mosek.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/socp_mosek.dir/depend

