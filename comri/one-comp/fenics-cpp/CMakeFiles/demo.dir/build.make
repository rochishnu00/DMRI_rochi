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
CMAKE_SOURCE_DIR = /home/singularity/Documents/comri/one-comp/fenics-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/singularity/Documents/comri/one-comp/fenics-cpp

# Include any dependencies generated for this target.
include CMakeFiles/demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo.dir/flags.make

CMakeFiles/demo.dir/main.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/singularity/Documents/comri/one-comp/fenics-cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/main.cpp.o -c /home/singularity/Documents/comri/one-comp/fenics-cpp/main.cpp

CMakeFiles/demo.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/singularity/Documents/comri/one-comp/fenics-cpp/main.cpp > CMakeFiles/demo.dir/main.cpp.i

CMakeFiles/demo.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/singularity/Documents/comri/one-comp/fenics-cpp/main.cpp -o CMakeFiles/demo.dir/main.cpp.s

CMakeFiles/demo.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/demo.dir/main.cpp.o.requires

CMakeFiles/demo.dir/main.cpp.o.provides: CMakeFiles/demo.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/demo.dir/build.make CMakeFiles/demo.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/demo.dir/main.cpp.o.provides

CMakeFiles/demo.dir/main.cpp.o.provides.build: CMakeFiles/demo.dir/main.cpp.o


# Object files for target demo
demo_OBJECTS = \
"CMakeFiles/demo.dir/main.cpp.o"

# External object files for target demo
demo_EXTERNAL_OBJECTS =

demo: CMakeFiles/demo.dir/main.cpp.o
demo: CMakeFiles/demo.dir/build.make
demo: /usr/local/lib/libdolfin.so.2018.1.0
demo: /usr/local/lib/libmshr.so
demo: /usr/lib/x86_64-linux-gnu/libboost_timer.so
demo: /usr/lib/x86_64-linux-gnu/hdf5/mpich/libhdf5.so
demo: /usr/lib/x86_64-linux-gnu/libsz.so
demo: /usr/lib/x86_64-linux-gnu/libz.so
demo: /usr/lib/x86_64-linux-gnu/libdl.so
demo: /usr/lib/x86_64-linux-gnu/libm.so
demo: /usr/local/slepc-32/lib/libslepc.so
demo: /usr/local/petsc-32/lib/libpetsc.so
demo: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
demo: /usr/lib/x86_64-linux-gnu/libmpich.so
demo: CMakeFiles/demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/singularity/Documents/comri/one-comp/fenics-cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo.dir/build: demo

.PHONY : CMakeFiles/demo.dir/build

CMakeFiles/demo.dir/requires: CMakeFiles/demo.dir/main.cpp.o.requires

.PHONY : CMakeFiles/demo.dir/requires

CMakeFiles/demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo.dir/clean

CMakeFiles/demo.dir/depend:
	cd /home/singularity/Documents/comri/one-comp/fenics-cpp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/singularity/Documents/comri/one-comp/fenics-cpp /home/singularity/Documents/comri/one-comp/fenics-cpp /home/singularity/Documents/comri/one-comp/fenics-cpp /home/singularity/Documents/comri/one-comp/fenics-cpp /home/singularity/Documents/comri/one-comp/fenics-cpp/CMakeFiles/demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo.dir/depend

