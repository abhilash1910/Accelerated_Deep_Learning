# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /root/miniconda3/envs/mkl-torch-majumder/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /root/miniconda3/envs/mkl-torch-majumder/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/majumder/torch_resnet_jit

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/majumder/torch_resnet_jit

# Include any dependencies generated for this target.
include CMakeFiles/torch_resnet_jit.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/torch_resnet_jit.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/torch_resnet_jit.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/torch_resnet_jit.dir/flags.make

CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.o: CMakeFiles/torch_resnet_jit.dir/flags.make
CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.o: torch_resnet_jit.cpp
CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.o: CMakeFiles/torch_resnet_jit.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/majumder/torch_resnet_jit/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.o -MF CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.o.d -o CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.o -c /home/majumder/torch_resnet_jit/torch_resnet_jit.cpp

CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/majumder/torch_resnet_jit/torch_resnet_jit.cpp > CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.i

CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/majumder/torch_resnet_jit/torch_resnet_jit.cpp -o CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.s

# Object files for target torch_resnet_jit
torch_resnet_jit_OBJECTS = \
"CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.o"

# External object files for target torch_resnet_jit
torch_resnet_jit_EXTERNAL_OBJECTS =

torch_resnet_jit: CMakeFiles/torch_resnet_jit.dir/torch_resnet_jit.cpp.o
torch_resnet_jit: CMakeFiles/torch_resnet_jit.dir/build.make
torch_resnet_jit: /home/majumder/libtorch/lib/libtorch.so
torch_resnet_jit: /home/majumder/libtorch/lib/libc10.so
torch_resnet_jit: /home/majumder/libtorch/lib/libkineto.a
torch_resnet_jit: /home/majumder/libtorch/lib/libc10.so
torch_resnet_jit: CMakeFiles/torch_resnet_jit.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/majumder/torch_resnet_jit/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable torch_resnet_jit"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torch_resnet_jit.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/torch_resnet_jit.dir/build: torch_resnet_jit
.PHONY : CMakeFiles/torch_resnet_jit.dir/build

CMakeFiles/torch_resnet_jit.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/torch_resnet_jit.dir/cmake_clean.cmake
.PHONY : CMakeFiles/torch_resnet_jit.dir/clean

CMakeFiles/torch_resnet_jit.dir/depend:
	cd /home/majumder/torch_resnet_jit && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/majumder/torch_resnet_jit /home/majumder/torch_resnet_jit /home/majumder/torch_resnet_jit /home/majumder/torch_resnet_jit /home/majumder/torch_resnet_jit/CMakeFiles/torch_resnet_jit.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/torch_resnet_jit.dir/depend

