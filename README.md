# go1_pybullet
Doccumentation on setting the pybullet enviornment for Go1 to run the robot with NMPC and to collect data for training


Todo Update the readme
```bash
    hi
``` 

1. Please make sure you have the ROS foxy installed

2. Installing Dependencies from robotpkg
Note: list of available packages from robotpkg
https://github.com/machines-in-motion/machines-in-motion.github.io/wiki/laas_package_from_binaries

    A) For now I Installed everything form robotpkg
    ```bash
    robotpkg-dynamic-graph-v3=4.4.3
    robotpkg-eigen-quadprog=1.0.1r1
    robotpkg-eiquadprog=1.2.5 
    robotpkg-example-robot-data=4.0.2
    robotpkg-hpp-fcl=2.2.0
    robotpkg-octomap=1.9.6
    robotpkg-pinocchio=2.6.17
    robotpkg-py38-eigenpy=2.9.2
    robotpkg-py38-hpp-fcl=2.2.0
    robotpkg-py38-pinocchio=2.6.17 
    robotpkg-py38-example-robot-data=4.0.7
    robotpkg-py38-crocoddyl=1.9.0r1
    robotpkg-py38-dynamic-graph-v3=4.0.11
    robotpkg-py38-quadprog=0.1.6r1 
    ``` 
Installation
```bash
    sudo apt install -y robotpkg-dynamic-graph-v3=VERSION
``` 

3. Installation for MIM
    A) Installing Treep
    ```bash
        sudo pip3 install treep
    ``` 
    B) Installing colcon
    ```bash
        pip install -U colcon-common-extensions pytest setuptools
    ``` 
    C) Machines in Motion (MIM) Github repository 
    https://github.com/machines-in-motion

    D) How to clone with treep:
    ```bash
        mkdir devel 
        cd  devel
        git clone git@github.com:machines-in-motion/treep_machines_in_motion.git
        treep --clone  "Packagename"
    ``` 
    clone the following using treep 
    ```bash
        treep --clone mim_control
        treep --clone biconvex_mpc
        treep --clone mim_data_utils
        treep --clone mpi_cmake_modules
        treep --clone 
    ```
       
    once these are cloneed using trep, clone this repository by making a directory temp to download the robot_go1_properties and examples to run the robot which are moved to devel/worskapce/src later

    




    E) How to build a workspace
    ```bash
        cd devel/workspace
        colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select
    ```

    F) Put followings in setup.bash
    ```bash
        # -------------------- Source ROS2 -------------------- #
        source /opt/ros/foxy/setup.bash
        source /opt/openrobots/setup.bash 
        source PATH-To/devel/workspace/install/setup.bash

        # --------------- Path for BiConvex MPC --------------- #
        export PYTHONPATH=$PYTHONPATH:PATH-To/devel/workspace/src/biconvex_mpc

        # --------------- Path for openrobots ----------------- #
        export PATH=/opt/openrobots/bin:$PATH
        export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
        export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
        export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH

        export CMAKE_PREFIX_PATH=/opt/openrobots
        export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH
    ```


in opt/opernrobots/setup.bash update
```
#! /bin/bash
export PATH="/opt/openrobots/bin:$PATH"
export PKG_CONFIG_PATH="/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="/opt/openrobots/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/openrobots/lib/dynamic-graph-plugins:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/openrobots/lib/python3.6/site-packages:$PYTHONPATH"
export ROS_PACKAGE_PATH="/opt/openrobots/share:$ROS_PACKAGE_PATH"
```
