"""resources

Define the resource files present in this package.

License: BSD 3-Clause License
Copyright (C) 2018-2021, New York University , Max Planck Gesellschaft
Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

from pathlib import Path


class Resources(object):
    def __init__(self, robot_name) -> None:
        super().__init__()

        self.package_path = str(Path(__file__).resolve().parent.absolute())

        self.robot_name = str(robot_name)

        self.resources_dir = Path(self.package_path) / "resources"

        self.urdf_path = str(
            self.resources_dir / "urdf" / ("go1_minimal" + ".urdf")  # go1_minimal previously
        )
        
        self.meshes_path = str(Path(self.package_path).parent)

        self.imp_ctrl_yaml_path = str(
            self.resources_dir / "impedance_ctrl.yaml"
        )

        self.resources_dir = str(self.resources_dir)