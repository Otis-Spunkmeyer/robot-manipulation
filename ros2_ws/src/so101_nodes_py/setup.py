from setuptools import find_packages, setup

package_name = "so101_nodes_py"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Alex",
    maintainer_email="alex@example.com",
    description="SO-101 Python nodes: joint_wave publisher and static world broadcaster.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "joint_wave = so101_nodes_py.joint_wave:main",
            "static_world = so101_nodes_py.static_world:main",
        ],
    },
)
