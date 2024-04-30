from distutils.core import setup

setup(
    name='diffusion_policy',
    version='0.0.1',
    packages=['diffusion_policy'],
    long_description=open('README.md').read(),
    install_requires=[
        "tqdm",
        "diffusers",
        "hydra-core",
        "h5py",
        "matplotlib",
        "numpy",
        "imageio[ffmpeg]"
    ]
)
