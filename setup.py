from setuptools import setup
setup(
    name='SurfaceMatch',
    version='1.0',
    author='Emmanuel Hartman, Martin Bauer, Yashil Sukurdeep, and Nicolas Charon',
    description='his Python package provides a set of tools for the comparison, matching and interpolation of triangulated surfaces within the elastic shape analysis setting.',
    long_description='This Python package provides a set of tools for the comparison, matching and interpolation of triangulated surfaces within the elastic shape analysis setting. It allows specifically to solve the geodesic matching and distance computation problem between two surfaces with respect to a second order Sobolev metric. In addition to basic shape matching, we develop a comprehensive statistical pipeline that allows for the computation of the Karcher means, tangent space principal component analysis, and motion transfer in the space of parametrized surfaces and in shape space. Thus, our framework is equipped to handle statistical analysis of populations of shapes. Further to improve the robustness of our model, we implement a weighted varifold matching framework for partial matching. By implementing partiality in our methods we allow for the analysis of shape populations that a include surfaces that are noisy or are missing data. Further, this adaptation allows for a more natural comparison of shapes with different mesh structures and even allows for the comparison of shapes with different topologies.',
    url='https://github.com/emmanuel-hartman/H2_SurfaceMatch',
    keywords='Surface, Matching, Registration',
    python_requires='>=3.7, <4',
    install_requires=[
        'torch>=1.13.0',
        'numpy>=1.19.0',
        'scipy >=1.6.0',
        'pykeops>=2.1.1',
        'open3d'
    ]
)