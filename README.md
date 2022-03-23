H2SurfaceMatching
=========

Description
-----------

This Python package provides a set of tools for the comparison, matching and interpolation of triangulated surfaces within the elastic shape analysis setting. It allows specifically to solve the geodesic matching and distance computation problem between two surfaces with respect to a second order Sobolev metric.

References
------------



Dependencies
------------

SRNFmatch is entirely written in Python while taking advantage of parallel computing on GPUs through CUDA. 
For that reason, it must be used on a machine equipped with an NVIDIA graphics card with recent CUDA drivers installed.
The code involves in addition the following Python libraries:

* Numpy 1.19.2 and Scipy 1.6.2
* Pytorch 1.4
* PyKeops 1.5 (https://www.kernel-operations.io/keops/index.html)
* Open3D 0.12.0

Note that Open3d is primarily used for surface reading, saving, visualization and simple mesh processing operations (decimation, subdivision...). Other libraries such as PyMesh could be used as potential replacement with relatively small modifications to our code.  


Usage
-----



Licence
-------

This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see http://www.gnu.org/licenses/.


Contacts
--------
Emmanuel Hartman

