![](https://img.shields.io/badge/SDK-v14.0.0-blue) <Please check version is the same as specified in requirements.txt>

# VIKTOR interaction with RFEM

This sample application is made to showcase the interaction between VIKTOR and RFEM. It builds a model representing a truss, uses a VIKTOR worker running in an environment where RFEM is running with its web service enabled and visualizes the calculations inside the VIKTOR interface.

## RFEM 6

This application uses [RFEM](https://www.dlubal.com/en). RFEM needs to be running in the same environment as the worker described below. Make sure you have enabled WebServices in the RFEM program settings as described [here](https://github.com/Dlubal-Software/RFEM_Python_Client).

## The Worker

To interact with RFEM using the VIKTOR application we use the VIKTOR worker. Some steps need to be taken to make it work. These steps are described [here](app/lib/README.md).