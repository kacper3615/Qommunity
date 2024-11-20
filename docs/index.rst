.. Qommunity documentation master file, created by
   sphinx-quickstart on Tue Nov 19 22:53:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Qommunity documentation!
===================================
The Qommunity library was designed to integrate several methods for detecting communities in graphs in one place. 
The implemented methods rely on classical, quantum, and hybrid solutions. 
The idea was to simplify and standardize the interface for conducting experiments.

Architecture
============
Several components compartmentalize the architecture, offering user-friendly features: Samplers, Searchers, and Iterative Searchers


.. toctree::
   :maxdepth: 1
   :caption: Architecture:

   samplers
   searchers
   iterative_searcher

.. raw:: html

   <img src="_static/qommunity_architecture.svg">