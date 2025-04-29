.. Distributed Multi Agent LLMs documentation master file, created by
   sphinx-quickstart on Tue Apr 29 01:26:12 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Distributed Multi Agent LLMs documentation
==========================================

This project implements a fault-tolerant distributed multi-agent LLM system that enables multiple language models to collaboratively process user queries and produce a consensus response. It features a FastAPI backend for coordinating agents, asynchronous job management, result aggregation, and detailed sentiment, emotion, and similarity analysis with visualization outputs. A smart load balancer ensures robust, retry-capable request routing, while an SQLite database maintains user accounts, interaction histories, and analysis results. The Gradio-based client interface supports login, query submission, ethical role assignment, model selection, and interaction history viewing and deletion. Designed for resilience, transparency, and user customization, the system can be used for applications such as ethical reasoning debates, policy recommendation systems, collaborative decision support, AI model evaluation, or any domain where combining multiple LLM opinions into a robust consensus is valuable.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
