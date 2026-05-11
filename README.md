# Scalable LiDAR: DMD–PLM Hybrid Beam Steering System

This repository documents a scalable hybrid LiDAR beam steering system based on a Digital Micromirror Device (DMD) and a Phase Light Modulator (PLM). The project focuses on programmable diffractive beam steering, computer-generated holography (CGH), phase-profile visualization, angular-resolution analysis, and time-of-flight distance validation across multiple diffraction orders.

The system combines the fast binary modulation capability of a DMD with the fine phase-control capability of a PLM. This hybrid architecture enables coarse and fine optical beam steering without relying on large mechanical scanning components.

---

## Project Overview

The DMD–PLM hybrid LiDAR system is designed to demonstrate a solid-state optical beam steering approach for scalable LiDAR applications.

In this architecture:

- The **DMD** provides binary spatial modulation, diffraction-order selection, and coarse field-of-view steering.
- The **PLM** provides calibrated phase modulation and fine sub-field-of-view beam steering.
- **CGH patterns** are generated to steer the beam into different angular directions.
- **Angular resolution** and **distance measurement accuracy** are analyzed across multiple diffraction orders.

This project includes MATLAB and Python scripts for generating phase profiles, binary CGH patterns, angular-resolution plots, and measured-distance validation plots.

---

## System Concept

A simplified transmitter-side optical path is:

```text
Laser source → Beam shaping optics → DMD → 4f relay optics → PLM → Projection optics → Target
