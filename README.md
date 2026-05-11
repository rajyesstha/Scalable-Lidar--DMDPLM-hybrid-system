# Scalable LiDAR: DMD–PLM Hybrid Beam Steering System

[![MATLAB](https://img.shields.io/badge/MATLAB-75.4%25-orange)](#)
[![Python](https://img.shields.io/badge/Python-24.6%25-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains MATLAB and Python code used to develop and analyze a scalable hybrid LiDAR beam steering system based on a Digital Micromirror Device (DMD) and a Phase Light Modulator (PLM). The project focuses on programmable diffractive beam steering, computer-generated holography (CGH), phase-profile visualization, angular-resolution analysis, diffraction-efficiency analysis, and time-of-flight distance validation across multiple diffraction orders.

The DMD provides fast binary spatial modulation and diffraction-order based field-of-view control, while the PLM provides calibrated phase modulation for fine angular steering. Together, the DMD and PLM enable a hybrid solid-state LiDAR architecture that can support sub-megapixel steering and can be scaled toward megapixel-class spatial addressing.

---

## Project Motivation

Mechanical LiDAR scanners can provide wide field-of-view coverage, but they often introduce limitations in scan speed, vibration tolerance, reliability, and system compactness. This project investigates a programmable optical beam steering architecture that reduces dependence on large mechanical scanning components.

The main idea is to combine:

- **DMD-based coarse steering** through binary diffractive modulation.
- **PLM-based fine steering** through calibrated phase modulation.
- **CGH-based beam control** for programmable steering directions.
- **Experimental validation** using angular-resolution and distance-measurement analysis.

---

## System Concept

A simplified transmitter-side optical path is:

```text
Laser source → Beam shaping optics → DMD → 4f relay optics → PLM → Projection optics → Target
