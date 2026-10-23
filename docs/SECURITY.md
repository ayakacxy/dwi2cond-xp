# Security policy

[English](en/SECURITY.md) | [简体中文](zh/SECURITY.md)

[Documentation](en/README.md) · [Project home](../README.md)

Security fixes target the latest release and the `main` branch.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/ayakacxy/dwi2cond-xp/security/advisories/new) for security defects. Include the affected version, reproduction steps, impact and any proposed mitigation. Avoid public disclosure of an unpatched vulnerability.

Use synthetic inputs. Remove credentials, identifying information and controlled-access medical data from attachments.

## Scope

The package reads diffusion images, tensors, meshes and configuration files, and can invoke an installed SimNIBS solver. It does not upload input data or collect telemetry. Unexpected numerical results belong in a bug report unless they arise from a security defect. See [Support](en/SUPPORT.md).
