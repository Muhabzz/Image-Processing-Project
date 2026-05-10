# Vulnerability-Based Network Isolation

## Overview

This project is a **Zero-Trust network security automation tool** designed to identify vulnerable servers and automatically isolate them from the production network. The script performs a security scan on a target server using **Nmap** to detect dangerous or insecure open ports such as:

* **21** → FTP
* **23** → Telnet
* Other configurable high-risk services

If any vulnerable ports are detected, the system automatically connects to a network switch using **Paramiko (SSH)** and moves the affected device’s switch port into a **Quarantine VLAN (VLAN 999)** to prevent potential lateral movement or unauthorized access.

All actions and detected threats are logged for auditing and incident response purposes.

---

## Features

* Automated network vulnerability scanning using Nmap
* Detection of insecure or legacy services
* Automatic isolation of compromised or non-compliant devices
* SSH-based switch management with Paramiko
* Quarantine VLAN assignment (VLAN 999)
* Event logging and alert tracking
* Supports Zero-Trust security principles

---

## Technologies Used

* **Python 3**
* **python-nmap** — For port scanning and vulnerability detection
* **Paramiko** — For SSH communication with network switches

---

## Workflow

1. The script scans a target server using Nmap.
2. It checks for dangerous open ports defined in the configuration.
3. If a risky service is found:

   * The script connects to the network switch via SSH.
   * The corresponding switch port is reassigned to **VLAN 999**.
4. The incident is logged with details such as:

   * Target IP
   * Detected ports
   * Isolation status
   * Timestamp

---

## Example Use Case

A company wants to enforce Zero-Trust policies inside its internal network. If a device exposes insecure services like Telnet or FTP, this automation tool immediately isolates the device to reduce the risk of unauthorized access, malware propagation, or data breaches.

---

## Requirements

Install the required Python packages:

```bash
pip install python-nmap paramiko
```

---

## Future Improvements

* Email/SMS alert integration
* Web dashboard for monitoring isolated devices
* Integration with SIEM platforms
* Support for multiple switch vendors
* Automated recovery after remediation

---

## Security Note

This project is intended for authorized security testing and network administration only. Ensure you have permission before scanning or modifying network infrastructure.
