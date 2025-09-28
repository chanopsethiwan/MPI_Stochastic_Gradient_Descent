# 🚀 MPI Setup Across Multiple Macs

## 0. Prerequisites (on all Macs) 
* Ensure each Mac is on the **same local network** (same wifi)
* Pick a **shared project folder** (outside iCloud for fewer sync issues) 

* Install ***openiMPI***:
```bash
brew install open-mpi
```

* Install ***mpi4py*** inside your Python environment:
```bash
pip install mpi4py
```

## 1. Verify MPI Installation
* On each Mac:
```bash
which mpiexec
mpiexec --version
```

* Should see something like:
```swift
/opt/homebrew/bin/mpiexec
mpiexec (Open MPI) 5.0.8
```

## 2. SSH Setup (from Master -> Workers)

### 2.1: Generate an SSH key (on Master only)

```bash
ssh-keygen -t rsa -b 4096
```

* Press Enter for defaults (stored in ~/.ssh/id_rsa)
* Press Enter for no passphrase

### 2.2: Copy the key to each Worker
```bash
ssh-copy-id <username>@192.168.1.25   # Worker 1
ssh-copy-id <username>@192.168.1.26   # Worker 2
```

* Replace <username> with the macOS username on each worker (check with `whoami`)

### 2.3: Test passwordless login
```bash
ssh <username>@192.168.1.25 hostname
ssh <username>@192.168.1.26 hostname
```
* They should return hostnames without asking for a password. 

## 3. Hostfile configuration
* On the Master, create a file hosts.txt
```txt
192.168.1.23 slots=2   # Master Mac
192.168.1.25 slots=2   # Worker 1
192.168.1.26 slots=2   # Worker 2
```
* Replace with the actual IPs of the Macs (ifconfig | grep inet).
* `slots=N` means how many MPI process can run on that machine

## 4. Run from Master Mac: 
```bash
mpiexec -n 3 python main_simple.py
```

* 🔑 Important: Every Mac must have the same files inside the same directory (same names and same relative paths). Otherwise, mpiexec will fail when trying to launch the program.
