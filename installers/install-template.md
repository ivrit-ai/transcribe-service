# Transcribe Service Installation Template

**This document is the source of truth for the installation process.**

All platform-specific installers must implement the flow defined here. Any changes to the installation process should be:
1. Documented in this template first
2. Propagated to all downstream installers:
   - `installers/osx/install-osx.sh`
   - `installers/windows/install-windows.ps1`
   - `installers/linux/install-linux.sh` (when created)

This document describes the standard installation process for the Transcribe Service across all platforms. Each platform-specific installer implements these same steps using platform-appropriate tools and conventions.

## Overview

The installer downloads and sets up a complete, self-contained transcription service that includes:
- Python 3.13 runtime environment
- All required dependencies
- The Whisper GGML model for transcription
- Application launcher

## Installation Steps

### Step 0: Pre-Installation Checks

**Architecture Validation**
- Verify the system architecture is supported
  - OSX: arm64 (Apple Silicon) only
  - Windows: x86_64 (64-bit) only
  - Linux: x86_64 (64-bit) or arm64

**Existing Installation Detection**
- Check for existing installation directories:
  - `transcribe-service/`
  - `uv/`
  - `bin/`
  - `venv/`
- If found, prompt user to uninstall and reinstall
- Clean up existing installation if user confirms

**Repository and Version Selection**
- Parse `REPO_PATH` environment variable (format: `org/repo/ref-name`)
- Default to latest release from `ivrit-ai/transcribe-service` if not specified
- Detect if reference is a tag or branch via GitHub API
- Support formats:
  - `latest` - fetches latest release
  - Tag reference - specific version
  - Branch reference - development branch

### Step 1: Download and Install UV

**Purpose**: UV is a fast Python package installer and environment manager

**Actions**:
- Create `uv/` directory in installation location
- Download UV from GitHub releases (latest version)
  - OSX: `uv-aarch64-apple-darwin.tar.gz`
  - Windows: `uv-x86_64-pc-windows-msvc.zip`
  - Linux: `uv-x86_64-unknown-linux-gnu.tar.gz` or `uv-aarch64-unknown-linux-gnu.tar.gz`
- Extract to `uv/` directory
- Make executable (Unix-like systems)

**Platform URLs**:
```
OSX:     https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz
Windows: https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
Linux:   https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
```

### Step 2: Download Transcribe Service

**Purpose**: Get the application source code

**Actions**:
- Determine download URL based on reference type:
  - Latest release: Query GitHub API for latest release tarball
  - Tag: `https://api.github.com/repos/{org}/{repo}/tarball/refs/tags/{tag}`
  - Branch: `https://api.github.com/repos/{org}/{repo}/tarball/{branch}`
- Download tarball from GitHub
- Extract to `transcribe-service/` directory (strip top-level directory)
- Fetch commit hash for version tracking:
  - Tags: Get commit hash from tag reference
  - Branches: Get latest commit hash
  - Truncate to 8 characters for brevity
- Create `VERSION` file with format: `{org}/{repo}/{ref}@{commit-hash} ({type})`

### Step 3: Create Virtual Environment

**Purpose**: Isolated Python environment for the application

**Actions**:
- Use UV to create virtual environment: `uv venv venv/ --python 3.13`
- Virtual environment location: `venv/` directory
- Python version: 3.13 (specified explicitly)

**Python Executable Paths**:
- Unix: `venv/bin/python`
- Windows: `venv\Scripts\python.exe`

### Step 4: Install Requirements

**Purpose**: Install core application dependencies

**Actions**:
- Use UV to install from requirements.txt: `uv pip install --python {python} -r transcribe-service/requirements.txt`
- This installs the base application dependencies

### Step 5: Install ivrit[all]

**Purpose**: Install the ivrit transcription library with all optional features

**Actions**:
- Use UV to install: `uv pip install --python {python} "ivrit[all]"`
- The `[all]` extra includes:
  - imageio-ffmpeg (for audio processing)
  - All optional transcription features

### Step 6: Setup FFmpeg

**Purpose**: Make ffmpeg accessible to the application

**Actions**:
- Create `bin/` directory
- Get ffmpeg path from imageio-ffmpeg: `python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`
- Make ffmpeg available:
  - Unix: Create symlink in `bin/` pointing to ffmpeg
  - Windows: Copy ffmpeg.exe to `bin/` (symlinks require admin privileges)

**FFmpeg Paths**:
- Source: From imageio-ffmpeg package
- Destination: `bin/ffmpeg` (Unix) or `bin\ffmpeg.exe` (Windows)

### Step 7: Setup Data Directory and Download Model

**Purpose**: Create user data directory and download the transcription model

**Actions**:
- Create data directory:
  - OSX: `$HOME/Library/ivrit.ai/transcribe-service`
  - Windows: `%APPDATA%\ivrit.ai\transcribe-service`
  - Linux: `$HOME/.local/share/ivrit.ai/transcribe-service`
- Create models directory: `models/` in installation directory
- Download model if not present:
  - URL: `https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ggml/resolve/main/ggml-model.bin`
  - Destination: `models/ivrit-ggml-model.bin`
  - Size: Large file, show progress
- If model exists, prompt to re-download (optional)

### Step 8: Create Application Launcher

**Purpose**: Provide easy way to launch the application

**Platform-Specific Implementation**:

**OSX**:
- Create `.app` bundle at `~/Applications/ivrit.ai.app`
- Bundle structure:
  - `Contents/Info.plist` - app metadata
  - `Contents/MacOS/ivrit.ai` - launcher script
- Launcher script based on `installers/osx/launch.sh`:
  - Patch with absolute paths to installation
  - Set DATA_DIR to absolute path
- Set application icon from `static/favicon.png`
- Bundle launches the service and opens browser

**Windows**:
- Create launcher script: `launch.ps1` (copied from `installers/windows/launch.ps1`)
- Create batch file wrapper: `ivrit.ai.bat`
- Create desktop shortcut: `Desktop\ivrit.ai Transcribe.lnk`
  - Target: PowerShell with launch script
  - Icon: `static/favicon.ico`
  - Hidden window mode

**Linux**:
- Create launcher script: `launch.sh` (copied from `installers/linux/launch.sh`)
- Make executable
- Create desktop entry: `~/.local/share/applications/ivrit-ai-transcribe.desktop`
- Set icon from `static/favicon.png`

## Launcher Behavior

All launchers implement the same behavior:

1. **Check for Running Instance**
   - Look for PID file in data directory
   - If found and process is running, open browser and exit
   - If found but process is dead, remove stale PID file

2. **Environment Setup**
   - Add `bin/` to PATH for ffmpeg access
   - Set working directory to application directory

3. **Launch Application**
   - Activate virtual environment (Unix) or use full path to Python (Windows)
   - Run: `python app.py --local --data-dir {data-dir} --models-dir {models-dir}`
   - Redirect output to log files in data directory:
     - `app.log` - application output
     - `launch.log` - startup errors
   - Save PID to PID file
   - Run in background/detached mode

4. **Wait for Startup**
   - Poll `http://localhost:4500` for up to 20 seconds
   - Check if process is still alive during wait
   - If server doesn't respond, show error and log location

5. **Open Browser**
   - Launch default browser to `http://localhost:4500`
   - Display success message and PID for reference

## Installation Output

All installers should display:

**During Installation**:
- Progress for each step (1/8, 2/8, etc.)
- Download progress for large files
- Success/error status for each operation

**On Completion**:
- Installation directory
- Models directory
- Data directory
- Launcher location
- How to start the service
- Installation log location

## Files Created

```
{install-dir}/
├── uv/                      # UV package manager
│   └── uv[.exe]
├── bin/                     # FFmpeg binary
│   └── ffmpeg[.exe]
├── transcribe-service/      # Application source code
│   ├── app.py
│   ├── requirements.txt
│   ├── static/
│   └── ...
├── venv/                    # Python virtual environment
│   ├── bin/python          # Unix
│   └── Scripts/python.exe  # Windows
├── models/                  # ML models
│   └── ivrit-ggml-model.bin
├── VERSION                  # Version information
└── install.log             # Installation log

{data-dir}/                 # User data directory
├── app.log                 # Application log
├── launch.log              # Launch errors
└── app.pid                 # Process ID file
```

## Configuration

All installers use these default values (customizable via script variables):

| Setting | Value |
|---------|-------|
| Python Version | 3.13 |
| Default Repository | ivrit-ai/transcribe-service |
| Default Reference | latest release |
| Server Port | 4500 |
| Startup Timeout | 20 seconds |
| Model | whisper-large-v3-turbo-ggml |

## Error Handling

All installers should:
- Validate prerequisites before starting
- Provide clear error messages with context
- Log all output for debugging
- Clean up on failure where appropriate
- Avoid leaving partial installations
- Prompt before overwriting existing installations

## Notes

- All installers create self-contained installations
- No system-wide changes except launcher shortcuts
- Uninstallation is simple directory deletion
- Multiple installations can coexist in different directories
- Installation can be relocated by updating launcher paths
