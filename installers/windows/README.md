# Windows Installer for ivrit.ai Transcribe Service

This directory contains the Windows installer for the ivrit.ai Transcribe Service. All installers implement the installation flow documented in `../install-template.md`.

## Installation Options

There are two ways to install on Windows:

### Option 1: PowerShell Script (Recommended for Quick Install)

This method downloads and runs a PowerShell script that performs the installation automatically.

**Install latest release:**
```powershell
iwr -useb https://raw.githubusercontent.com/ivrit-ai/transcribe-service/main/installers/windows/install-windows.ps1 | iex
```

**Install from a specific branch or tag:**
```powershell
$env:REPO_PATH="ivrit-ai/transcribe-service/v1.0.0"
iwr -useb https://raw.githubusercontent.com/$env:REPO_PATH/installers/windows/install-windows.ps1 | iex
```

**Advantages:**
- Quick and simple
- Single command execution
- Can specify branch or tag easily
- No additional software needed

### Option 2: InnoSetup Installer (Recommended for Distribution)

This method uses InnoSetup to create a traditional Windows installer executable that wraps the PowerShell installer.

**Building the installer:**
1. Install [Inno Setup 6.0+](https://jrsoftware.org/isinfo.php)
2. Open `install-windows.iss` in Inno Setup Compiler
3. Click "Build" to compile
4. The installer will be created as `Output/ivrit-ai-transcribe-setup.exe`

**Using the installer:**
1. Double-click `ivrit-ai-transcribe-setup.exe`
2. Follow the installation wizard
3. The PowerShell installer will run and show progress
4. Desktop shortcuts will be created automatically

**Advantages:**
- Professional Windows installer experience
- GUI-based installation wizard
- Automatic Start Menu and Desktop shortcuts
- Easier for non-technical users
- Can be distributed as a single executable
- Uses the same PowerShell installer script internally

## Files

- `install-windows.ps1` - Standalone PowerShell installer script
- `install-windows.iss` - InnoSetup installer script (self-contained, no external dependencies)
- `README.md` - This file

## What Gets Installed

Both installation methods create the same installation:

```
{InstallDir}/
├── uv/                      # UV package manager
├── bin/                     # FFmpeg binary
├── transcribe-service/      # Application source
├── venv/                    # Python virtual environment
├── models/                  # ML models
├── launch.ps1              # Launcher script
├── launch.bat              # Batch wrapper for launcher
└── VERSION                 # Version information

%APPDATA%\ivrit.ai\transcribe-service/
├── app.log                 # Application log
├── launch.log              # Launch errors
└── app.pid                 # Process ID file
```

## Launching the Service

After installation, you can start the service:

1. **Desktop Shortcut:** Double-click "ivrit.ai Transcribe" on your desktop
2. **Start Menu:** Search for "ivrit.ai Transcribe Service" in Start Menu
3. **Command Line:** Run `launch.bat` from the installation directory

The service will:
- Start in the background
- Check if already running (and reuse if so)
- Wait for the server to be ready
- Automatically open your browser to `http://localhost:4500`

## Uninstallation

### For PowerShell Installation:
1. Delete the installation directory
2. Delete `%APPDATA%\ivrit.ai\transcribe-service`
3. Remove the desktop shortcut (if created)

### For InnoSetup Installation:
1. Use "Add or Remove Programs" in Windows Settings
2. Find "ivrit.ai Transcribe Service" and click Uninstall

## Requirements

- Windows 10 or later (64-bit)
- Internet connection (for downloading dependencies and model)
- Approximately 2GB of free disk space

## Troubleshooting

**Installation fails:**
- Check the `install.log` file in the installation directory
- Ensure you have internet connectivity
- Try running PowerShell as Administrator

**Service won't start:**
- Check `%APPDATA%\ivrit.ai\transcribe-service\launch.log` for errors
- Ensure port 4500 is not already in use
- Try running `launch.bat` directly to see error messages

**Model download is slow:**
- The model file is ~1.5GB, download time depends on your connection
- The download can be resumed if interrupted

## Architecture

**PowerShell Installer (`install-windows.ps1`):**
- Contains all installation logic
- Can be run standalone or via InnoSetup
- Follows the 8-step process documented in `../install-template.md`

**InnoSetup Wrapper (`install-windows.iss`):**
- Provides GUI installer experience
- Embeds and invokes `install-windows.ps1`
- Adds Start Menu and Desktop shortcuts
- Packages everything into a single `.exe`

Both installation methods execute the same PowerShell script, ensuring consistent behavior whether you use the command-line or GUI installer.

## Installation Process

Both methods follow the same 8-step process:
1. Download and install UV package manager
2. Download transcribe-service from GitHub
3. Create Python 3.13 virtual environment
4. Install Python requirements
5. Install ivrit[all] package
6. Setup FFmpeg
7. Download transcription model
8. Create launcher scripts and shortcuts

See `../install-template.md` for detailed documentation of the installation process.
