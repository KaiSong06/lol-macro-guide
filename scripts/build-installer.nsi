!define APP_NAME "lol-macro-guide"
!define COMPANY_NAME "lolcoach"
!define VERSION "0.1.0"

Name "${APP_NAME}"
OutFile "dist\lol-macro-guide-setup.exe"
InstallDir "$LOCALAPPDATA\${APP_NAME}"
RequestExecutionLevel user

Page directory
Page instfiles

Section "Install"
  SetOutPath "$INSTDIR"
  File "dist\lol-macro-guide.exe"
  File "config.yaml.example"
  CreateShortcut "$SMPROGRAMS\${APP_NAME}.lnk" "$INSTDIR\lol-macro-guide.exe"
  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

Section "Uninstall"
  Delete "$SMPROGRAMS\${APP_NAME}.lnk"
  Delete "$INSTDIR\lol-macro-guide.exe"
  Delete "$INSTDIR\config.yaml.example"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"
SectionEnd
