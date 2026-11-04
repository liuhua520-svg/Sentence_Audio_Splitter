pyinstaller --noconfirm --clean ^
  --name "SentenceAudioSplitter" ^
  --windowed ^
  --onefile ^
  --hidden-import=timeline_core ^
  --hidden-import=sounddevice ^
  --hidden-import=soundfile ^
  --hidden-import=numpy ^
  --collect-all sounddevice ^
  --collect-all soundfile ^
  splitter_gui.py