# Antenna pattern files

Drop `.msi` antenna pattern files (horizontal + vertical principal-plane cuts)
here. They will appear in the antenna dropdown of the web planner's transmitter
editor, and can be loaded in code with:

```python
from wifisim import make_msi_antenna
ant = make_msi_antenna("patterns/your_antenna.msi")
```

Point the app at a different folder with the `WIFISIM_PATTERNS` env var.
