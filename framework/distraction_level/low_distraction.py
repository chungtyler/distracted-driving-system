# Import PIL?

# Path to indicator files

_last_state = None

def show_indicator(active: bool):
    global _last_state
    if active == _last_state:
        return

    _last_state = active

    if active:
        print("\033[91m⚠️  WARNING: PAY ATTENTION TO THE ROAD!\033[0m")
    else:
        print("\033[92m✔️ Attention normal.\033[0m")
