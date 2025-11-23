# framework/distraction_level/low_distraction.py

def show_indicator(active: bool):
    """
    When active=True → print a warning message.
    When active=False → print nothing.
    """
    if active:
        print("⚠️  PLEASE PAY ATTENTION TO THE ROAD  ⚠️")
