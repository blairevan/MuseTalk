def build_ping_pong_cycle(items):
    """Return a forward-and-reverse cycle without repeated endpoints."""
    if len(items) <= 1:
        return items.copy()
    return items + items[-2:0:-1]
