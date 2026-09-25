from _bootstrap import *  # noqa: F401,F403

from sybil_shield.api.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sybil_shield.api.app:app", host="0.0.0.0", port=5000, reload=False)
