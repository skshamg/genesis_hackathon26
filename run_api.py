from bad_people_finder.api.app import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bad_people_finder.api.app:app", host="0.0.0.0", port=8000, reload=False)
