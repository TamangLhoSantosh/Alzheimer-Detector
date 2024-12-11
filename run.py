import uvicorn

# The block below runs the FastAPI app using Uvicorn when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)
