from fastapi import FastAPI
#uvicorn main:app --reload

user_db = {
    "alice": {"name": "Alice", "age": 30},
    "bob": {"name": "Bob", "age": 25},
}


app = FastAPI()


@app.get("/")
def my_second_fastapi():
    return {"message": "Hello, FastAPI!"}

@app.get("/users/{username}")
def get_users_path(username: str):
    return user_db[username]