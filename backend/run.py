import uvicorn

if __name__ == "__main__":
    # 使用 uvicorn 运行 app.main 模块中的 app 实例
    # reload=True 开启热重载，方便开发
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
