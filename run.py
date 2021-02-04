from app.main import multiThreadRequest
from app import create_app


if __name__ == "__main__":
    app = create_app('default')
    app.run(host='0.0.0.0', port=5001, debug=True)

