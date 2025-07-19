from flask import Flask, render_template, request, jsonify
from model import get_response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# Backend API to link the input from the website to the RAG pipeline
@app.route("/query/", methods=["POST"])
def query_model():
    user_prompt = request.json.get("prompt")
    response = get_response(user_prompt) if user_prompt else ""
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)