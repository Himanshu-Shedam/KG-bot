from flask import Flask, render_template, request, jsonify
from PIL import Image
from chat import get_response, get_gemini_response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    file = request.files.get("image")
    text = request.form.get("text")

    if text and not file:
        answer = get_response(text)
        return jsonify({"response": answer})
    
    elif file and not text:
        image = Image.open(file)

        input = "Extract only error text from image."
        response = get_gemini_response(input,image)

        answer = get_response(response)
        print(answer)
        return jsonify({"response": answer})
    
    elif file and text:
        image = Image.open(file)

        input = "Extract only error text from image."
        response = get_gemini_response(input,image)

        query = text + " " + response
        answer = get_response(query)
        return jsonify({"response": answer})
    
    return jsonify({"response": "Please provide a message or an image."})


if __name__ == "__main__":
    app.run(debug=False)
