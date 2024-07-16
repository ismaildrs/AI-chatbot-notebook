from flask import Flask, request, jsonify
from model import AiResponse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all route

@app.route('/question', methods=['POST'])
def post_route():
    # Ensure the Content-Type is application/json
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    
    # Extract the 'question' field from the request
    data = request.get_json()
    question = data.get('question')

    # Check if the 'question' field is provided
    if not question:
        return jsonify({"error": "No question field provided"}), 400

    answer = AiResponse(question=question)

    # Perform some operations with the 'question' field
    response_data = {
        "message": "Question received",
        "answer": answer
    }

    # Return a JSON response
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
