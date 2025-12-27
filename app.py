from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ingest_files import ingest_text_to_pinecone
from rag import answer_user_question
from dotenv import load_dotenv
import markdown
import os

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"txt"}

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/info", methods=["GET"])
def upload_notes_page():
    return render_template("index.html")


@app.route("/info/upload", methods=["POST"])
def upload_notes():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    # user_id = request.form.get("user_id")

    # if not user_id:
    #     return jsonify({"error": "Missing user_id"}), 400

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .txt files allowed"}), 400

    filename = secure_filename(file.filename)
    print(filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Read file contents
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Ingest into Pinecone
    num_chunks = ingest_text_to_pinecone(
        text=text,
        filename=filename,
        index=os.getenv("PINECONE_INDEX_NAME"),
        namespace=os.getenv("PINECONE_NAMESPACE")
    )

    return jsonify({
        "status": "success",
        "filename": filename,
        "chunks_added": num_chunks
    })

@app.route("/info/assistant", methods=["GET"])
def assistant_page():
    return render_template("assistant.html")

@app.route("/info/ask", methods=["POST"])
def ask_question():
    question = request.form.get("question")

    if not question:
        return "Missing question or user_id", 400

    result = answer_user_question(
        question=question,
        index=os.getenv("PINECONE_INDEX_NAME"),
        namespace=os.getenv("PINECONE_NAMESPACE")
    )

    answer_html = markdown.markdown(
        result["answer"],
        extensions=["fenced_code", "tables"]
    )
    return render_template(
        "assistant.html",
        answer=answer_html,
        source=result["sources"],
    )


if __name__ == "__main__":
    app.run(debug=True)