import { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please choose an image first.");
      return;
    }

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/extract", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Upload failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#15141f] p-8">
      <div className="mx-auto max-w-3xl">
        <h1 className="text-3xl font-bold mb-4">IDExtract</h1>
        <p className="text-white-600 mb-6">
          Upload an image and extract text from it.
        </p>

        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="mb-4 block w-full"
        />

        <button
          onClick={handleUpload}
          className="rounded-xl bg-teal-800 px-5 py-2 text-white hover:bg-teal-900 transition"
          disabled={loading}
        >
          {loading ? "Processing..." : "Extract Info"}
        </button>

        {error && <p className="mt-4 text-red-500">{error}</p>}

        {preview && (
          <div className="mt-6">
            <h2 className="text-xl font-semibold mb-2">Preview</h2>
            <img
              src={preview}
              alt="Preview"
              className="max-h-80 rounded-lg border"
            />
          </div>
        )}

        {result && (
          <div className="mt-6 rounded-xl border p-4">
            <h2 className="text-xl font-semibold mb-2">Results</h2>
            <p><strong>Name:</strong> {result.name|| "Not found"}</p>
            <p><strong>ID Number:</strong> {result.id_number || "Not found"}</p>
            <p><strong>Date of Birth:</strong> {result.date_of_birth || "Not found"}</p>
            <p><strong>Expiry Date:</strong> {result.expiry_date || "Not found"}</p>

            <div className="mt-4">
              <h3 className="font-semibold">Raw OCR Text</h3>
              <pre className="mt-2 p-3 text-sm">
                {result.raw_text || "No text returned."}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
